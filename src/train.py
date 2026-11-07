"""Скрипт высокопроизводительного обучения нейросети DOENetwork (U-Net) на GPU/CUDA

(Milestone M2, Хакатон МИФИ по фотонике).

Архитектура:
- Модель: U-Net (7.85M параметров, 128->64->32->16->8 с боттлнеком 8x8).
- Физический слой: Single-FFT дифракция Френеля на 4 плоскости (24.1, 25.3, 26.5, 27.7 мм), нормировка |K| к максимуму 1.0.
- Функция потерь: CompositeLoss = L1 + (1 - SSIM) + (1 - Pearson CC).
- Аугментация: динамическая случайная группировка (cross-image grouping) из 40 092 патчей + on-the-fly синтетические маски (p_synth = 0.3).
- Кэширование: быстрый оперативный кэш в RAM для исключения задержек дискового ввода-вывода.
- Валидация: регулярный расчет метрик (Pearson CC, SSIM) на тестовых группах из 20 папок (test_dataset).
- Логирование: training_log.csv и training_log.json.
- Сохранение: лучший чекпоинт весов сохраняется в E:\\Фотоника\\code\\doe_network.pth.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

# Пути проекта
SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parent
PHOTONICS_DIR = CODE_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from doe_network import (
    GRID_SIZE,
    PIXEL_SIZE,
    WAVELENGTH,
    Z_DISTANCES,
    CompositeLoss,
    DOEAugmentedDataset,
    DOENetwork,
    FresnelPropagation,
    calculate_metrics,
    generate_synthetic_mask,
)


class FastDOEAugmentedDataset(Dataset):
    """Высокоскоростной датасет для обучения ДОЭ с кэшированием патчей в RAM.
    
    Устраняет узкое место дискового I/O: однажды прочитанные патчи сохраняются в RAM,
    обеспечивая максимальную утилизацию вычислительных ядер GPU (RTX 4060).
    """

    def __init__(
        self,
        folder: Path,
        mode: str = "train",
        p_synth: float = 0.3,
        grid_size: int = GRID_SIZE,
        max_cache_size: int = 15000,
        epoch_length: int = 2400,
    ):
        super().__init__()
        self.mode = mode
        self.p_synth = p_synth
        self.grid_size = grid_size
        self.max_cache_size = max_cache_size
        self.epoch_length = epoch_length
        self.transform = ToTensor()
        self.cache: dict[str, torch.Tensor] = {}

        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Папка датасета не найдена: {self.folder}")

        if self.mode == "test":
            all_groups = sorted(
                p for p in self.folder.glob("group_*") if p.is_dir()
            )
            self.groups = [
                g for g in all_groups if len(list(g.glob("*.png"))) == 4
            ]
            if not self.groups:
                raise FileNotFoundError(f"Тестовые группы не найдены в {self.folder}")
        else:
            self.patch_files: list[Path] = []
            for entry in os.scandir(self.folder):
                if entry.is_dir():
                    for f in os.scandir(entry.path):
                        if f.name.endswith(".png"):
                            self.patch_files.append(Path(f.path))
            if not self.patch_files:
                self.patch_files = list(self.folder.glob("*.png"))
            if not self.patch_files:
                raise FileNotFoundError(f"Патчи изображений не найдены в {self.folder}")

    def __len__(self) -> int:
        if self.mode == "test":
            return len(self.groups)
        return self.epoch_length

    def _get_patch(self, path: Path) -> torch.Tensor:
        key = str(path)
        if key in self.cache:
            return self.cache[key]

        with Image.open(path) as img:
            img_l = img.convert("L")
            if img_l.size != (self.grid_size, self.grid_size):
                img_l = img_l.resize(
                    (self.grid_size, self.grid_size), Image.Resampling.BILINEAR
                )
            t = self.transform(img_l)

        if len(self.cache) < self.max_cache_size:
            self.cache[key] = t
        return t

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.mode == "test":
            files = sorted(self.groups[index].glob("*.png"))
            images = [self._get_patch(p) for p in files]
            return torch.cat(images, dim=0)

        # Cross-image grouping + синтетические маски
        channels = []
        for _ in range(4):
            if random.random() < self.p_synth:
                ch = generate_synthetic_mask(self.grid_size)
            else:
                idx = random.randint(0, len(self.patch_files) - 1)
                ch = self._get_patch(self.patch_files[idx])
            channels.append(ch)

        return torch.cat(channels, dim=0)  # [4, 128, 128]


def get_device(requested_device: str = "auto") -> torch.device:
    """Инициализация вычислительного устройства."""
    if requested_device != "auto":
        dev = torch.device(requested_device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_mem_free = (
            torch.cuda.mem_get_info()[0] / (1024**3)
            if hasattr(torch.cuda, "mem_get_info")
            else gpu_mem_total
        )
        print(
            f"[Устройство] Выбран GPU: {gpu_name} (Всего VRAM: {gpu_mem_total:.2f} GB, Свободно: {gpu_mem_free:.2f} GB)"
        )
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Устройство] Выбран Apple Silicon MPS")
    else:
        dev = torch.device("cpu")
        print("[Устройство] Выбран CPU")
    return dev


def find_dataset_folders() -> tuple[Path, Path]:
    """Поиск директорий обучающего и тестового датасетов."""
    train_candidates = [
        PHOTONICS_DIR / "data" / "processed" / "convertered_dataset",
        CODE_DIR / "data" / "dataset" / "convertered_dataset",
        PHOTONICS_DIR / "data" / "dataset" / "convertered_dataset",
    ]
    train_folder = None
    for p in train_candidates:
        if p.exists() and any(p.iterdir()):
            train_folder = p
            break
    if train_folder is None:
        raise FileNotFoundError(f"Обучающий датасет не найден среди {train_candidates}")

    test_candidates = [
        PHOTONICS_DIR / "data" / "processed" / "test_dataset",
        CODE_DIR / "data" / "dataset" / "test_dataset",
        PHOTONICS_DIR / "data" / "dataset" / "test_dataset",
    ]
    test_folder = None
    for p in test_candidates:
        if p.exists() and any(p.iterdir()):
            test_folder = p
            break
    if test_folder is None:
        raise FileNotFoundError(f"Тестовый датасет не найден среди {test_candidates}")

    return train_folder, test_folder


def evaluate_on_test(
    model: nn.Module,
    propagator: nn.Module,
    criterion: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Оценка модели на тестовом наборе данных (20 фиксированных групп)."""
    model.eval()
    total_val_loss = 0.0
    total_val_l1 = 0.0
    total_val_ssim = 0.0
    total_val_pcc = 0.0
    sample_count = 0

    with torch.no_grad():
        for targets in test_loader:
            targets = targets.to(device)  # [B, 4, 128, 128]
            B = targets.shape[0]

            phase_doe = model(targets)  # [B, 1, 128, 128]
            rec_norm = propagator(phase_doe, normalize=True)  # [B, 4, 128, 128]

            targets_norm = targets / torch.clamp_min(
                targets.amax(dim=(-2, -1), keepdim=True), 1e-12
            )

            loss, l1, ssim_val, pcc_val = criterion(rec_norm, targets_norm)

            total_val_loss += loss.item() * B
            total_val_l1 += l1.item() * B
            total_val_ssim += ssim_val.item() * B
            total_val_pcc += pcc_val.item() * B
            sample_count += B

    return {
        "val_loss": total_val_loss / max(1, sample_count),
        "val_l1": total_val_l1 / max(1, sample_count),
        "val_ssim": total_val_ssim / max(1, sample_count),
        "val_pcc": total_val_pcc / max(1, sample_count),
    }


def train_pipeline(
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    p_synth: float = 0.3,
    steps_per_epoch: int = 150,
    save_path: Path = CODE_DIR / "doe_network.pth",
    log_csv_path: Path = CODE_DIR / "training_log.csv",
    log_json_path: Path = CODE_DIR / "training_log.json",
    requested_device: str = "auto",
):
    """Главный пайплайн обучения модели U-Net до сходимости."""
    device = get_device(requested_device)
    train_folder, test_folder = find_dataset_folders()

    print(f"[Данные] Обучающие патчи: {train_folder}")
    print(f"[Данные] Тестовые группы: {test_folder}")

    epoch_samples = steps_per_epoch * batch_size
    train_dataset = FastDOEAugmentedDataset(
        folder=train_folder,
        mode="train",
        p_synth=p_synth,
        grid_size=GRID_SIZE,
        max_cache_size=15000,
        epoch_length=epoch_samples,
    )
    test_dataset = FastDOEAugmentedDataset(
        folder=test_folder,
        mode="test",
        grid_size=GRID_SIZE,
    )

    print(
        f"[Данные] Доступно обучающих патчей: {len(train_dataset.patch_files):,}, тестовых групп: {len(test_dataset)}"
    )

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=0,
        pin_memory=pin_mem,
    )

    # Модель, дифракционный слой и составная функция потерь
    model = DOENetwork(in_channels=4, base_channels=32).to(device)
    propagator = FresnelPropagation(
        z_list=Z_DISTANCES,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        grid_size=GRID_SIZE,
    ).to(device)
    criterion = CompositeLoss(
        window_size=11, sigma=1.5, eps_pcc=1e-6, w_l1=1.0, w_ssim=1.0, w_pcc=1.0
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Модель] U-Net инициализирован. Параметров: {total_params:,}")

    # Оптимизатор и шедулер
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Оценка до обучения
    print("\n[Валидация] Оценка случайных весов до обучения...")
    init_val = evaluate_on_test(model, propagator, criterion, test_loader, device)
    print(
        f"  Старт: Val Loss={init_val['val_loss']:.4f}, L1={init_val['val_l1']:.4f}, "
        f"SSIM={init_val['val_ssim']:.4f}, Pearson CC={init_val['val_pcc']:.4f}"
    )

    # Инициализация логирования
    history = []
    csv_headers = [
        "epoch",
        "train_loss",
        "train_l1",
        "train_ssim",
        "train_pcc",
        "val_loss",
        "val_l1",
        "val_ssim",
        "val_pcc",
        "lr",
        "epoch_time_sec",
    ]

    with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    best_val_loss = float("inf")
    best_pcc = -1.0
    best_epoch = -1
    total_training_start = time.perf_counter()

    print(
        f"\n[Обучение] Старт: {epochs} эпох, батч {batch_size}, {steps_per_epoch} шагов в эпохе"
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_train_loss = 0.0
        total_train_l1 = 0.0
        total_train_ssim = 0.0
        total_train_pcc = 0.0
        step_count = 0

        current_lr = optimizer.param_groups[0]["lr"]

        for batch_idx, targets in enumerate(train_loader, start=1):
            if batch_idx > steps_per_epoch:
                break

            targets = targets.to(device, non_blocking=True)  # [B, 4, 128, 128]
            optimizer.zero_grad(set_to_none=True)

            predicted_doe = model(targets)  # [B, 1, 128, 128], [-pi, +pi]
            rec_norm = propagator(predicted_doe, normalize=True)  # [B, 4, 128, 128]

            targets_norm = targets / torch.clamp_min(
                targets.amax(dim=(-2, -1), keepdim=True), 1e-12
            )

            loss, l1, ssim_val, pcc_val = criterion(rec_norm, targets_norm)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_l1 += l1.item()
            total_train_ssim += ssim_val.item()
            total_train_pcc += pcc_val.item()
            step_count += 1

            if batch_idx % max(1, steps_per_epoch // 4) == 0 or batch_idx == steps_per_epoch:
                elapsed = time.perf_counter() - epoch_start
                it_per_sec = batch_idx / max(elapsed, 1e-6)
                pct = batch_idx / steps_per_epoch * 100
                print(
                    f"  Эпоха {epoch:2d}/{epochs:2d} | шаг {batch_idx:3d}/{steps_per_epoch} ({pct:5.1f}%) | "
                    f"Loss: {loss.item():.4f} (L1: {l1.item():.4f}, SSIM: {ssim_val.item():.4f}, PCC: {pcc_val.item():.4f}) | "
                    f"{it_per_sec:.1f} батч/с | RAM кэш: {len(train_dataset.cache):,}"
                )

        scheduler.step()
        epoch_time = time.perf_counter() - epoch_start

        avg_train_loss = total_train_loss / max(1, step_count)
        avg_train_l1 = total_train_l1 / max(1, step_count)
        avg_train_ssim = total_train_ssim / max(1, step_count)
        avg_train_pcc = total_train_pcc / max(1, step_count)

        # Валидация на тестовых группах
        val_metrics = evaluate_on_test(model, propagator, criterion, test_loader, device)

        print(
            f"[Эпоха {epoch:2d}/{epochs:2d}] "
            f"Train Loss: {avg_train_loss:.4f} (SSIM: {avg_train_ssim:.4f}, PCC: {avg_train_pcc:.4f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} (SSIM: {val_metrics['val_ssim']:.4f}, PCC: {val_metrics['val_pcc']:.4f}) | "
            f"Время: {epoch_time:.1f}с, LR: {current_lr:.2e}"
        )

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "train_l1": round(avg_train_l1, 5),
            "train_ssim": round(avg_train_ssim, 5),
            "train_pcc": round(avg_train_pcc, 5),
            "val_loss": round(val_metrics["val_loss"], 5),
            "val_l1": round(val_metrics["val_l1"], 5),
            "val_ssim": round(val_metrics["val_ssim"], 5),
            "val_pcc": round(val_metrics["val_pcc"], 5),
            "lr": round(current_lr, 8),
            "epoch_time_sec": round(epoch_time, 2),
        }
        history.append(epoch_data)

        with open(log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch_data[h] for h in csv_headers])

        with open(log_json_path, mode="w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # Сохранение наилучшего чекпоинта по качеству валидации (val_loss и val_pcc)
        if val_metrics["val_loss"] < best_val_loss or val_metrics["val_pcc"] > best_pcc:
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
            if val_metrics["val_pcc"] > best_pcc:
                best_pcc = val_metrics["val_pcc"]
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(
                f"  >>> Рекорд валидации! Чекпоинт сохранен: {save_path} (Val Loss: {val_metrics['val_loss']:.4f}, PCC: {val_metrics['val_pcc']:.4f})"
            )

    total_time = time.perf_counter() - total_training_start
    print(
        f"\n[Обучение успешно завершено] Время обучения: {total_time / 60:.2f} мин. "
        f"Лучшая эпоха: {best_epoch} (Val Loss={best_val_loss:.4f}, PCC={best_pcc:.4f})."
    )
    print(f"Итоговый файл весов: {save_path}")
    print(f"Файлы динамики обучения:\n  - {log_csv_path}\n  - {log_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение U-Net ДОЭ для многоплоскостного волнового фронта"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер батча")
    parser.add_argument("--lr", type=float, default=2e-4, help="Начальная скорость обучения")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay AdamW")
    parser.add_argument("--p_synth", type=float, default=0.3, help="Вероятность синтетических масок")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=150,
        help="Количество шагов оптимизации в эпохе",
    )
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu или auto")
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        p_synth=args.p_synth,
        steps_per_epoch=args.steps_per_epoch,
        requested_device=args.device,
    )
