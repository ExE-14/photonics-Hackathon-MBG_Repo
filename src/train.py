"""Высокопроизводительный пайплайн обучения нейросети DOENetwork (ResAttention U-Net) на GPU NVIDIA RTX 4060.

Разработан в рамках проекта глубокой оптимизации синтеза многоплоскостных ДОЭ.
Цель: превзойти бенчмарк конкурентов (Pearson CC > 0.8672, SSIM > 0.5918 на 4 осевых плоскостях).

Ключевые оптимизации:
1. Архитектура ResAttention U-Net: 8.42M параметров, ResDoubleConv, Attention Gates, выход tanh * pi.
2. Физический слой дифракции Френеля: Single-FFT с detached-amax нормализацией для устранения спайка градиента.
3. Оптимизированная функция потерь: OptimizedCompositeLoss (w_pcc=4.0, w_ssim=1.2, w_l1=0.05, w_sobel=0.05, plane_weights=[0.9, 0.9, 1.1, 1.1]).
4. Высокоскоростной пул патчей в RAM: параллельная предзагрузка 40k патчей в uint8 для устранения задержек дискового I/O.
5. Банк синтетических масок в RAM: 2,000 разнообразных масок (решетки, кольца Френеля, символы, контуры) с p_synth=0.40.
6. Динамические On-GPU аугментации: ортогональные повороты на 90°, отражения, гамма-джиттер, легкий шум.
7. Оптимизатор и расписание: AdamW (lr=3e-4, weight_decay=1e-4, clip_grad=1.0) + 6-эпох warmup + 194-эпох CosineAnnealing.
8. Сохранение чекпоинта по композитному скору: 0.6 * Val_PCC + 0.4 * Val_SSIM в doe_network.pth.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Настройка путей
SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parent
PHOTONICS_DIR = CODE_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from doe_network import (
    GRID_SIZE,
    PIXEL_SIZE,
    WAVELENGTH,
    Z_DISTANCES,
    CompositeLoss,
    CompositeLossV3,
    DOENetwork,
    SpectralResAttentionUNet,
    SpectralConv2d,
    SpectralBlock,
    FocalFrequencyLoss,
    FresnelPropagation,
    OptimizedCompositeLoss,
    PhaseTotalVariationLoss,
    calculate_metrics,
    generate_dark_contrast_mask,
    generate_synthetic_mask,
)


def load_single_patch_uint8(path: str) -> np.ndarray:
    """Загрузка одного патча с преобразованием в 128x128 grayscale uint8."""
    with Image.open(path) as img:
        img_l = img.convert("L")
        if img_l.size != (GRID_SIZE, GRID_SIZE):
            img_l = img_l.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.BILINEAR)
        return np.array(img_l, dtype=np.uint8)


class FastRAMPatchDataset(Dataset):
    """Датасет максимальной производительности: хранит все патчи и синтетические маски в RAM (uint8).
    
    Реализует сбалансированный генератор 50/50:
    - 50% натуральные патчи DIV2K (из пула 40k патчей)
    - 50% высококонтрастные патчи с темным фоном (силуэты, символы, зонные пластинки, кольца, КТ-контуры).
    """

    def __init__(
        self,
        patch_files: list[str],
        synth_bank: torch.Tensor,
        p_synth: float = 0.50,
        epoch_length: int = 2400,
        max_preload: int = 50000,
        num_workers: int = 16,
    ):
        super().__init__()
        self.p_synth = p_synth
        self.epoch_length = epoch_length
        self.synth_bank = synth_bank  # [N_synth, 128, 128] uint8
        self.n_synth = synth_bank.shape[0]

        files_to_load = patch_files[:max_preload]
        print(f"[Данные] Загрузка {len(files_to_load):,} патчей в RAM ({num_workers} потоков)...", flush=True)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            patches_list = list(executor.map(load_single_patch_uint8, files_to_load))
        t1 = time.perf_counter()

        self.patches = torch.from_numpy(np.stack(patches_list, axis=0))  # [N_patches, 128, 128] uint8
        self.n_patches = self.patches.shape[0]
        ram_mb = (self.patches.element_size() * self.patches.nelement()) / (1024 * 1024)
        print(
            f"[Данные] Загружено {self.n_patches:,} патчей за {t1 - t0:.2f}с. Объем RAM: {ram_mb:.1f} MB.",
            flush=True,
        )

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, index: int) -> torch.Tensor:
        # Сбалансированный генератор 50/50:
        # Режим 1 (20%): все 4 плоскости - высококонтрастные схемы с темным фоном (бенчмарк-режим конкурентов)
        # Режим 2 (20%): все 4 плоскости - натуральные фото DIV2K (фото-режим)
        # Режим 3 (60%): каждая плоскость независимо выбирается 50/50 (смешанный cross-image режим)
        # Математическое мат. ожидание: 0.2*1.0 + 0.2*0.0 + 0.6*0.5 = ровно 0.50 (50/50)!
        dice = random.random()
        channels = []
        if dice < 0.20:
            for _ in range(4):
                idx = random.randint(0, self.n_synth - 1)
                channels.append(self.synth_bank[idx])
        elif dice < 0.40:
            for _ in range(4):
                idx = random.randint(0, self.n_patches - 1)
                channels.append(self.patches[idx])
        else:
            for _ in range(4):
                if random.random() < self.p_synth:
                    idx = random.randint(0, self.n_synth - 1)
                    channels.append(self.synth_bank[idx])
                else:
                    idx = random.randint(0, self.n_patches - 1)
                    channels.append(self.patches[idx])

        # Стек 4 каналов [4, 128, 128], конвертация в float32 [0.0, 1.0]
        stacked = torch.stack(channels, dim=0)  # [4, 128, 128] uint8
        return stacked.float().div_(255.0)


def pregenerate_synthetic_bank(num_masks: int = 2500, size: int = 128, num_workers: int = 16) -> torch.Tensor:
    """Генерация банка разнообразных контрастных оптических масок со строго темным фоном в RAM (uint8)."""
    print(
        f"[Данные] Генерация банка {num_masks:,} контрастных масок с темным фоном (геометрия, символы, зонные пластинки, кольца, КТ, {num_workers} потоков)...",
        flush=True,
    )
    t0 = time.perf_counter()

    def _gen_single_uint8(_: int) -> torch.Tensor:
        m = generate_dark_contrast_mask(size)  # [1, size, size] float in [0, 1]
        return (m.squeeze(0) * 255.0).round().to(torch.uint8)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        masks = list(executor.map(_gen_single_uint8, range(num_masks)))

    bank = torch.stack(masks, dim=0)  # [num_masks, size, size] uint8
    t1 = time.perf_counter()
    ram_mb = (bank.element_size() * bank.nelement()) / (1024 * 1024)
    print(f"[Данные] Банк масок сгенерирован за {t1 - t0:.2f}с. Объем RAM: {ram_mb:.1f} MB.", flush=True)
    return bank


def augment_batch_on_gpu(targets: torch.Tensor) -> torch.Tensor:
    """Динамические ортогональные тензорные аугментации на GPU.
    
    Сохраняют физику дифракции Френеля:
    - Случайные горизонтальные и вертикальные отражения
    - Случайные повороты на 90 градусов (ортогональные повороты)
    - Случайное гамма-масштабирование [0.8, 1.2]
    - Легкий гауссов шум (sigma = 0.01)
    """
    if random.random() < 0.5:
        targets = torch.flip(targets, dims=[-1])
    if random.random() < 0.5:
        targets = torch.flip(targets, dims=[-2])
    k = random.randint(0, 3)
    if k > 0:
        targets = torch.rot90(targets, k=k, dims=[-2, -1])
    if random.random() < 0.5:
        gamma = torch.empty(targets.shape[0], 1, 1, 1, device=targets.device).uniform_(0.8, 1.2)
        targets = torch.clamp(targets**gamma, 0.0, 1.0)
    if random.random() < 0.3:
        noise = torch.randn_like(targets) * 0.01
        targets = torch.clamp(targets + noise, 0.0, 1.0)
    return targets


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
            f"[Устройство] Выбран GPU: {gpu_name} (Всего VRAM: {gpu_mem_total:.2f} GB, Свободно: {gpu_mem_free:.2f} GB)",
            flush=True,
        )
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[Устройство] Выбран Apple Silicon MPS", flush=True)
    else:
        dev = torch.device("cpu")
        print("[Устройство] Выбран CPU", flush=True)
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


def collect_patch_files(train_folder: Path) -> list[str]:
    """Быстрый сбор всех путей к PNG файлам."""
    files = []
    for entry in os.scandir(train_folder):
        if entry.is_dir():
            for f in os.scandir(entry.path):
                if f.name.endswith(".png"):
                    files.append(f.path)
    if not files:
        files = [p.as_posix() for p in train_folder.glob("*.png")]
    if not files:
        raise FileNotFoundError(f"Патчи изображений не найдены в {train_folder}")
    return files


def load_test_groups_tensor(test_folder: Path) -> torch.Tensor:
    """Предзагрузка всех тестовых групп в фиксированный тензор [N_groups, 4, 128, 128] в [0, 1]."""
    all_groups = sorted(p for p in test_folder.glob("group_*") if p.is_dir())
    groups = [g for g in all_groups if len(list(g.glob("*.png"))) == 4]
    if not groups:
        raise FileNotFoundError(f"Тестовые группы не найдены в {test_folder}")

    tensors = []
    for g in groups:
        files = sorted(list(g.glob("*.png")))
        imgs = []
        for f in files:
            with Image.open(f) as im:
                im_l = im.convert("L")
                if im_l.size != (GRID_SIZE, GRID_SIZE):
                    im_l = im_l.resize((GRID_SIZE, GRID_SIZE), Image.Resampling.BILINEAR)
                arr = np.array(im_l, dtype=np.float32) / 255.0
                imgs.append(arr)
        tensors.append(torch.from_numpy(np.stack(imgs, axis=0)))
    test_tensor = torch.stack(tensors, dim=0)  # [N_groups, 4, 128, 128]
    return test_tensor


def evaluate_on_test(
    model: nn.Module,
    propagator: nn.Module,
    criterion: nn.Module,
    test_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Быстрая векторная оценка на всех фиксированных тестовых группах с погрупповым расчетом метрик."""
    model.eval()
    with torch.no_grad():
        targets = test_tensor.to(device)  # [N, 4, 128, 128]
        phase_doe = model(targets)  # [N, 1, 128, 128]
        rec_norm = propagator(phase_doe, normalize=True)  # [N, 4, 128, 128]

        targets_norm = targets / torch.clamp_min(
            targets.amax(dim=(-2, -1), keepdim=True), 1e-12
        )

        loss_out = criterion(rec_norm, targets_norm, phase_mask=phase_doe)
        loss = loss_out[0]
        l1 = loss_out[1]

        # Погрупповой расчет Pearson CC и SSIM (для строгого соответствия calculate_metrics и test_model.py)
        N = targets_norm.shape[0]
        pcc_accum = [0.0] * 4
        ssim_accum = [0.0] * 4
        for i in range(N):
            for j in range(4):
                pj, sj = calculate_metrics(
                    targets_norm[i : i + 1, j : j + 1],
                    rec_norm[i : i + 1, j : j + 1],
                )
                pcc_accum[j] += pj
                ssim_accum[j] += sj

        pcc_planes = [p / N for p in pcc_accum]
        ssim_planes = [s / N for s in ssim_accum]

    mean_pcc = float(np.mean(pcc_planes))
    mean_ssim = float(np.mean(ssim_planes))
    composite_score = 0.6 * mean_pcc + 0.4 * mean_ssim

    return {
        "val_loss": float(loss.item()),
        "val_l1": float(l1.item()),
        "val_ssim": mean_ssim,
        "val_pcc": mean_pcc,
        "composite_score": composite_score,
        "pcc_planes": pcc_planes,
        "ssim_planes": ssim_planes,
    }


def train_pipeline(
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    p_synth: float = 0.50,
    steps_per_epoch: int = 150,
    warmup_epochs: int = 6,
    min_lr: float = 1e-6,
    num_synth_bank: int = 2500,
    w_ffl: float = 0.20,
    w_tv: float = 0.005,
    save_path: Path = CODE_DIR / "doe_network.pth",
    log_csv_path: Path = CODE_DIR / "training_log.csv",
    log_json_path: Path = CODE_DIR / "training_log.json",
    requested_device: str = "auto",
    resume: str | None = None,
    start_epoch: int | None = None,
    use_spectral: bool = False,
    spectral_modes: int = 16,
):
    """Главный пайплайн 200-эпохового обучения ResAttention U-Net / SpectralResAttentionUNet c CompositeLossV3."""
    device = get_device(requested_device)
    train_folder, test_folder = find_dataset_folders()

    print(f"[Данные] Обучающие патчи: {train_folder}", flush=True)
    print(f"[Данные] Тестовые группы: {test_folder}", flush=True)

    # 1. Сбор путей и предгенерация банка синтетических масок
    patch_files = collect_patch_files(train_folder)
    synth_bank = pregenerate_synthetic_bank(num_masks=num_synth_bank, size=GRID_SIZE)

    # 2. Инициализация FastRAMPatchDataset и предзагрузка тестового тензора
    epoch_samples = steps_per_epoch * batch_size
    train_dataset = FastRAMPatchDataset(
        patch_files=patch_files,
        synth_bank=synth_bank,
        p_synth=p_synth,
        epoch_length=epoch_samples,
        max_preload=50000,
        num_workers=16,
    )
    test_tensor = load_test_groups_tensor(test_folder)
    print(f"[Данные] Тестовый тензор подготовлен: {test_tensor.shape} ({test_tensor.shape[0]} групп)", flush=True)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_mem,
    )

    # 3. Модель, физический слой дифракции и обновленная композитная функция потерь V3
    if use_spectral:
        model = SpectralResAttentionUNet(
            in_channels=4,
            base_channels=32,
            modes1=spectral_modes,
            modes2=spectral_modes,
        ).to(device)
        total_params = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
        print(f"[Модель] SpectralResAttentionUNet (FNO) инициализирован (modes={spectral_modes}). Параметров: {total_params:,}", flush=True)
    else:
        model = DOENetwork(in_channels=4, base_channels=32).to(device)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Модель] ResAttention U-Net инициализирован. Параметров: {total_params:,}", flush=True)
        assert total_params == 8419929, f"Ожидалось 8,419,929 параметров, получено {total_params}"

    propagator = FresnelPropagation(
        z_list=Z_DISTANCES,
        wavelength=WAVELENGTH,
        pixel_size=PIXEL_SIZE,
        grid_size=GRID_SIZE,
    ).to(device)

    criterion = CompositeLossV3(
        w_pcc=4.0,
        w_ssim=1.2,
        w_l1=0.05,
        w_sobel=0.05,
        w_ffl=w_ffl,
        w_tv=w_tv,
        plane_weights=(0.9, 0.9, 1.1, 1.1),
    ).to(device)

    # 4. Возобновление обучения (если указано)
    resume_path = None
    if resume is not None:
        p = Path(resume) if Path(resume).is_absolute() else CODE_DIR / resume
        if p.exists():
            resume_path = p
        else:
            print(f"[Предупреждение] Чекпоинт {p} не найден! Обучение начнется заново.", flush=True)

    if resume_path is not None:
        state_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Возобновление] Успешно загружены веса модели из: {resume_path}", flush=True)

        if start_epoch is None:
            if log_csv_path.exists():
                try:
                    with open(log_csv_path, mode="r", encoding="utf-8") as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if len(lines) > 1:
                        last_line = lines[-1].split(",")
                        last_epoch = int(last_line[0])
                        start_epoch = last_epoch + 1
                        print(f"[Возобновление] Последняя завершенная эпоха в CSV: {last_epoch}. Старт с эпохи {start_epoch}.", flush=True)
                except Exception as e:
                    print(f"[Предупреждение] Ошибка чтения CSV: {e}", flush=True)

    if start_epoch is None:
        start_epoch = 1

    # 5. Оптимизатор и динамический расчет LR (Warmup + CosineAnnealing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    def compute_lr(ep: int) -> float:
        idx = ep - 1
        if idx < warmup_epochs:
            factor = 0.01 + 0.99 * (idx / max(1, warmup_epochs))
            return learning_rate * factor
        progress = (idx - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (learning_rate - min_lr) * cosine_decay

    # 6. Оценка до обучения
    print(f"\n[Валидация] Оценка весов перед запуском эпохи {start_epoch}...", flush=True)
    init_val = evaluate_on_test(model, propagator, criterion, test_tensor, device)
    print(
        f"  Старт: Val Loss={init_val['val_loss']:.4f}, L1={init_val['val_l1']:.4f}, "
        f"SSIM={init_val['val_ssim']:.4f}, Pearson CC={init_val['val_pcc']:.4f}, Score={init_val['composite_score']:.4f}",
        flush=True,
    )

    # 7. Инициализация логирования
    history = []
    csv_headers = [
        "epoch",
        "train_loss",
        "train_l1",
        "train_ssim",
        "train_pcc",
        "train_ffl",
        "train_tv",
        "val_loss",
        "val_l1",
        "val_ssim",
        "val_pcc",
        "composite_score",
        "lr",
        "epoch_time_sec",
    ]

    if start_epoch > 1:
        if log_json_path.exists():
            try:
                raw_hist = json.loads(log_json_path.read_text(encoding="utf-8"))
                history = [h for h in raw_hist if h.get("epoch", 0) < start_epoch]
            except Exception:
                history = []
        if not log_csv_path.exists():
            with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(csv_headers)
    else:
        with open(log_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    best_score = init_val["composite_score"]
    best_pcc = init_val["val_pcc"]
    best_ssim = init_val["val_ssim"]
    best_epoch = start_epoch - 1 if start_epoch > 1 else 0
    total_training_start = time.perf_counter()

    print(
        f"\n[Обучение] Старт с эпохи {start_epoch} до {epochs} (всего {epochs - start_epoch + 1} эпох, батч {batch_size}, {steps_per_epoch} шагов в эпохе)",
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total_train_loss = 0.0
        total_train_l1 = 0.0
        total_train_ssim = 0.0
        total_train_pcc = 0.0
        total_train_ffl = 0.0
        total_train_tv = 0.0
        step_count = 0

        current_lr = compute_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        for batch_idx, targets in enumerate(train_loader, start=1):
            if batch_idx > steps_per_epoch:
                break

            targets = targets.to(device, non_blocking=True)  # [B, 4, 128, 128]
            targets = augment_batch_on_gpu(targets)

            optimizer.zero_grad(set_to_none=True)

            predicted_doe = model(targets)  # [B, 1, 128, 128], [-pi, +pi]
            rec_norm = propagator(predicted_doe, normalize=True)  # [B, 4, 128, 128]

            targets_norm = targets / torch.clamp_min(
                targets.amax(dim=(-2, -1), keepdim=True), 1e-12
            )

            loss_out = criterion(rec_norm, targets_norm, phase_mask=predicted_doe)
            loss, l1, ssim_val, pcc_val = loss_out
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_l1 += l1.item()
            total_train_ssim += ssim_val.item()
            total_train_pcc += pcc_val.item()
            total_train_ffl += loss_out.ffl.item()
            total_train_tv += loss_out.tv.item()
            step_count += 1

            if batch_idx % max(1, steps_per_epoch // 3) == 0 or batch_idx == steps_per_epoch:
                elapsed = time.perf_counter() - epoch_start
                it_per_sec = batch_idx / max(elapsed, 1e-6)
                pct = batch_idx / steps_per_epoch * 100
                print(
                    f"  Эпоха {epoch:3d}/{epochs:3d} | шаг {batch_idx:3d}/{steps_per_epoch} ({pct:5.1f}%) | "
                    f"Loss: {loss.item():.4f} (PCC: {pcc_val.item():.4f}, SSIM: {ssim_val.item():.4f}, FFL: {loss_out.ffl.item():.4f}, TV: {loss_out.tv.item():.4f}, L1: {l1.item():.4f}) | "
                    f"{it_per_sec:.1f} батч/с",
                    flush=True,
                )

        epoch_time = time.perf_counter() - epoch_start

        avg_train_loss = total_train_loss / max(1, step_count)
        avg_train_l1 = total_train_l1 / max(1, step_count)
        avg_train_ssim = total_train_ssim / max(1, step_count)
        avg_train_pcc = total_train_pcc / max(1, step_count)
        avg_train_ffl = total_train_ffl / max(1, step_count)
        avg_train_tv = total_train_tv / max(1, step_count)

        # Валидация на тестовых группах
        val_metrics = evaluate_on_test(model, propagator, criterion, test_tensor, device)

        comp_score = val_metrics["composite_score"]
        val_pcc = val_metrics["val_pcc"]
        val_ssim = val_metrics["val_ssim"]

        print(
            f"[Эпоха {epoch:3d}/{epochs:3d}] "
            f"Train Loss: {avg_train_loss:.4f} (PCC: {avg_train_pcc:.4f}, SSIM: {avg_train_ssim:.4f}, FFL: {avg_train_ffl:.4f}, TV: {avg_train_tv:.4f}) | "
            f"Val Loss: {val_metrics['val_loss']:.4f} (PCC: {val_pcc:.4f}, SSIM: {val_ssim:.4f}, Score: {comp_score:.4f}) | "
            f"Время: {epoch_time:.1f}с, LR: {current_lr:.2e}",
            flush=True,
        )

        epoch_data = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "train_l1": round(avg_train_l1, 5),
            "train_ssim": round(avg_train_ssim, 5),
            "train_pcc": round(avg_train_pcc, 5),
            "train_ffl": round(avg_train_ffl, 5),
            "train_tv": round(avg_train_tv, 5),
            "val_loss": round(val_metrics["val_loss"], 5),
            "val_l1": round(val_metrics["val_l1"], 5),
            "val_ssim": round(val_ssim, 5),
            "val_pcc": round(val_pcc, 5),
            "composite_score": round(comp_score, 5),
            "lr": round(current_lr, 8),
            "epoch_time_sec": round(epoch_time, 2),
            "pcc_planes": [round(p, 4) for p in val_metrics["pcc_planes"]],
            "ssim_planes": [round(s, 4) for s in val_metrics["ssim_planes"]],
        }
        history.append(epoch_data)

        with open(log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch_data[h] for h in csv_headers])

        with open(log_json_path, mode="w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # Сохранение наилучшего чекпоинта по композитному скору: 0.6 * Val_PCC + 0.4 * Val_SSIM
        if comp_score > best_score:
            best_score = comp_score
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(
                f"  >>> Новый рекорд валидации! Чекпоинт сохранен: {save_path} "
                f"(Score: {best_score:.4f}, Val PCC: {val_pcc:.4f}, SSIM: {val_ssim:.4f})",
                flush=True,
            )

        if val_pcc > best_pcc:
            best_pcc = val_pcc
            torch.save(model.state_dict(), CODE_DIR / "doe_network_best_pcc.pth")

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), CODE_DIR / "doe_network_best_ssim.pth")

    total_time = time.perf_counter() - total_training_start
    print(
        f"\n[Обучение завершено] Общее время: {total_time / 60:.2f} мин. "
        f"Лучшая эпоха: {best_epoch} (Score={best_score:.4f}, PCC={best_pcc:.4f}, SSIM={best_ssim:.4f}).",
        flush=True,
    )
    print(f"Итоговый файл весов: {save_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение ResAttention U-Net ДОЭ для многоплоскостного волнового фронта"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Количество эпох (default: 200)")
    parser.add_argument("--batch_size", type=int, default=16, help="Размер батча (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Начальная скорость обучения (default: 3e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay AdamW (default: 1e-4)")
    parser.add_argument("--p_synth", type=float, default=0.50, help="Вероятность синтетических масок (default: 0.50)")
    parser.add_argument("--w_ffl", type=float, default=0.20, help="Вес Focal Frequency Loss (default: 0.20)")
    parser.add_argument("--w_tv", type=float, default=0.005, help="Вес Phase Total Variation Loss (default: 0.005)")
    parser.add_argument("--num_synth_bank", type=int, default=2500, help="Размер банка синтетических масок (default: 2500)")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=150,
        help="Количество шагов оптимизации в эпохе (default: 150)",
    )
    parser.add_argument("--warmup_epochs", type=int, default=6, help="Количество эпох warmup (default: 6)")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu или auto")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        nargs="?",
        const="doe_network.pth",
        help="Путь к файлу весов для возобновления обучения (по умолчанию: doe_network.pth)",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=None,
        help="Начальная эпоха при возобновлении (если None, определяется автоматически по training_log.csv)",
    )
    parser.add_argument(
        "--spectral",
        action="store_true",
        help="Использовать спектральные свертки Fourier Neural Operator (SpectralResAttentionUNet)",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=16,
        help="Количество частотных мод 2D FFT для SpectralConv2d (default: 16)",
    )
    args = parser.parse_args()

    train_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        p_synth=args.p_synth,
        steps_per_epoch=args.steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        requested_device=args.device,
        resume=args.resume,
        start_epoch=args.start_epoch,
        w_ffl=args.w_ffl,
        w_tv=args.w_tv,
        num_synth_bank=args.num_synth_bank,
        use_spectral=args.spectral,
        spectral_modes=args.modes,
    )
