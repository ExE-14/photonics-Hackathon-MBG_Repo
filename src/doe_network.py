import os
import random
import time
from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

# Базовые физические константы проекта хакатона МИФИ
WAVELENGTH: float = 532e-9      # 532 нм (зеленый лазер)
PIXEL_SIZE: float = 10e-6       # 10 мкм (линейный шаг пикселя ПВМС)
GRID_SIZE: int = 128            # Разрешение сетки 128x128
Z_DISTANCES: list[float] = [0.0241, 0.0253, 0.0265, 0.0277]  # Осевые расстояния в метрах

# Автоматическое определение устройства вычислений (CUDA -> MPS -> CPU)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

PARENT_PATH = Path(__file__).resolve().parent.parent
# Обучающие фрагменты: data/processed/convertered_dataset или data/dataset/convertered_dataset
input_folder = PARENT_PATH / "data" / "dataset" / "convertered_dataset"
if not input_folder.exists():
    input_folder = PARENT_PATH.parent / "data" / "processed" / "convertered_dataset"


class DoubleConv(nn.Module):
    """Двойной свёрточный блок: Conv3x3 -> BatchNorm2d -> LeakyReLU(0.2) -> Conv3x3 -> BatchNorm2d -> LeakyReLU(0.2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DOENetwork(nn.Module):
    """Полноценная архитектура U-Net для расчёта фазового ДОЭ по 4 входным изображениям.
    
    Энкодер: 4 уровня сжатия (128->64->32->16->8) с каналами 4->32->64->128->256->512.
    Боттлнек: 8x8 пикселей с 512 каналами, обеспечивающий глобальное поле зрения (RF >= 128).
    Декодер: билинейный апсемплинг x2, сквозные связи (skip connections) конкатенацией и DoubleConv.
    Выходной слой: 1x1 свёртка с активацией tanh(x) * pi, строго ограничивающая фазу в [-pi, +pi].
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # Входной блок (128x128, 32 канала)
        self.inc = DoubleConv(in_channels, base_channels)

        # 4 стадии сжатия (энкодер)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2),  # 64x64, 64
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, base_channels * 4),  # 32x32, 128
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 4, base_channels * 8),  # 16x16, 256
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 8, base_channels * 16),  # 8x8, 512 (bottleneck)
        )

        # 4 стадии декодера с билинейным апсемплингом
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = DoubleConv(
            base_channels * 16 + base_channels * 8, base_channels * 8
        )  # 512+256 -> 256

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = DoubleConv(
            base_channels * 8 + base_channels * 4, base_channels * 4
        )  # 256+128 -> 128

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up3 = DoubleConv(
            base_channels * 4 + base_channels * 2, base_channels * 2
        )  # 128+64 -> 64

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up4 = DoubleConv(
            base_channels * 2 + base_channels, base_channels
        )  # 64+32 -> 32

        # Выходная 1x1 проекция в 1 канал фазы
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вход: x [B, 4, 128, 128]
        Выход: phase_mask [B, 1, 128, 128] строго в диапазоне [-pi, +pi]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Энкодер
        x0 = self.inc(x)         # [B, 32, 128, 128]
        x1 = self.down1(x0)      # [B, 64, 64, 64]
        x2 = self.down2(x1)      # [B, 128, 32, 32]
        x3 = self.down3(x2)      # [B, 256, 16, 16]
        x4 = self.down4(x3)      # [B, 512, 8, 8] боттлнек

        # Декодер со skip-connections
        u1 = self.conv_up1(torch.cat([self.up1(x4), x3], dim=1))  # [B, 256, 16, 16]
        u2 = self.conv_up2(torch.cat([self.up2(u1), x2], dim=1))  # [B, 128, 32, 32]
        u3 = self.conv_up3(torch.cat([self.up3(u2), x1], dim=1))  # [B, 64, 64, 64]
        u4 = self.conv_up4(torch.cat([self.up4(u3), x0], dim=1))  # [B, 32, 128, 128]

        out = self.final_conv(u4)                                  # [B, 1, 128, 128]
        phase_mask = torch.tanh(out) * torch.pi                    # строго [-pi, +pi]
        return phase_mask


class FresnelPropagation(nn.Module):
    """Дифференцируемый физический слой расчета дифракции Френеля (Single-FFT).
    
    Вычисляет распространение световой волны от фазового ДОЭ до 4 осевых плоскостей
    с нормировкой восстановленных модулей амплитуд |K| к максимуму 1.0.
    """

    def __init__(
        self,
        z_list: Sequence[float] = Z_DISTANCES,
        wavelength: float = WAVELENGTH,
        pixel_size: float = PIXEL_SIZE,
        grid_size: int = GRID_SIZE,
    ):
        super().__init__()
        self.z_list = list(z_list)
        self.wavelength = float(wavelength)
        self.pixel_size = float(pixel_size)
        self.grid_size = int(grid_size)

        k = 2.0 * torch.pi / self.wavelength

        # Координатная сетка в плоскости ПВМС [-L/2, L/2)
        coords = (
            torch.arange(-self.grid_size // 2, self.grid_size // 2, dtype=torch.float32)
            * self.pixel_size
        )
        Y, X = torch.meshgrid(coords, coords, indexing="ij")
        R2 = X**2 + Y**2  # [128, 128]

        # Предварительно рассчитанные квадратичные фазовые множители чирпа
        # Размерность: [1, len(z_list), grid_size, grid_size], complex64
        pre_factors = torch.stack(
            [torch.exp(1j * (k / (2.0 * z)) * R2) for z in self.z_list],
            dim=0,
        ).unsqueeze(0).to(torch.complex64)

        # Регистрация буфера (persistent=False, чтобы не сохранять константу в state_dict)
        self.register_buffer("pre_factors", pre_factors, persistent=False)

    def forward(self, phase_mask: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Вход: phase_mask [B, 1, H, W] или [B, H, W], диапазон [-pi, +pi], float32.
        Выход: [B, len(z_list), H, W], модули амплитуд |K|, нормированные на max 1.0.
        """
        if phase_mask.ndim == 3:
            phase_mask = phase_mask.unsqueeze(1)

        H, W = phase_mask.shape[-2:]
        if (
            H != self.grid_size
            or W != self.grid_size
            or self.pre_factors.device != phase_mask.device
        ):
            k = 2.0 * torch.pi / self.wavelength
            coords_y = (
                torch.arange(-H // 2, H // 2, dtype=torch.float32, device=phase_mask.device)
                * self.pixel_size
            )
            coords_x = (
                torch.arange(-W // 2, W // 2, dtype=torch.float32, device=phase_mask.device)
                * self.pixel_size
            )
            Y, X = torch.meshgrid(coords_y, coords_x, indexing="ij")
            R2 = X**2 + Y**2
            pre_factors = torch.stack(
                [torch.exp(1j * (k / (2.0 * z)) * R2) for z in self.z_list],
                dim=0,
            ).unsqueeze(0).to(torch.complex64)
        else:
            pre_factors = self.pre_factors

        # Падающая волна единичной амплитуды с фазовой модуляцией: exp(1j * phi)
        field = torch.polar(torch.ones_like(phase_mask), phase_mask)

        # Батчевое широковещательное умножение по всем 4 расстояниям: [B, 4, H, W]
        field_mod = field * pre_factors

        # Двумерное центрированное преобразование Фурье
        shifted = torch.fft.ifftshift(field_mod, dim=(-2, -1))
        fft_field = torch.fft.fft2(shifted, dim=(-2, -1))
        fft_centered = torch.fft.fftshift(fft_field, dim=(-2, -1))

        # Модуль комплексной амплитуды |K|
        amplitude = torch.abs(fft_centered)

        if normalize:
            # Нормировка каждого изображения на свой максимум (максимум становится ровно 1.0)
            amplitude = amplitude / torch.clamp_min(
                amplitude.amax(dim=(-2, -1), keepdim=True), 1e-12
            )

        return amplitude


def fresnel_propagation(
    phase_mask: torch.Tensor,
    z_list: Sequence[float] = Z_DISTANCES,
    wavelength: float = WAVELENGTH,
    pixel_size: float = PIXEL_SIZE,
    normalize: bool = True,
) -> torch.Tensor:
    """Функциональный интерфейс дифракции Френеля для обратной совместимости."""
    propagator = FresnelPropagation(
        z_list=z_list,
        wavelength=wavelength,
        pixel_size=pixel_size,
        grid_size=phase_mask.shape[-1],
    ).to(phase_mask.device)
    return propagator(phase_mask, normalize=normalize)


def normalize_amplitude(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Нормировка комплексного модуля или интенсивности на максимум (максимум становится ровно 1.0).
    
    Для численной устойчивости и сохранения инвариантности масштаба при малых интенсивностях
    используется torch.clamp_min(amax, eps).
    """
    amax = tensor.amax(dim=(-2, -1), keepdim=True)
    return tensor / torch.clamp_min(amax, eps)


class CompositeLoss(nn.Module):
    """Составная дифференцируемая функция потерь: L = w1*L1 + w2*(1 - SSIM) + w3*(1 - Pearson CC).
    
    Рассчитывается по всем плоскостям дифракции. Гарантирует устойчивость без NaNs/Infs.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        eps_pcc: float = 1e-6,
        w_l1: float = 1.0,
        w_ssim: float = 1.0,
        w_pcc: float = 1.0,
    ):
        super().__init__()
        self.eps_pcc = eps_pcc
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_pcc = w_pcc
        self.c1 = 0.01**2
        self.c2 = 0.03**2

        # 11x11 Гауссов фильтр со среднеквадратичным отклонением sigma=1.5
        coords = torch.arange(-(window_size // 2), window_size // 2 + 1, dtype=torch.float32)
        gauss_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        window = (gauss_1d[:, None] * gauss_1d[None, :]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window, persistent=False)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Вход: pred [B, C, H, W], target [B, C, H, W] со значениями в [0, 1]
        Выход: (total_loss, l1, ssim_val, pcc_val)
        """
        # 1. L1 loss
        l1 = F.l1_loss(pred, target)

        # 2. Дифференцируемый SSIM
        B, C, H, W = pred.shape
        p_flat = pred.view(B * C, 1, H, W)
        t_flat = target.view(B * C, 1, H, W)
        pad = self.window.shape[-1] // 2
        p_pad = F.pad(p_flat, (pad, pad, pad, pad), mode="reflect")
        t_pad = F.pad(t_flat, (pad, pad, pad, pad), mode="reflect")

        win = self.window.to(device=pred.device, dtype=pred.dtype)
        mu_p = F.conv2d(p_pad, win)
        mu_t = F.conv2d(t_pad, win)
        mu_p_sq = mu_p.square()
        mu_t_sq = mu_t.square()
        mu_pt = mu_p * mu_t

        # F.relu предотвращает отрицательные дисперсии из-за погрешностей float32
        sig_p_sq = F.relu(F.conv2d(p_pad.square(), win) - mu_p_sq)
        sig_t_sq = F.relu(F.conv2d(t_pad.square(), win) - mu_t_sq)
        sig_pt = F.conv2d(p_pad * t_pad, win) - mu_pt

        num = (2.0 * mu_pt + self.c1) * (2.0 * sig_pt + self.c2)
        den = (mu_p_sq + mu_t_sq + self.c1) * (sig_p_sq + sig_t_sq + self.c2)
        ssim_map = num / den
        ssim_val = ssim_map.view(B, C, -1).mean()

        # 3. Дифференцируемый коэффициент корреляции Пирсона (Pearson CC)
        p_mean = pred.mean(dim=(-2, -1), keepdim=True)
        t_mean = target.mean(dim=(-2, -1), keepdim=True)
        p_res = pred - p_mean
        t_res = target - t_mean

        cov = (p_res * t_res).mean(dim=(-2, -1))
        var_p = (p_res**2).mean(dim=(-2, -1))
        var_t = (t_res**2).mean(dim=(-2, -1))

        # eps_pcc внутри sqrt устраняет бесконечную производную при нулевой дисперсии
        pcc = cov / (torch.sqrt(var_p + self.eps_pcc) * torch.sqrt(var_t + self.eps_pcc))
        pcc = torch.clamp(pcc, -1.0, 1.0)
        pcc_val = pcc.mean()

        total_loss = (
            self.w_l1 * l1
            + self.w_ssim * (1.0 - ssim_val)
            + self.w_pcc * (1.0 - pcc_val)
        )
        return total_loss, l1, ssim_val, pcc_val


def calculate_metrics(
    target: torch.Tensor, reconstructed: torch.Tensor
) -> tuple[float, float]:
    """Возвращает Pearson CC и SSIM для пары нормированных одноканальных изображений.
    
    Безопасен к константным изображениям (не падает с ошибкой, возвращает корректные значения).
    """
    t = target.detach().float()
    r = reconstructed.detach().float()
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(0) if t.shape[0] == 1 else t.unsqueeze(1)
    if r.ndim == 2:
        r = r.unsqueeze(0).unsqueeze(0)
    elif r.ndim == 3:
        r = r.unsqueeze(0) if r.shape[0] == 1 else r.unsqueeze(1)

    t_flat = t.reshape(-1)
    r_flat = r.reshape(-1)
    t_res = t_flat - t_flat.mean()
    r_res = r_flat - r_flat.mean()
    denom = torch.sqrt(t_res.square().sum() * r_res.square().sum())
    if denom < 1e-8:
        pearson = 1.0 if torch.allclose(t, r) else 0.0
    else:
        pearson = float(((t_res * r_res).sum() / denom).clamp(-1.0, 1.0).item())

    # SSIM
    coords = torch.arange(-5, 6, device=t.device, dtype=torch.float32)
    gauss = torch.exp(-(coords**2) / (2.0 * 1.5**2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] * gauss[None, :]).view(1, 1, 11, 11)

    t_pad = F.pad(t, (5, 5, 5, 5), mode="reflect")
    r_pad = F.pad(r, (5, 5, 5, 5), mode="reflect")
    mu_t = F.conv2d(t_pad, window)
    mu_r = F.conv2d(r_pad, window)
    mu_t_sq = mu_t.square()
    mu_r_sq = mu_r.square()
    mu_tr = mu_t * mu_r

    var_t = F.relu(F.conv2d(t_pad.square(), window) - mu_t_sq)
    var_r = F.relu(F.conv2d(r_pad.square(), window) - mu_r_sq)
    cov_tr = F.conv2d(t_pad * r_pad, window) - mu_tr

    c1 = 0.01**2
    c2 = 0.03**2
    den = (mu_t_sq + mu_r_sq + c1) * (var_t + var_r + c2)
    ssim = ((2.0 * mu_tr + c1) * (2.0 * cov_tr + c2) / den).mean()
    return float(pearson), float(ssim.item())


def generate_synthetic_mask(size: int = 128) -> torch.Tensor:
    """Генерирует контрастную синтетическую маску [1, size, size] со значениями в [0.0, 1.0].
    
    Поддерживает: символы/цифры, кольца/концентрические окружности, прямоугольники, кресты, круги.
    """
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    kind = random.choice(["char", "ring", "rect", "cross", "circle"])
    invert = random.random() < 0.3

    if kind == "char":
        chars = "0123456789MIPHABCXY"
        char = random.choice(chars)
        font_size = random.randint(40, 75)
        font = None
        for fc in [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("C:/Windows/Fonts/consola.ttf"),
        ]:
            if fc.exists():
                try:
                    font = ImageFont.truetype(str(fc), size=font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = max(0, min(size - w, (size - w) // 2 + random.randint(-15, 15)))
        y = max(0, min(size - h, (size - h) // 2 + random.randint(-15, 15)))
        draw.text((x, y), char, fill=255, font=font)

    elif kind == "ring":
        cx = size // 2 + random.randint(-12, 12)
        cy = size // 2 + random.randint(-12, 12)
        r1 = random.randint(12, 24)
        r2 = random.randint(32, 54)
        w1 = random.randint(2, 6)
        w2 = random.randint(2, 6)
        draw.ellipse([cx - r1, cy - r1, cx + r1, cy + r1], outline=255, width=w1)
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], outline=255, width=w2)
        if random.random() < 0.4:
            r3 = random.randint(4, 10)
            draw.ellipse([cx - r3, cy - r3, cx + r3, cy + r3], fill=255)

    elif kind == "rect":
        x0 = random.randint(8, 25)
        y0 = random.randint(8, 25)
        x1 = random.randint(size - 25, size - 8)
        y1 = random.randint(size - 25, size - 8)
        w = random.randint(2, 7)
        draw.rectangle([x0, y0, x1, y1], outline=255, width=w)
        if random.random() < 0.5:
            draw.rectangle([x0 + 15, y0 + 15, x1 - 15, y1 - 15], outline=255, width=max(1, w - 2))

    elif kind == "cross":
        cx = size // 2 + random.randint(-10, 10)
        cy = size // 2 + random.randint(-10, 10)
        w = random.randint(3, 8)
        arm = random.randint(35, 55)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=255, width=w)
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=255, width=w)
        if random.random() < 0.3:
            draw.line([(cx - arm // 2, cy - arm // 2), (cx + arm // 2, cy + arm // 2)], fill=255, width=w)
            draw.line([(cx - arm // 2, cy + arm // 2), (cx + arm // 2, cy - arm // 2)], fill=255, width=w)

    elif kind == "circle":
        cx = size // 2 + random.randint(-12, 12)
        cy = size // 2 + random.randint(-12, 12)
        r = random.randint(20, 50)
        if random.random() < 0.5:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        else:
            w = random.randint(2, 8)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=w)

    if invert:
        img = ImageOps.invert(img)

    return ToTensor()(img)


class DOEAugmentedDataset(Dataset):
    """Датасет для обучения и валидации ДОЭ.
    
    Поддерживает два режима:
    - mode='train': динамическая случайная группировка 4 независимых изображений (cross-image grouping)
      из пула всех патчей (40 092 патча) + генерация на лету контрастных синтетических масок
      с вероятностью p_synth (по умолчанию 0.3).
    - mode='test': детерминированная загрузка фиксированных групп из папок group_*.
    """

    def __init__(
        self,
        folder: Union[str, Path, None] = None,
        mode: str = "train",
        p_synth: float = 0.3,
        grid_size: int = GRID_SIZE,
    ):
        super().__init__()
        self.mode = mode
        self.p_synth = p_synth
        self.grid_size = grid_size
        self.transform = ToTensor()

        if folder is None:
            if mode == "train":
                folder = input_folder
            else:
                folder = PARENT_PATH / "data" / "dataset" / "test_dataset"
                if not folder.exists():
                    folder = PARENT_PATH.parent / "data" / "processed" / "test_dataset"

        self.folder = Path(folder)
        if not self.folder.exists():
            raise FileNotFoundError(f"Директория датасета не найдена: {self.folder}")

        if self.mode == "test":
            # Детерминированный режим: читаем существующие группы
            all_groups = sorted(
                path for path in self.folder.glob("group_*") if path.is_dir()
            )
            self.groups = [
                group for group in all_groups if len(list(group.glob("*.png"))) == 4
            ]
            if not self.groups:
                raise FileNotFoundError(f"Тестовые группы не найдены в {self.folder}")
        else:
            # Обучающий режим: собираем все доступные PNG-файлы в плоский список
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
            self.length = max(1, len(self.patch_files) // 4)

    def __len__(self) -> int:
        if self.mode == "test":
            return len(self.groups)
        return self.length

    def _load_patch(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            img_l = image.convert("L")
            if img_l.size != (self.grid_size, self.grid_size):
                img_l = img_l.resize(
                    (self.grid_size, self.grid_size), Image.Resampling.BILINEAR
                )
            return self.transform(img_l)

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.mode == "test":
            files = sorted(self.groups[index].glob("*.png"))
            images = [self._load_patch(p) for p in files]
            return torch.cat(images, dim=0)

        # Обучающий режим: cross-image grouping + синтетические маски
        channels = []
        for _ in range(4):
            if random.random() < self.p_synth:
                ch = generate_synthetic_mask(self.grid_size)
            else:
                idx = random.randint(0, len(self.patch_files) - 1)
                ch = self._load_patch(self.patch_files[idx])
            channels.append(ch)

        return torch.cat(channels, dim=0)  # [4, 128, 128]


class ImageGroupsDataset(DOEAugmentedDataset):
    """Класс-обертка для полной обратной совместимости со старыми скриптами.
    
    По умолчанию загружает детерминированные группы из папок group_*.
    """

    def __init__(self, folder: Union[str, Path, None] = None):
        if folder is None:
            folder = input_folder
        super().__init__(folder=folder, mode="test")


if __name__ == "__main__":
    print(f"Используем устройство: {device}", flush=True)

    print("Инициализация обучающего датасета с cross-image grouping и синтетическими масками...", flush=True)
    dataset = DOEAugmentedDataset(input_folder, mode="train", p_synth=0.3)
    print(f"Найдено патчей для обучения: {len(dataset.patch_files)}, эквивалентно групп: {len(dataset)}", flush=True)

    batch_size = 16 if torch.cuda.is_available() else 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = DOENetwork().to(device)
    propagator = FresnelPropagation(z_list=Z_DISTANCES).to(device)
    criterion = CompositeLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    epochs = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print(f"Начинаю обучение: {epochs} эпох, батч {batch_size}, {len(loader)} шагов в каждой.", flush=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_ssim = 0.0
        total_pcc = 0.0
        epoch_started = time.perf_counter()
        progress_interval = max(1, len(loader) // 50)

        for batch_index, targets in enumerate(loader, start=1):
            targets = targets.to(device)  # [B, 4, 128, 128]
            optimizer.zero_grad()

            predicted_doe = model(targets)  # [B, 1, 128, 128]
            reconstructed_norm = propagator(predicted_doe, normalize=True)  # [B, 4, 128, 128]

            targets_norm = targets / torch.clamp_min(
                targets.amax(dim=(-2, -1), keepdim=True), 1e-12
            )

            loss, l1, ssim_val, pcc_val = criterion(reconstructed_norm, targets_norm)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_l1 += l1.item()
            total_ssim += ssim_val.item()
            total_pcc += pcc_val.item()

            if batch_index % progress_interval == 0 or batch_index == len(loader):
                elapsed = time.perf_counter() - epoch_started
                steps_per_second = batch_index / max(elapsed, 1e-8)
                remaining = (len(loader) - batch_index) / max(steps_per_second, 1e-8)
                progress = batch_index / len(loader) * 100
                print(
                    f"\rЭпоха {epoch + 1}/{epochs} | "
                    f"шаг {batch_index}/{len(loader)} ({progress:5.1f}%) | "
                    f"Loss: {loss.item():.4f} (L1: {l1.item():.3f}, SSIM: {ssim_val.item():.3f}, PCC: {pcc_val.item():.3f}) | "
                    f"осталось: {remaining / 60:.1f} мин",
                    end="",
                    flush=True,
                )

        scheduler.step()
        average_loss = total_loss / len(loader)
        epoch_elapsed = time.perf_counter() - epoch_started
        print(
            f"\nЭпоха {epoch + 1}/{epochs} завершена. "
            f"Средний Loss: {average_loss:.4f} "
            f"(L1: {total_l1 / len(loader):.3f}, SSIM: {total_ssim / len(loader):.3f}, PCC: {total_pcc / len(loader):.3f}), "
            f"время: {epoch_elapsed / 60:.1f} мин",
            flush=True,
        )

    print("Сохраняю веса модели...", flush=True)
    torch.save(model.state_dict(), PARENT_PATH / "doe_network.pth")
    print("Обучение завершено, веса сохранены.")
