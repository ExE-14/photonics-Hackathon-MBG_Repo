import os
import random
import time
from pathlib import Path
from typing import Sequence, Union

import numpy as np
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


class ResDoubleConv(nn.Module):
    """Residual Double Convolution Block with identity/projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act2(out + res)


class AttentionGate(nn.Module):
    """Attention Gate for filtering skip connections across multiplane diffraction features."""

    def __init__(self, f_g: int, f_l: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer (Fourier Neural Operator, FNO).
    
    Performs global convolution in the spatial frequency domain via Fast Fourier Transform (FFT).
    Features:
    - 2D Real FFT (torch.fft.rfft2) with orthonormal scaling (norm='ortho') for gradient stability.
    - Truncated frequency modes (modes1, modes2) capturing low- and mid-frequency wave optical harmonics.
    - Complex learnable weights (torch.cfloat / complex64) parameterizing positive and negative frequencies.
    - Adaptive mode slicing (min-clamped to H//2 and W//2+1) ensuring resolution independence without crashes.
    - Optional grouped spectral convolutions (groups > 1) for reduced parameter count and ultra-low latency.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        groups: int = 1,
        norm: str = "ortho",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.groups = groups
        self.norm = norm

        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")

        cin_g = in_channels // groups
        cout_g = out_channels // groups
        scale = (2.0 / (cin_g + cout_g)) ** 0.5

        if groups == 1:
            self.weights1 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) / 1.41421356
            )
            self.weights2 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat) / 1.41421356
            )
        else:
            self.weights1 = nn.Parameter(
                scale * torch.randn(groups, cin_g, cout_g, modes1, modes2, dtype=torch.cfloat) / 1.41421356
            )
            self.weights2 = nn.Parameter(
                scale * torch.randn(groups, cin_g, cout_g, modes1, modes2, dtype=torch.cfloat) / 1.41421356
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm=self.norm)
        m1 = min(self.modes1, H // 2)
        m2 = min(self.modes2, W // 2 + 1)

        if self.groups == 1:
            out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
            out_ft[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
            return torch.fft.irfft2(out_ft, s=(H, W), norm=self.norm)
        else:
            cin_g = self.in_channels // self.groups
            cout_g = self.out_channels // self.groups
            x_ft_g = x_ft.view(B, self.groups, cin_g, H, W // 2 + 1)
            out_ft = torch.zeros(B, self.groups, cout_g, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
            out_ft[:, :, :, :m1, :m2] = torch.einsum("bgixy,gioxy->bgoxy", x_ft_g[:, :, :, :m1, :m2], self.weights1[:, :, :, :m1, :m2])
            out_ft[:, :, :, -m1:, :m2] = torch.einsum("bgixy,gioxy->bgoxy", x_ft_g[:, :, :, -m1:, :m2], self.weights2[:, :, :, -m1:, :m2])
            return torch.fft.irfft2(out_ft.view(B, self.out_channels, H, W // 2 + 1), s=(H, W), norm=self.norm)


class SpectralBlock(nn.Module):
    """Fourier Neural Operator Block combining spectral convolution and spatial shortcut.

    Structure:
    - SpectralConv2d branch (global frequency domain filtering)
    - Conv2d 1x1 branch (local spatial shortcut)
    - BatchNorm2d + LeakyReLU(0.2)
    - Optional identity residual connection when in_channels == out_channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 16,
        modes2: int = 16,
        groups: int = 1,
        residual: bool = True,
        norm: str = "ortho",
    ):
        super().__init__()
        self.spectral_conv = SpectralConv2d(
            in_channels,
            out_channels,
            modes1=modes1,
            modes2=modes2,
            groups=groups,
            norm=norm,
        )
        self.w = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.residual = residual and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.spectral_conv(x) + self.w(x)
        out = self.act(self.norm(out))
        if self.residual:
            out = out + x
        return out


class DOENetwork(nn.Module):
    """ResAttention U-Net architecture for high-fidelity multiplane DOE phase mask synthesis.

    Combines residual double convolution blocks with attention-gated skip connections.
    By default (use_spectral=False), preserves exact 8,419,929 parameters and 100% backward
    compatibility with existing weights checkpoints (e.g. doe_network.pth).
    When use_spectral=True, incorporates Fourier Neural Operator (FNO) SpectralConv2d blocks
    in the bottleneck and decoder for optical wavefront harmonic processing.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        use_spectral: bool = False,
        modes1: int = 16,
        modes2: int = 16,
        spectral_groups: int = 8,
    ):
        super().__init__()
        b = base_channels
        self.in_channels = in_channels
        self.base_channels = b
        self.use_spectral = use_spectral

        # Encoder stages with residual connections
        self.inc = ResDoubleConv(in_channels, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(b * 4, b * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResDoubleConv(b * 8, b * 16))

        # Bottleneck Spectral FNO Block (if enabled)
        if self.use_spectral:
            self.bottleneck_fno = SpectralBlock(
                b * 16,
                b * 16,
                modes1=8,
                modes2=8,
                groups=spectral_groups,
                residual=True,
            )

        # Decoder stages with Attention Gates
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att1 = AttentionGate(f_g=b * 16, f_l=b * 8, f_int=b * 4)
        self.conv_up1 = ResDoubleConv(b * 16 + b * 8, b * 8)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att2 = AttentionGate(f_g=b * 8, f_l=b * 4, f_int=b * 2)
        self.conv_up2 = ResDoubleConv(b * 8 + b * 4, b * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att3 = AttentionGate(f_g=b * 4, f_l=b * 2, f_int=b)
        self.conv_up3 = ResDoubleConv(b * 4 + b * 2, b * 2)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.att4 = AttentionGate(f_g=b * 2, f_l=b, f_int=b // 2)
        self.conv_up4 = ResDoubleConv(b * 2 + b, b)

        # Decoder Spectral FNO Block (if enabled, operates directly at 128x128 before output projection)
        if self.use_spectral:
            self.spectral_dec = SpectralBlock(
                b,
                b,
                modes1=modes1,
                modes2=modes2,
                groups=1,
                residual=True,
            )

        # Output projection to single-channel phase mask [-pi, +pi]
        self.final_conv = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Вход: x [B, 4, 128, 128] в [0, 1].

        Выход: phase_mask [B, 1, 128, 128] строго в диапазоне [-pi, +pi].
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Encoder
        x0 = self.inc(x)        # [B, 32, 128, 128]
        x1 = self.down1(x0)     # [B, 64, 64, 64]
        x2 = self.down2(x1)     # [B, 128, 32, 32]
        x3 = self.down3(x2)     # [B, 256, 16, 16]
        x4 = self.down4(x3)     # [B, 512, 8, 8]

        # Bottleneck FNO processing
        if self.use_spectral:
            x4 = self.bottleneck_fno(x4)

        # Decoder with attention-gated skip connections
        g1 = self.up1(x4)
        a1 = self.att1(g=g1, x=x3)
        u1 = self.conv_up1(torch.cat([g1, a1], dim=1))  # [B, 256, 16, 16]

        g2 = self.up2(u1)
        a2 = self.att2(g=g2, x=x2)
        u2 = self.conv_up2(torch.cat([g2, a2], dim=1))  # [B, 128, 32, 32]

        g3 = self.up3(u2)
        a3 = self.att3(g=g3, x=x1)
        u3 = self.conv_up3(torch.cat([g3, a3], dim=1))  # [B, 64, 64, 64]

        g4 = self.up4(u3)
        a4 = self.att4(g=g4, x=x0)
        u4 = self.conv_up4(torch.cat([g4, a4], dim=1))  # [B, 32, 128, 128]

        # Decoder FNO wavefront synthesis
        if self.use_spectral:
            u4 = self.spectral_dec(u4)

        out = self.final_conv(u4)
        return torch.tanh(out) * torch.pi


class SpectralResAttentionUNet(DOENetwork):
    """Fourier Neural Operator (FNO) Augmented ResAttention U-Net for Optical Wavefront Synthesis.

    Equipped with 2D FFT spectral convolutions (SpectralConv2d) in the bottleneck and decoder,
    allowing the network to directly learn and modulate optical spatial harmonics.
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        modes1: int = 16,
        modes2: int = 16,
        spectral_groups: int = 8,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            use_spectral=True,
            modes1=modes1,
            modes2=modes2,
            spectral_groups=spectral_groups,
        )


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
            # Нормировка каждого изображения на свой максимум с detach() для исключения спайка градиента
            m = torch.clamp_min(amplitude.amax(dim=(-2, -1), keepdim=True).detach(), 1e-12)
            amplitude = amplitude / m

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


class OptimizedCompositeLoss(nn.Module):
    """Оптимизированная составная функция потерь для достижения Pearson CC > 0.8672 и SSIM > 0.5918."""

    def __init__(
        self,
        w_pcc: float = 4.0,
        w_ssim: float = 1.2,
        w_l1: float = 0.05,
        w_sobel: float = 0.05,
        plane_weights: Sequence[float] = (0.9, 0.9, 1.1, 1.1),
        window_size: int = 11,
        sigma: float = 1.5,
        eps_pcc: float = 1e-6,
    ):
        super().__init__()
        self.w_pcc = w_pcc
        self.w_ssim = w_ssim
        self.w_l1 = w_l1
        self.w_sobel = w_sobel
        self.eps_pcc = eps_pcc

        # Нормированные веса плоскостей (сумма равна количеству плоскостей = 4.0)
        pw = torch.tensor(plane_weights, dtype=torch.float32)
        pw = (pw / pw.sum()) * len(plane_weights)
        self.register_buffer("plane_weights", pw.view(1, -1, 1, 1), persistent=False)

        # Гауссов фильтр 11x11 для SSIM
        coords = torch.arange(-(window_size // 2), window_size // 2 + 1, dtype=torch.float32)
        gauss_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        window = (gauss_1d[:, None] * gauss_1d[None, :]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window, persistent=False)

        # Ядра Собеля для выделения контуров
        sobel_x = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
            .unsqueeze(0)
            .unsqueeze(0)
            / 4.0
        )
        sobel_y = (
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
            .unsqueeze(0)
            .unsqueeze(0)
            / 4.0
        )
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        self.c1 = 0.01**2
        self.c2 = 0.03**2

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = pred.shape
        pw = self.plane_weights.to(pred.device)
        pw_plane = pw.squeeze(-1).squeeze(-1)  # [1, C]

        # 1. Plane-weighted L1
        diff = torch.abs(pred - target)
        l1_per_plane = (diff * pw).mean(dim=(-2, -1))
        l1 = l1_per_plane.mean()

        # 2. Plane-weighted SSIM
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

        sig_p_sq = F.relu(F.conv2d(p_pad.square(), win) - mu_p_sq)
        sig_t_sq = F.relu(F.conv2d(t_pad.square(), win) - mu_t_sq)
        sig_pt = F.conv2d(p_pad * t_pad, win) - mu_pt

        num = (2.0 * mu_pt + self.c1) * (2.0 * sig_pt + self.c2)
        den = (mu_p_sq + mu_t_sq + self.c1) * (sig_p_sq + sig_t_sq + self.c2)
        ssim_map = num / den
        ssim_per_plane = ssim_map.view(B, C, -1).mean(dim=-1)
        ssim_weighted = (ssim_per_plane * pw_plane).mean()
        ssim_val = ssim_per_plane.mean()

        # 3. Plane-weighted Pearson CC
        p_mean = pred.mean(dim=(-2, -1), keepdim=True)
        t_mean = target.mean(dim=(-2, -1), keepdim=True)
        p_res = pred - p_mean
        t_res = target - t_mean

        cov = (p_res * t_res).mean(dim=(-2, -1))
        var_p = (p_res**2).mean(dim=(-2, -1))
        var_t = (t_res**2).mean(dim=(-2, -1))

        pcc = cov / (torch.sqrt(var_p + self.eps_pcc) * torch.sqrt(var_t + self.eps_pcc))
        pcc = torch.clamp(pcc, -1.0, 1.0)
        pcc_weighted = (pcc * pw_plane).mean()
        pcc_val = pcc.mean()

        total_loss = (
            self.w_pcc * (1.0 - pcc_weighted)
            + self.w_ssim * (1.0 - ssim_weighted)
            + self.w_l1 * l1
        )

        # 4. Sobel Gradient Loss
        if self.w_sobel > 0:
            sx = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
            sy = self.sobel_y.to(device=pred.device, dtype=pred.dtype)
            gx_p = F.conv2d(p_flat, sx, padding=1)
            gy_p = F.conv2d(p_flat, sy, padding=1)
            gx_t = F.conv2d(t_flat, sx, padding=1)
            gy_t = F.conv2d(t_flat, sy, padding=1)
            sobel_loss = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
            total_loss = total_loss + self.w_sobel * sobel_loss

        return total_loss, l1, ssim_val, pcc_val


class LossOutput(tuple):
    """Кортеж результатов функции потерь с поддержкой 4-элементной распаковки и именованных атрибутов.
    
    Обеспечивает 100% обратную совместимость с кодом вида:
        loss, l1, ssim, pcc = criterion(pred, target)
    А также именованный доступ к расширенным метрикам V3:
        out.ffl, out.tv, out.sobel, out.total, out.l1, out.ssim, out.pcc
    """

    def __new__(cls, total, l1, ssim, pcc, ffl=None, tv=None, sobel=None):
        return super().__new__(cls, (total, l1, ssim, pcc))

    def __init__(self, total, l1, ssim, pcc, ffl=None, tv=None, sobel=None):
        self.total = total
        self.l1 = l1
        self.ssim = ssim
        self.pcc = pcc
        self.ffl = ffl if ffl is not None else torch.tensor(0.0)
        self.tv = tv if tv is not None else torch.tensor(0.0)
        self.sobel = sobel if sobel is not None else torch.tensor(0.0)


class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss (FFL) для частотной оптимизации дифракционного волнового фронта.
    
    Основан на методологии ICCV 2021 (Focal Frequency Loss for Image Reconstruction and Synthesis).
    Переводит реконструированное и целевое поле в 2D частотное пространство через ортонормированное FFT,
    вычисляет квадратичное эвклидово расстояние между спектрами и динамически модулирует его
    весовой матрицей сложности:
        w(u, v) = (|F_recon - F_target| / max(|F_recon - F_target| + eps))^alpha
    Это динамически штрафует размытие высоких частот и контуров, снижая вес легко восстановимых низких частот.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8, loss_weight: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.loss_weight = float(loss_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        plane_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pred, target: [B, C, H, W]
        p_freq = torch.fft.fft2(pred, dim=(-2, -1), norm="ortho")
        t_freq = torch.fft.fft2(target, dim=(-2, -1), norm="ortho")
        diff = p_freq - t_freq

        # Квадратичное эвклидово расстояние: (Delta R)^2 + (Delta I)^2
        freq_dist_sq = diff.real.square() + diff.imag.square()

        # Динамическая спектральная весовая матрица (detached)
        with torch.no_grad():
            diff_mag = torch.sqrt(freq_dist_sq + 1e-12)
            amax = diff_mag.amax(dim=(-2, -1), keepdim=True)
            weight_matrix = (diff_mag / torch.clamp_min(amax, self.eps)) ** self.alpha

        loss_map = weight_matrix * freq_dist_sq
        if plane_weights is not None:
            loss_map = loss_map * plane_weights

        return self.loss_weight * loss_map.mean()


class PhaseTotalVariationLoss(nn.Module):
    """Phase Total Variation (TV) Loss: регуляризация гладкости фазовой маски ДОЭ.
    
    Подавляет высокочастотный фазовый шум (speckle noise) и снижает паразитный cross-talk
    между 4 осевыми дифракционными плоскостями.
    
    Учитывает физическую периодичность оптической фазы на окружности S^1 (2π-wrapping)
    через atan2(sin(Delta), cos(Delta)), устраняя ложные штрафы на границах wrapping'а [-pi, +pi].
    """

    def __init__(self, loss_weight: float = 1.0, wrapped: bool = True):
        super().__init__()
        self.loss_weight = float(loss_weight)
        self.wrapped = bool(wrapped)

    def forward(self, phase_mask: torch.Tensor) -> torch.Tensor:
        # phase_mask: [B, 1, H, W] или [B, H, W] в диапазоне [-pi, +pi]
        if phase_mask.ndim == 3:
            phase_mask = phase_mask.unsqueeze(1)

        diff_h = phase_mask[:, :, 1:, :] - phase_mask[:, :, :-1, :]
        diff_w = phase_mask[:, :, :, 1:] - phase_mask[:, :, :, :-1]

        if self.wrapped:
            diff_h = torch.atan2(torch.sin(diff_h), torch.cos(diff_h))
            diff_w = torch.atan2(torch.sin(diff_w), torch.cos(diff_w))

        tv_loss = diff_h.abs().mean() + diff_w.abs().mean()
        return self.loss_weight * tv_loss


PhaseTVLoss = PhaseTotalVariationLoss


class CompositeLossV3(nn.Module):
    """Обновленная композитная функция потерь 3-го поколения (V3) для проекта ДОЭ МИФИ.
    
    Объединяет 6 физических и спектральных компонент:
    1. Pearson CC Loss (w_pcc=4.0): максимизация корреляции Пирсона между нормированными полями.
    2. SSIM Loss (w_ssim=1.2): сохранение локального структурного сходства (11x11 Gauss).
    3. Focal Frequency Loss (w_ffl=0.20): 2D Фурье-штраф с динамическими весами сложности для контуров.
    4. Phase TV Loss (w_tv=0.005): 2π-периодическая регуляризация фазовой маски для подавления шума и cross-talk.
    5. Sobel Loss (w_sobel=0.05): дифференцируемое сопоставление пространственных градиентов X/Y.
    6. Plane-weighted L1 Loss (w_l1=0.05): базовое L1-соответствие амплитуд.
    """

    def __init__(
        self,
        w_pcc: float = 4.0,
        w_ssim: float = 1.2,
        w_l1: float = 0.05,
        w_sobel: float = 0.05,
        w_ffl: float = 0.20,
        w_tv: float = 0.005,
        plane_weights: Sequence[float] = (0.9, 0.9, 1.1, 1.1),
        window_size: int = 11,
        sigma: float = 1.5,
        eps_pcc: float = 1e-6,
        ffl_alpha: float = 1.0,
        wrapped_tv: bool = True,
    ):
        super().__init__()
        self.w_pcc = float(w_pcc)
        self.w_ssim = float(w_ssim)
        self.w_l1 = float(w_l1)
        self.w_sobel = float(w_sobel)
        self.w_ffl = float(w_ffl)
        self.w_tv = float(w_tv)
        self.eps_pcc = float(eps_pcc)

        # Осевые веса плоскостей (сумма = 4.0)
        pw = torch.tensor(plane_weights, dtype=torch.float32)
        pw = (pw / pw.sum()) * len(plane_weights)
        self.register_buffer("plane_weights", pw.view(1, -1, 1, 1), persistent=False)

        # Гауссов фильтр 11x11 для SSIM
        coords = torch.arange(-(window_size // 2), window_size // 2 + 1, dtype=torch.float32)
        gauss_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        window = (gauss_1d[:, None] * gauss_1d[None, :]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("window", window, persistent=False)

        # Ядра Собеля для выделения контуров
        sobel_x = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
            .unsqueeze(0)
            .unsqueeze(0)
            / 4.0
        )
        sobel_y = (
            torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
            .unsqueeze(0)
            .unsqueeze(0)
            / 4.0
        )
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

        # Модули FFL и Phase TV
        self.ffl = FocalFrequencyLoss(alpha=ffl_alpha)
        self.tv = PhaseTotalVariationLoss(wrapped=wrapped_tv)

        self.c1 = 0.01**2
        self.c2 = 0.03**2
        self.last_metrics: dict[str, float] = {}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        phase_mask: torch.Tensor | None = None,
    ) -> LossOutput:
        B, C, H, W = pred.shape
        pw = self.plane_weights.to(pred.device)
        pw_plane = pw.squeeze(-1).squeeze(-1)  # [1, C]

        # 1. Plane-weighted L1
        diff = torch.abs(pred - target)
        l1_per_plane = (diff * pw).mean(dim=(-2, -1))
        l1 = l1_per_plane.mean()

        # 2. Plane-weighted SSIM
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

        sig_p_sq = F.relu(F.conv2d(p_pad.square(), win) - mu_p_sq)
        sig_t_sq = F.relu(F.conv2d(t_pad.square(), win) - mu_t_sq)
        sig_pt = F.conv2d(p_pad * t_pad, win) - mu_pt

        num = (2.0 * mu_pt + self.c1) * (2.0 * sig_pt + self.c2)
        den = (mu_p_sq + mu_t_sq + self.c1) * (sig_p_sq + sig_t_sq + self.c2)
        ssim_map = num / den
        ssim_per_plane = ssim_map.view(B, C, -1).mean(dim=-1)
        ssim_weighted = (ssim_per_plane * pw_plane).mean()
        ssim_val = ssim_per_plane.mean()

        # 3. Plane-weighted Pearson CC
        p_mean = pred.mean(dim=(-2, -1), keepdim=True)
        t_mean = target.mean(dim=(-2, -1), keepdim=True)
        p_res = pred - p_mean
        t_res = target - t_mean

        cov = (p_res * t_res).mean(dim=(-2, -1))
        var_p = (p_res**2).mean(dim=(-2, -1))
        var_t = (t_res**2).mean(dim=(-2, -1))

        pcc = cov / (torch.sqrt(var_p + self.eps_pcc) * torch.sqrt(var_t + self.eps_pcc))
        pcc = torch.clamp(pcc, -1.0, 1.0)
        pcc_weighted = (pcc * pw_plane).mean()
        pcc_val = pcc.mean()

        total_loss = (
            self.w_pcc * (1.0 - pcc_weighted)
            + self.w_ssim * (1.0 - ssim_weighted)
            + self.w_l1 * l1
        )

        # 4. Sobel Gradient Loss
        sobel_loss = torch.tensor(0.0, device=pred.device)
        if self.w_sobel > 0:
            sx = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
            sy = self.sobel_y.to(device=pred.device, dtype=pred.dtype)
            gx_p = F.conv2d(p_flat, sx, padding=1)
            gy_p = F.conv2d(p_flat, sy, padding=1)
            gx_t = F.conv2d(t_flat, sx, padding=1)
            gy_t = F.conv2d(t_flat, sy, padding=1)
            sobel_loss = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
            total_loss = total_loss + self.w_sobel * sobel_loss

        # 5. Focal Frequency Loss (FFL)
        ffl_loss = torch.tensor(0.0, device=pred.device)
        if self.w_ffl > 0:
            ffl_loss = self.ffl(pred, target, plane_weights=pw)
            total_loss = total_loss + self.w_ffl * ffl_loss

        # 6. Phase Total Variation Loss (TV-loss)
        tv_loss = torch.tensor(0.0, device=pred.device)
        if self.w_tv > 0 and phase_mask is not None:
            tv_loss = self.tv(phase_mask)
            total_loss = total_loss + self.w_tv * tv_loss

        self.last_metrics = {
            "loss": float(total_loss.item()),
            "l1": float(l1.item()),
            "ssim": float(ssim_val.item()),
            "pcc": float(pcc_val.item()),
            "sobel": float(sobel_loss.item()),
            "ffl": float(ffl_loss.item()),
            "tv": float(tv_loss.item()),
        }

        return LossOutput(
            total=total_loss,
            l1=l1,
            ssim=ssim_val,
            pcc=pcc_val,
            ffl=ffl_loss,
            tv=tv_loss,
            sobel=sobel_loss,
        )


class CompositeLoss(OptimizedCompositeLoss):
    """Составная функция потерь с обратной совместимостью и оптимизированными весами по умолчанию."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        eps_pcc: float = 1e-6,
        w_l1: float = 0.05,
        w_ssim: float = 1.2,
        w_pcc: float = 4.0,
        w_sobel: float = 0.05,
        plane_weights: Sequence[float] = (0.9, 0.9, 1.1, 1.1),
    ):
        super().__init__(
            w_pcc=w_pcc,
            w_ssim=w_ssim,
            w_l1=w_l1,
            w_sobel=w_sobel,
            plane_weights=plane_weights,
            window_size=window_size,
            sigma=sigma,
            eps_pcc=eps_pcc,
        )


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


_SYSTEM_FONTS: list[Path] = [
    p
    for p in [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/cour.ttf"),
    ]
    if p.exists()
]


def _get_cached_font(size: int) -> ImageFont.ImageFont:
    if _SYSTEM_FONTS:
        fpath = random.choice(_SYSTEM_FONTS)
        try:
            return ImageFont.truetype(str(fpath), size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_dark_geometric(size: int = 128) -> Image.Image:
    """Геометрические силуэты с темным фоном (звезды, многоугольники, шестерни, кресты, ромбы, рамки)."""
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2 + random.randint(-8, 8), size // 2 + random.randint(-8, 8)
    shape_type = random.choice(["star", "polygon", "cross", "gear", "nested_rect", "diamond"])
    fill_val = random.randint(180, 255)
    outline_val = random.randint(200, 255)

    if shape_type == "star":
        points = random.choice([5, 6, 8])
        r_outer = random.randint(28, 54)
        r_inner = int(r_outer * random.uniform(0.35, 0.55))
        coords = []
        angle_step = np.pi / points
        rot = random.uniform(0, 2 * np.pi)
        for i in range(2 * points):
            r = r_outer if i % 2 == 0 else r_inner
            a = rot + i * angle_step
            coords.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        if random.random() < 0.6:
            draw.polygon(coords, fill=fill_val)
        else:
            draw.polygon(coords, outline=outline_val, width=random.randint(2, 6))

    elif shape_type == "polygon":
        n_sides = random.choice([3, 5, 6, 8])
        radius = random.randint(25, 52)
        rot = random.uniform(0, 2 * np.pi)
        coords = [
            (
                cx + radius * np.cos(rot + 2 * np.pi * i / n_sides),
                cy + radius * np.sin(rot + 2 * np.pi * i / n_sides),
            )
            for i in range(n_sides)
        ]
        if random.random() < 0.6:
            draw.polygon(coords, fill=fill_val)
        else:
            draw.polygon(coords, outline=outline_val, width=random.randint(2, 6))

    elif shape_type == "gear":
        teeth = random.randint(6, 12)
        r_base = random.randint(24, 42)
        r_tooth = r_base + random.randint(8, 14)
        coords = []
        rot = random.uniform(0, 2 * np.pi)
        step = 2 * np.pi / (teeth * 2)
        for i in range(teeth * 2):
            r = r_tooth if i % 2 == 0 else r_base
            a = rot + i * step
            coords.append((cx + r * np.cos(a), cy + r * np.sin(a)))
        draw.polygon(coords, fill=fill_val)
        hole_r = random.randint(8, 15)
        draw.ellipse([cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r], fill=0)

    elif shape_type == "cross":
        w = random.randint(4, 12)
        arm = random.randint(28, 54)
        draw.line([(cx, cy - arm), (cx, cy + arm)], fill=fill_val, width=w)
        draw.line([(cx - arm, cy), (cx + arm, cy)], fill=fill_val, width=w)
        if random.random() < 0.4:
            diag_arm = int(arm * 0.7)
            draw.line([(cx - diag_arm, cy - diag_arm), (cx + diag_arm, cy + diag_arm)], fill=fill_val, width=w)
            draw.line([(cx - diag_arm, cy + diag_arm), (cx + diag_arm, cy - diag_arm)], fill=fill_val, width=w)

    elif shape_type == "nested_rect":
        for s in range(random.randint(2, 4)):
            w = random.randint(16 + s * 12, 28 + s * 14)
            h = random.randint(16 + s * 12, 28 + s * 14)
            draw.rectangle([cx - w, cy - h, cx + w, cy + h], outline=outline_val, width=random.randint(2, 5))

    elif shape_type == "diamond":
        w = random.randint(24, 52)
        h = random.randint(24, 52)
        coords = [(cx, cy - h), (cx + w, cy), (cx, cy + h), (cx - w, cy)]
        if random.random() < 0.6:
            draw.polygon(coords, fill=fill_val)
        else:
            draw.polygon(coords, outline=outline_val, width=random.randint(2, 6))

    return img


def make_dark_symbols(size: int = 128) -> Image.Image:
    """Символы и типографика на черном фоне (греческие буквы, латиница, цифры, мат. операторы, слова)."""
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    mode = random.choice(["char", "word", "math", "greek"])

    if mode == "char":
        text = random.choice("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZБГДЖИЛПФЦЧШЩ")
        font_size = random.randint(48, 80)
        font = _get_cached_font(font_size)
    elif mode == "word":
        text = random.choice(["MEPhI", "DOE", "LASER", "AI", "OPT", "4060", "Z1", "Z2", "Z3", "Z4"])
        font_size = random.randint(26, 42)
        font = _get_cached_font(font_size)
    elif mode == "greek":
        text = random.choice(["\u03b1", "\u03b2", "\u03b3", "\u03b4", "\u03bb", "\u03bc", "\u03c0", "\u03c3", "\u03c9", "\u0394", "\u03a9", "\u03a8", "\u03a6"])
        font_size = random.randint(48, 76)
        font = _get_cached_font(font_size)
    else:
        text = random.choice(["\u2211", "\u222b", "\u221e", "\u2207", "\u2248", "\u2260", "\u00b1", "\u2192", "\u2194", "\u2714", "\u2716"])
        font_size = random.randint(45, 72)
        font = _get_cached_font(font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 + random.randint(-6, 6)
    y = (size - th) // 2 + random.randint(-6, 6)
    val = random.randint(210, 255)
    draw.text((x, y), text, fill=val, font=font)
    return img


def make_dark_fresnel_zone_plate(size: int = 128) -> Image.Image:
    """Зонные пластинки Френеля на темном фоне (бинарные и синусоидальные с квадратичной фазой r_n ~ sqrt(n))."""
    cx = size // 2 + random.randint(-6, 6)
    cy = size // 2 + random.randint(-6, 6)
    y, x = np.ogrid[:size, :size]
    r2 = (x - cx) ** 2 + (y - cy) ** 2

    mode = random.choice(["binary", "sinusoidal"])
    pitch = random.uniform(8.0, 24.0)

    if mode == "binary":
        phase = (r2 / pitch) % 2.0
        arr = np.where(phase < 1.0, 255, 0).astype(np.uint8)
    else:
        phase = (r2 / pitch) * np.pi
        arr = np.clip((0.5 * (1.0 + np.cos(phase)) * 255.0), 0, 255).astype(np.uint8)

    mask_r = random.randint(44, 62)
    aperture = r2 <= mask_r**2
    arr = arr * aperture.astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_dark_radial_rings_spokes(size: int = 128) -> Image.Image:
    """Радиальные кольца, мишени, спицы (звезда Сименса) и спирали на темном фоне."""
    cx = size // 2 + random.randint(-6, 6)
    cy = size // 2 + random.randint(-6, 6)
    mode = random.choice(["siemens_star", "rings", "spiral"])

    y, x = np.ogrid[:size, :size]
    dy = y - cy
    dx = x - cx
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    if mode == "siemens_star":
        spokes = random.choice([8, 12, 16, 20, 24])
        pattern = np.sin(spokes * theta)
        arr = np.where(pattern > 0, 255, 0).astype(np.uint8)
        outer_r = random.randint(44, 62)
        arr[r > outer_r] = 0
        arr[r < random.randint(2, 6)] = random.choice([0, 255])
        return Image.fromarray(arr, mode="L")
    elif mode == "spiral":
        n_arms = random.choice([1, 2, 3])
        tightness = random.uniform(2.0, 5.0)
        spiral_phase = (theta * n_arms + r / tightness) % (2 * np.pi)
        arr = np.where(spiral_phase < np.pi, 255, 0).astype(np.uint8)
        arr[r > random.randint(46, 62)] = 0
        return Image.fromarray(arr, mode="L")
    else:
        img = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(img)
        num_rings = random.randint(3, 7)
        r_step = random.randint(7, 14)
        for i in range(1, num_rings + 1):
            cur_r = i * r_step
            if cur_r >= size // 2:
                break
            w = random.randint(2, 5)
            val = random.randint(190, 255)
            draw.ellipse([cx - cur_r, cy - cur_r, cx + cur_r, cy + cur_r], outline=val, width=w)
        if random.random() < 0.5:
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=255)
        return img


def make_dark_pseudo_ct_phantom(size: int = 128) -> Image.Image:
    """Псевдо-КТ анатомические контуры (фантом головы/тела, кортикальная кость, внутренние полости)."""
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2 + random.randint(-5, 5), size // 2 + random.randint(-5, 5)

    outer_rx = random.randint(40, 54)
    outer_ry = random.randint(42, 56)
    draw.ellipse([cx - outer_rx, cy - outer_ry, cx + outer_rx, cy + outer_ry], fill=random.randint(60, 110))
    draw.ellipse([cx - outer_rx, cy - outer_ry, cx + outer_rx, cy + outer_ry], outline=random.randint(220, 255), width=random.randint(2, 4))

    for _ in range(random.randint(2, 5)):
        ox = cx + random.randint(-outer_rx // 2, outer_rx // 2)
        oy = cy + random.randint(-outer_ry // 2, outer_ry // 2)
        orx = random.randint(6, 16)
        ory = random.randint(6, 16)
        val = random.choice([0, random.randint(140, 255), random.randint(30, 80)])
        draw.ellipse([ox - orx, oy - ory, ox + orx, oy + ory], fill=val)
        if random.random() < 0.5:
            draw.ellipse([ox - orx, oy - ory, ox + orx, oy + ory], outline=255, width=random.randint(1, 3))

    for _ in range(random.randint(1, 4)):
        lx = cx + random.randint(-outer_rx + 10, outer_rx - 10)
        ly = cy + random.randint(-outer_ry + 10, outer_ry - 10)
        lr = random.randint(2, 5)
        draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=255)

    return img


def make_dark_gratings_dots(size: int = 128) -> Image.Image:
    """1D/2D дифракционные решетки, сетки и точечные матрицы (пины) на темном фоне."""
    mode = random.choice(["1d_grating", "2d_grid", "dot_array"])
    arr = np.zeros((size, size), dtype=np.uint8)
    if mode == "1d_grating":
        period = random.choice([3, 4, 6, 8, 12, 16, 24])
        duty = random.uniform(0.3, 0.7)
        direction = random.choice(["h", "v", "diag"])
        val = random.randint(200, 255)
        for y in range(size):
            for x in range(size):
                coord = y if direction == "h" else (x if direction == "v" else (x + y))
                if (coord % period) < int(period * duty):
                    arr[y, x] = val
    elif mode == "2d_grid":
        period = random.choice([8, 12, 16, 24])
        w = random.randint(1, 4)
        val = random.randint(200, 255)
        for y in range(size):
            if (y % period) < w:
                arr[y, :] = val
        for x in range(size):
            if (x % period) < w:
                arr[:, x] = val
    else:
        spacing = random.choice([8, 12, 16, 20])
        r = random.randint(1, 4)
        val = random.randint(220, 255)
        img = Image.new("L", (size, size), color=0)
        draw = ImageDraw.Draw(img)
        offset_x = random.randint(0, spacing - 1)
        offset_y = random.randint(0, spacing - 1)
        for y in range(offset_y, size, spacing):
            for x in range(offset_x, size, spacing):
                draw.ellipse([x - r, y - r, x + r, y + r], fill=val)
        return img
    return Image.fromarray(arr, mode="L")


def generate_dark_contrast_mask(size: int = 128) -> torch.Tensor:
    """Генерация высококонтрастной оптической маски со строго темным фоном (0.0).
    
    Включает 6 категорий:
    1. Геометрические силуэты (звезды, многоугольники, шестерни, кресты, ромбы, рамки).
    2. Символы и типографика (греческий алфавит, буквы, цифры, мат. операторы, слова).
    3. Зонные пластинки Френеля (бинарные и синусоидальные с квадратичной фазой r_n ~ sqrt(n)).
    4. Радиальные кольца и спицы (звезда Сименса, кольца, спирали).
    5. Псевдо-КТ контуры (анатомические фантомы Shepp-Logan, костные и тканевые слои).
    6. Дифракционные решетки и точечные матрицы (1D/2D periodic gratings, pinhole array).
    """
    gen_fn = random.choice([
        make_dark_geometric,
        make_dark_symbols,
        make_dark_fresnel_zone_plate,
        make_dark_radial_rings_spokes,
        make_dark_pseudo_ct_phantom,
        make_dark_gratings_dots,
    ])
    img = gen_fn(size)
    return ToTensor()(img)


def generate_synthetic_mask(size: int = 128) -> torch.Tensor:
    """Обратно-совместимая генерация контрастной синтетической маски с темным фоном."""
    return generate_dark_contrast_mask(size)


class DOEAugmentedDataset(Dataset):
    """Датасет для обучения и валидации ДОЭ.
    
    Поддерживает два режима:
    - mode='train': сбалансированная 50/50 группировка 4 независимых изображений:
      50% натуральные патчи DIV2K + 50% темные высококонтрастные маски (силуэты, символы, зонные пластинки).
    - mode='test': детерминированная загрузка фиксированных групп из папок group_*.
    """

    def __init__(
        self,
        folder: Union[str, Path, None] = None,
        mode: str = "train",
        p_synth: float = 0.50,
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

        # Сбалансированный генератор 50/50:
        # 20% - все 4 плоскости темные схемы (бенчмарк-режим конкурентов)
        # 20% - все 4 плоскости натуральные фото DIV2K (фото-режим)
        # 60% - каждая плоскость независимо выбирается 50/50
        dice = random.random()
        channels = []
        if dice < 0.20:
            for _ in range(4):
                channels.append(generate_dark_contrast_mask(self.grid_size))
        elif dice < 0.40:
            for _ in range(4):
                idx = random.randint(0, len(self.patch_files) - 1)
                channels.append(self._load_patch(self.patch_files[idx]))
        else:
            for _ in range(4):
                if random.random() < self.p_synth:
                    channels.append(generate_dark_contrast_mask(self.grid_size))
                else:
                    idx = random.randint(0, len(self.patch_files) - 1)
                    channels.append(self._load_patch(self.patch_files[idx]))

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
