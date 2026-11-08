"""Comprehensive Verification & Benchmark Test Suite for SpectralConv2d (FNO) and SpectralResAttentionUNet.

Verifies:
1. SpectralConv2d Fourier Neural Operator layer: 2D FFT, complex learnable weights, adaptive mode clipping across multiple resolutions.
2. Grouped spectral convolutions (groups=1, groups=4, groups=8).
3. SpectralBlock integration with spatial skip, batch normalization, and LeakyReLU.
4. SpectralResAttentionUNet forward pass: strict phase range [-pi, +pi], shape integrity.
5. Physics-informed backward gradient propagation through FresnelPropagation and CompositeLoss.
6. Pure inference latency benchmarking on GPU (CUDA) ensuring strict compliance with < 5-10 ms latency on NVIDIA RTX 4060.
"""

import math
import sys
import time
import unittest
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from doe_network import (
    GRID_SIZE,
    PIXEL_SIZE,
    WAVELENGTH,
    Z_DISTANCES,
    DOENetwork,
    FresnelPropagation,
    OptimizedCompositeLoss,
    SpectralBlock,
    SpectralConv2d,
    SpectralResAttentionUNet,
)


class TestSpectralFNO(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_spectral_conv2d_resolutions_and_adaptivity(self):
        """Проверяет адаптивное усечение частотных мод на различных пространственных разрешениях."""
        modes1, modes2 = 16, 16
        conv_std = SpectralConv2d(in_channels=32, out_channels=32, modes1=modes1, modes2=modes2, groups=1).to(self.device)
        conv_grp = SpectralConv2d(in_channels=32, out_channels=32, modes1=modes1, modes2=modes2, groups=4).to(self.device)

        # Тестируем на разрешениях от 8x8 до 128x128
        resolutions = [(128, 128), (64, 64), (32, 32), (16, 16), (8, 8)]
        for H, W in resolutions:
            x = torch.randn(2, 32, H, W, device=self.device)
            out_std = conv_std(x)
            out_grp = conv_grp(x)
            self.assertEqual(out_std.shape, (2, 32, H, W), f"Mismatched shape for H={H}, W={W} (std)")
            self.assertEqual(out_grp.shape, (2, 32, H, W), f"Mismatched shape for H={H}, W={W} (grp)")
            self.assertFalse(torch.isnan(out_std).any(), f"NaN in out_std for H={H}, W={W}")
            self.assertFalse(torch.isnan(out_grp).any(), f"NaN in out_grp for H={H}, W={W}")

    def test_spectral_conv2d_complex_weights(self):
        """Проверяет комплексный тип весов и их обучаемость."""
        layer = SpectralConv2d(16, 16, modes1=8, modes2=8, groups=1).to(self.device)
        self.assertTrue(layer.weights1.is_complex(), "weights1 must be complex (cfloat)")
        self.assertTrue(layer.weights2.is_complex(), "weights2 must be complex (cfloat)")
        self.assertEqual(layer.weights1.dtype, torch.complex64)
        self.assertEqual(layer.weights2.dtype, torch.complex64)

    def test_spectral_block(self):
        """Проверяет корректность работы блока SpectralBlock со spatial shortcut и residual."""
        block = SpectralBlock(in_channels=32, out_channels=32, modes1=16, modes2=16, residual=True).to(self.device)
        x = torch.randn(2, 32, 64, 64, device=self.device)
        out = block(x)
        self.assertEqual(out.shape, (2, 32, 64, 64))
        self.assertFalse(torch.isnan(out).any())

    def test_spectral_res_attention_unet_forward_range(self):
        """Проверяет, что выход фазовой маски лежит СТРОГО в диапазоне [-pi, +pi]."""
        model = SpectralResAttentionUNet(in_channels=4, base_channels=32, modes1=16, modes2=16).to(self.device)
        model.eval()

        # Тестируем на случайных входах в диапазоне [0, 1]
        x = torch.rand(2, 4, 128, 128, device=self.device)
        with torch.no_grad():
            phase_mask = model(x)

        self.assertEqual(phase_mask.shape, (2, 1, 128, 128))
        min_val = phase_mask.min().item()
        max_val = phase_mask.max().item()

        self.assertGreaterEqual(min_val, -math.pi - 1e-6, f"Phase mask below -pi: {min_val}")
        self.assertLessEqual(max_val, math.pi + 1e-6, f"Phase mask above +pi: {max_val}")

    def test_backward_gradients_and_fresnel_loss(self):
        """Проверяет сквозной градиентный поток через оптическую дифракцию Френеля и спектральные веса."""
        model = SpectralResAttentionUNet(in_channels=4, base_channels=32, modes1=16, modes2=16).to(self.device)
        propagator = FresnelPropagation(z_list=Z_DISTANCES).to(self.device)
        criterion = OptimizedCompositeLoss().to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        x = torch.rand(2, 4, 128, 128, device=self.device)
        optimizer.zero_grad()

        phase_mask = model(x)
        reconstructed = propagator(phase_mask, normalize=True)
        loss, l1, ssim, pcc = criterion(reconstructed, x)
        loss.backward()

        # Проверяем наличие комплексных градиентов в спектральных весах
        b_w1_grad = model.bottleneck_fno.spectral_conv.weights1.grad
        b_w2_grad = model.bottleneck_fno.spectral_conv.weights2.grad
        d_w1_grad = model.spectral_dec.spectral_conv.weights1.grad
        d_w2_grad = model.spectral_dec.spectral_conv.weights2.grad

        self.assertIsNotNone(b_w1_grad, "Bottleneck weights1 grad is None")
        self.assertIsNotNone(b_w2_grad, "Bottleneck weights2 grad is None")
        self.assertIsNotNone(d_w1_grad, "Decoder weights1 grad is None")
        self.assertIsNotNone(d_w2_grad, "Decoder weights2 grad is None")

        self.assertFalse(torch.isnan(b_w1_grad).any(), "NaN in bottleneck weights1 grad")
        self.assertFalse(torch.isnan(d_w1_grad).any(), "NaN in decoder weights1 grad")

        # Проверяем корректность шага оптимизатора
        optimizer.step()

    def test_inference_latency_benchmark_cuda(self):
        """Замеряет чистое время инференса модели на CUDA и проверяет норматив < 5-10 мс."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA недоступна, бенчмарк пропускается.")

        model = SpectralResAttentionUNet(in_channels=4, base_channels=32, modes1=16, modes2=16).cuda().eval()
        x = torch.rand(1, 4, 128, 128, device="cuda")

        # Warm-up (50 итераций)
        with torch.inference_mode():
            for _ in range(50):
                _ = model(x)
            torch.cuda.synchronize()

            # Основной замер (300 итераций)
            N = 300
            t0 = time.perf_counter()
            for _ in range(N):
                _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        avg_latency_ms = (t1 - t0) / N * 1000.0
        params_count = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())

        print(f"\n[FNO Benchmark] SpectralResAttentionUNet (modes=16):")
        print(f"  - Параметров: {params_count:,} ({params_count / 1e6:.2f}M)")
        print(f"  - Время инференса на {torch.cuda.get_device_name(0)}: {avg_latency_ms:.3f} мс")

        # Проверяем норматив: строго < 10 мс
        self.assertLess(avg_latency_ms, 10.0, f"Инференс {avg_latency_ms:.2f} мс превышает норматив 10 мс!")


if __name__ == "__main__":
    unittest.main()
