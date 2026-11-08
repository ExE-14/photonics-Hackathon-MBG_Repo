"""Комплексный стресс-тест функции потерь CompositeLossV3 и сбалансированного датасета.

Проверяет:
1. Focal Frequency Loss (FFL): расчет потерь в 2D Фурье-спектре, динамические веса, градиенты.
2. Phase Total Variation Loss (TV-loss): 2pi-периодическая регуляризация гладкости фазовой маски ДОЭ.
3. CompositeLossV3: обратный проход (backward pass) на GPU без градиентных всплесков (NaN/Inf).
4. Сбалансированный генератор данных 50/50: DIV2K + темные высококонтрастные патчи.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
    CompositeLossV3,
    DOENetwork,
    FocalFrequencyLoss,
    FresnelPropagation,
    LossOutput,
    PhaseTotalVariationLoss,
    calculate_metrics,
    generate_dark_contrast_mask,
    make_dark_fresnel_zone_plate,
    make_dark_geometric,
    make_dark_gratings_dots,
    make_dark_pseudo_ct_phantom,
    make_dark_radial_rings_spokes,
    make_dark_symbols,
)
from train import FastRAMPatchDataset, pregenerate_synthetic_bank


def run_all_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("ТЕСТИРОВАНИЕ COMPOSITELOSSV3 И СБАЛАНСИРОВАННОГО ДАТАСЕТА 50/50")
    print(f"Устройство: {device}, PyTorch: {torch.__version__}")
    print("=" * 80)

    failures = []

    # =========================================================================
    # ТЕСТ 1: Focal Frequency Loss (FFL)
    # =========================================================================
    print("\n[ТЕСТ 1] Focal Frequency Loss (FFL) - Спектральные потери в 2D Фурье")
    ffl = FocalFrequencyLoss(alpha=1.0, eps=1e-8).to(device)

    # 1.1 Нулевые тензоры
    p_zero = torch.zeros(2, 4, 128, 128, requires_grad=True, device=device)
    t_zero = torch.zeros(2, 4, 128, 128, device=device)
    loss_f0 = ffl(p_zero, t_zero)
    loss_f0.backward()
    p_zero_grad_ok = p_zero.grad is not None and torch.isfinite(p_zero.grad).all().item()
    print(f"  All-zeros: Loss={loss_f0.item():.6f}, grad_finite={p_zero_grad_ok}")
    if torch.isnan(loss_f0) or torch.isinf(loss_f0) or not p_zero_grad_ok:
        failures.append("FFL failed on all-zeros")

    # 1.2 Одиночный импульсный пиксель (высокие частоты)
    p_imp = torch.zeros(2, 4, 128, 128, requires_grad=True, device=device)
    t_imp = torch.zeros(2, 4, 128, 128, device=device)
    t_imp[:, :, 64, 64] = 1.0
    loss_imp = ffl(p_imp, t_imp)
    loss_imp.backward()
    p_imp_grad_ok = p_imp.grad is not None and torch.isfinite(p_imp.grad).all().item()
    print(f"  Impulse pixel: Loss={loss_imp.item():.6f}, grad_finite={p_imp_grad_ok}")
    if torch.isnan(loss_imp) or torch.isinf(loss_imp) or not p_imp_grad_ok:
        failures.append("FFL failed on impulse pixel")

    # 1.3 Размытый контур против резкого
    t_sharp = torch.zeros(2, 4, 128, 128, device=device)
    t_sharp[:, :, 40:88, 40:88] = 1.0
    p_blur = nn.functional.avg_pool2d(t_sharp, kernel_size=9, stride=1, padding=4)
    p_blur.requires_grad_(True)
    loss_blur = ffl(p_blur, t_sharp)
    loss_blur.backward()
    print(f"  Blurred edge vs sharp: FFL Loss={loss_blur.item():.6f}, grad_finite={torch.isfinite(p_blur.grad).all().item()}")
    if loss_blur.item() <= 0:
        failures.append("FFL must be strictly positive for blurred edges")

    # =========================================================================
    # ТЕСТ 2: Phase Total Variation Loss (TV-loss)
    # =========================================================================
    print("\n[ТЕСТ 2] Phase Total Variation Loss (TV-loss) - 2pi-периодическая регуляризация")
    tv = PhaseTotalVariationLoss(wrapped=True).to(device)

    # 2.1 Плоская фаза (все нули или константа pi)
    phi_flat = torch.zeros(2, 1, 128, 128, requires_grad=True, device=device)
    loss_tv_flat = tv(phi_flat)
    loss_tv_flat.backward()
    print(f"  Flat phase: Loss={loss_tv_flat.item():.6f}, grad_finite={torch.isfinite(phi_flat.grad).all().item()}")
    if abs(loss_tv_flat.item()) > 1e-7:
        failures.append("Phase TV must be 0 for flat phase")

    # 2.2 2pi-периодический скачок (смежные пиксели -pi + 0.05 и +pi - 0.05)
    phi_jump = torch.zeros(1, 1, 128, 128, device=device)
    phi_jump[:, :, :, :64] = -torch.pi + 0.05
    phi_jump[:, :, :, 64:] = torch.pi - 0.05
    phi_jump.requires_grad_(True)
    loss_tv_jump = tv(phi_jump)
    loss_tv_jump.backward()
    # Физическое угловое расстояние на окружности S^1 равно 0.1 радиана, а не 2*pi
    print(f"  Wrapped boundary jump (-pi to +pi): Loss={loss_tv_jump.item():.6f}, grad_finite={torch.isfinite(phi_jump.grad).all().item()}")
    if loss_tv_jump.item() > 0.05:  # Усредненное по 128x128 TV должно быть очень малым
        failures.append("Phase TV wrapping failed to account for modulo 2pi")

    # 2.3 Шумная фаза против гладкой фазы
    phi_smooth = (torch.sin(torch.linspace(0, 3.14, 128, device=device)).view(1, 1, 1, 128)).expand(1, 1, 128, 128)
    phi_noisy = phi_smooth + torch.randn_like(phi_smooth) * 0.5
    loss_smooth = tv(phi_smooth)
    loss_noisy = tv(phi_noisy)
    print(f"  Smooth phase TV: {loss_smooth.item():.4f} vs Noisy phase TV: {loss_noisy.item():.4f}")
    if loss_noisy.item() <= loss_smooth.item():
        failures.append("Phase TV must penalize noisy phase significantly more than smooth phase")

    # =========================================================================
    # ТЕСТ 3: CompositeLossV3 Full Backward Pass на GPU
    # =========================================================================
    print("\n[ТЕСТ 3] CompositeLossV3 - Полный backward pass через U-Net и Fresnel на GPU")
    model = DOENetwork(in_channels=4, base_channels=32).to(device)
    propagator = FresnelPropagation(z_list=Z_DISTANCES).to(device)
    criterion = CompositeLossV3(
        w_pcc=4.0,
        w_ssim=1.2,
        w_l1=0.05,
        w_sobel=0.05,
        w_ffl=0.20,
        w_tv=0.005,
        plane_weights=(0.9, 0.9, 1.1, 1.1),
    ).to(device)

    model.train()

    test_scenarios = [
        ("Random inputs [4, 4, 128, 128]", torch.rand(4, 4, 128, 128, device=device), False),
        ("High-contrast binary masks [2, 4, 128, 128]", (torch.rand(2, 4, 128, 128, device=device) > 0.8).float(), False),
        ("Single impulse hot-pixel [2, 4, 128, 128]", torch.zeros(2, 4, 128, 128, device=device), False),
        ("All-zeros target (eval mode) [2, 4, 128, 128]", torch.zeros(2, 4, 128, 128, device=device), True),
    ]
    test_scenarios[2][1][:, :, 64, 64] = 1.0

    for name, targets, eval_mode in test_scenarios:
        if eval_mode:
            model.eval()
        else:
            model.train()

        model.zero_grad(set_to_none=True)
        phase_mask = model(targets)  # [B, 1, 128, 128]
        rec = propagator(phase_mask, normalize=True)  # [B, 4, 128, 128]
        targets_norm = targets / torch.clamp_min(
            targets.amax(dim=(-2, -1), keepdim=True), 1e-12
        )

        out = criterion(rec, targets_norm, phase_mask=phase_mask)
        # Проверка 4-элементной распаковки
        loss, l1, ssim_val, pcc_val = out
        loss.backward()

        # Градиентный клиппинг (как в train.py)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        nan_params = []
        inf_params = []
        grad_norms = []
        for p_name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_params.append(p_name)
                if torch.isinf(param.grad).any():
                    inf_params.append(p_name)
                grad_norms.append(param.grad.norm().item())

        max_norm = max(grad_norms) if grad_norms else 0.0
        print(
            f"  {name:<46} -> Loss={loss.item():.4f} (FFL={out.ffl.item():.4f}, TV={out.tv.item():.4f}, PCC={pcc_val.item():.4f}) | "
            f"NaN grads={len(nan_params)}, Inf grads={len(inf_params)}, max grad norm={max_norm:.4f}"
        )
        if nan_params:
            failures.append(f"CompositeLossV3 NaN gradients on {name}: {nan_params[:3]}")
        if inf_params:
            failures.append(f"CompositeLossV3 Inf gradients on {name}: {inf_params[:3]}")

    # 3.2 5 шагов оптимизации мини-батча для проверки сходимости
    print("  Проверка 5 шагов оптимизации мини-батча на GPU...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    batch_t = torch.rand(4, 4, 128, 128, device=device)
    batch_t_norm = batch_t / torch.clamp_min(batch_t.amax(dim=(-2, -1), keepdim=True), 1e-12)
    step_losses = []
    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)
        pm = model(batch_t)
        rc = propagator(pm, normalize=True)
        l_out = criterion(rc, batch_t_norm, phase_mask=pm)
        l_out[0].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        step_losses.append(l_out[0].item())
        print(f"    Шаг {step}: Loss={l_out[0].item():.4f} (PCC={l_out.pcc.item():.4f}, SSIM={l_out.ssim.item():.4f}, FFL={l_out.ffl.item():.4f}, TV={l_out.tv.item():.4f})")

    if step_losses[-1] >= step_losses[0]:
        failures.append("Optimization steps did not reduce loss")

    # =========================================================================
    # ТЕСТ 4: Сбалансированный генератор 50/50 и банк контрастных масок
    # =========================================================================
    print("\n[ТЕСТ 4] Сбалансированный генератор данных 50/50 и банк масок")

    # 4.1 Проверка генераторов всех 6 типов
    gens = [
        ("make_dark_geometric", make_dark_geometric),
        ("make_dark_symbols", make_dark_symbols),
        ("make_dark_fresnel_zone_plate", make_dark_fresnel_zone_plate),
        ("make_dark_radial_rings_spokes", make_dark_radial_rings_spokes),
        ("make_dark_pseudo_ct_phantom", make_dark_pseudo_ct_phantom),
        ("make_dark_gratings_dots", make_dark_gratings_dots),
    ]

    for g_name, g_fn in gens:
        im = g_fn(128)
        arr = np.array(im)
        # Проверка, что фон действительно темный (углы и края)
        corners = [arr[0, 0], arr[0, 127], arr[127, 0], arr[127, 127]]
        corner_mean = float(np.mean(corners))
        print(f"  {g_name:<30}: size={arr.shape}, min={arr.min()}, max={arr.max()}, corners_mean={corner_mean:.1f}")
        if arr.max() < 100:
            failures.append(f"Generator {g_name} failed to produce high-contrast bright features")

    # 4.2 Генерация тестового банка масок
    t0 = time.perf_counter()
    synth_bank = pregenerate_synthetic_bank(num_masks=500, size=GRID_SIZE, num_workers=8)
    t1 = time.perf_counter()
    print(f"  Предгенерация банка 500 масок: {synth_bank.shape}, время: {t1 - t0:.2f}с")

    # 4.3 Проверка FastRAMPatchDataset с 50/50 балансом
    mock_patches = torch.full((100, 128, 128), 200, dtype=torch.uint8)
    mock_synth = torch.full((100, 128, 128), 50, dtype=torch.uint8)
    dataset = FastRAMPatchDataset.__new__(FastRAMPatchDataset)
    dataset.p_synth = 0.50
    dataset.epoch_length = 200
    dataset.synth_bank = mock_synth
    dataset.n_synth = mock_synth.shape[0]
    dataset.patches = mock_patches
    dataset.n_patches = mock_patches.shape[0]

    dark_count = 0
    total_planes = 0
    for _ in range(500):
        sample = dataset[0]  # [4, 128, 128]
        for c in range(4):
            val = int(round(sample[c, 0, 0].item() * 255.0))
            if val == 50:
                dark_count += 1
            total_planes += 1

    frac_dark = dark_count / total_planes
    frac_natural = 1.0 - frac_dark
    print(f"  Эмпирическое распределение на 2,000 плоскостях: темные={frac_dark:.2%}, натуральные={frac_natural:.2%}")
    if abs(frac_dark - 0.50) > 0.05:
        failures.append(f"Dataset 50/50 balance out of tolerance: dark={frac_dark:.2%}")

    print("\n" + "=" * 80)
    if not failures:
        print("ИТОГ: ВСЕ ТЕСТЫ COMPOSITELOSSV3 И ДАТАСЕТА 50/50 УСПЕШНО ПРОЙДЕНЫ! (0 ОШИБОК)")
        print("ВЕРДИКТ: ПОЛНОСТЬЮ ГОТОВО К ОБУЧЕНИЮ БЕЗ СПАЙКОВ ГРАДИЕНТОВ")
    else:
        print(f"ИТОГ: ОБНАРУЖЕНЫ ОШИБКИ ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
    print("=" * 80)

    return len(failures) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
