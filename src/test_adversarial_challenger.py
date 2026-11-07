import sys
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure code/src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from doe_network import (
    WAVELENGTH, PIXEL_SIZE, GRID_SIZE, Z_DISTANCES, device,
    DOENetwork, FresnelPropagation, CompositeLoss, fresnel_propagation,
    calculate_metrics, normalize_amplitude
)

def run_adversarial_suite():
    print("=" * 80)
    print("CHALLENGER M1-1: ADVERSARIAL & NUMERICAL STRESS TESTING SUITE")
    print("=" * 80)
    print(f"Device: {device}, PyTorch: {torch.__version__}")
    
    failures = []
    
    # =========================================================================
    # SUITE 1: Zero / Completely Dark Inputs (CompositeLoss & Backprop)
    # =========================================================================
    print("\n[SUITE 1] Zero / Completely Dark Inputs Stress Testing")
    criterion = CompositeLoss()
    model = DOENetwork()
    propagator = FresnelPropagation()
    
    test_cases_zeros = [
        ("All-zeros target [4, 4, 128, 128]", torch.zeros(4, 4, 128, 128)),
        ("All-zeros pred & target [2, 4, 128, 128]", torch.zeros(2, 4, 128, 128)),
        ("Near-zero target (1e-12) [2, 4, 128, 128]", torch.full((2, 4, 128, 128), 1e-12)),
        ("Subnormal float32 target (1e-40) [2, 4, 128, 128]", torch.full((2, 4, 128, 128), 1e-40)),
        ("Single non-zero pixel in target [2, 4, 128, 128]", None),
    ]
    
    # 1.1 All-zeros target with model forward + backward
    model.zero_grad()
    zero_target = torch.zeros(4, 4, 128, 128)
    phase = model(zero_target)
    rec = propagator(phase, normalize=True)
    loss, l1, ssim_val, pcc_val = criterion(rec, zero_target)
    
    print(f"  All-zeros target -> Loss: {loss.item():.6f}, L1: {l1.item():.6f}, SSIM: {ssim_val.item():.6f}, PCC: {pcc_val.item():.6f}")
    if torch.isnan(loss) or torch.isinf(loss):
        failures.append("SUITE 1: Loss is NaN/Inf on all-zeros target")
    
    loss.backward()
    nan_grads = [name for name, p in model.named_parameters() if p.grad is not None and torch.isnan(p.grad).any()]
    inf_grads = [name for name, p in model.named_parameters() if p.grad is not None and torch.isinf(p.grad).any()]
    if nan_grads:
        failures.append(f"SUITE 1: NaN gradients on all-zeros target in {nan_grads}")
    if inf_grads:
        failures.append(f"SUITE 1: Inf gradients on all-zeros target in {inf_grads}")
    print(f"  Gradients checked on all-zeros target: NaN={len(nan_grads)}, Inf={len(inf_grads)}")

    # 1.2 Both pred and target are all-zeros directly in CompositeLoss
    p_zero = torch.zeros(2, 4, 128, 128, requires_grad=True)
    t_zero = torch.zeros(2, 4, 128, 128)
    loss_zz, l1_zz, ssim_zz, pcc_zz = criterion(p_zero, t_zero)
    loss_zz.backward()
    p_grad_finite = torch.isfinite(p_zero.grad).all().item()
    print(f"  Both pred & target all-zeros -> Loss: {loss_zz.item():.6f}, p_grad_finite: {p_grad_finite}")
    if not p_grad_finite or torch.isnan(loss_zz) or torch.isinf(loss_zz):
        failures.append("SUITE 1: Loss or pred.grad non-finite when both pred and target are zero")

    # 1.3 Target with single hot pixel (impulse)
    t_impulse = torch.zeros(2, 4, 128, 128)
    t_impulse[:, :, 64, 64] = 1.0
    model.zero_grad()
    phase = model(t_impulse)
    rec = propagator(phase, normalize=True)
    loss_imp, _, ssim_imp, pcc_imp = criterion(rec, t_impulse)
    loss_imp.backward()
    imp_grad_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    print(f"  Impulse target -> Loss: {loss_imp.item():.6f}, SSIM: {ssim_imp.item():.6f}, PCC: {pcc_imp.item():.6f}, Grads finite: {imp_grad_finite}")
    if not imp_grad_finite:
        failures.append("SUITE 1: Gradients non-finite on impulse target")

    # =========================================================================
    # SUITE 2: Constant & Uniform Inputs (Zero Variance Denominators)
    # =========================================================================
    print("\n[SUITE 2] Constant & Uniform Inputs (Zero Variance Denominators)")
    constant_values = [0.0, 1.0, 0.5, 1e-5, 0.99999, 100.0]
    
    for val in constant_values:
        t_const = torch.full((2, 4, 128, 128), val)
        p_const = torch.full((2, 4, 128, 128), val, requires_grad=True)
        
        # Test directly in CompositeLoss
        loss_c, l1_c, ssim_c, pcc_c = criterion(p_const, t_const)
        loss_c.backward()
        g_finite = torch.isfinite(p_const.grad).all().item()
        
        print(f"  Constant val={val:8.5f} -> Loss: {loss_c.item():.6f}, SSIM: {ssim_c.item():.4f}, PCC: {pcc_c.item():.4f}, Grad finite: {g_finite}")
        if torch.isnan(loss_c) or torch.isinf(loss_c):
            failures.append(f"SUITE 2: Loss NaN/Inf on constant {val}")
        if not g_finite:
            failures.append(f"SUITE 2: Grad non-finite on constant {val}")
            
        # Test calculate_metrics on identical constants
        pcc_m, ssim_m = calculate_metrics(t_const[0, 0], t_const[0, 0])
        if math.isnan(pcc_m) or math.isinf(pcc_m) or math.isnan(ssim_m) or math.isinf(ssim_m):
            failures.append(f"SUITE 2: calculate_metrics returned NaN/Inf for identical constant {val}")
            
    # Test calculate_metrics on two DIFFERENT constants (e.g. 0.2 and 0.8)
    t_c1 = torch.full((128, 128), 0.2)
    t_c2 = torch.full((128, 128), 0.8)
    pcc_diff, ssim_diff = calculate_metrics(t_c1, t_c2)
    print(f"  calculate_metrics on diff constants (0.2 vs 0.8) -> PCC: {pcc_diff}, SSIM: {ssim_diff}")
    if math.isnan(pcc_diff) or math.isinf(pcc_diff) or math.isnan(ssim_diff) or math.isinf(ssim_diff):
        failures.append("SUITE 2: calculate_metrics returned NaN/Inf for different constants")

    # =========================================================================
    # SUITE 3: Extreme Phase Bounds [-pi, +pi] & Adversarial Attack
    # =========================================================================
    print("\n[SUITE 3] Phase Mask Bounding [-pi, +pi] & Adversarial Optimization")
    model.eval()
    
    # 3.1 Massive input values (positive & negative extremes)
    extreme_inputs = [
        ("Huge positive (+1e9)", torch.full((2, 4, 128, 128), 1e9)),
        ("Huge negative (-1e9)", torch.full((2, 4, 128, 128), -1e9)),
        ("Extreme float32 (+1e38)", torch.full((2, 4, 128, 128), 1e38)),
        ("Extreme float32 (-1e38)", torch.full((2, 4, 128, 128), -1e38)),
        ("Gaussian high-variance (mean=0, std=1000)", torch.randn(2, 4, 128, 128) * 1000.0),
        ("Adversarial salt-and-pepper (+1e6 / -1e6)", (torch.randint(0, 2, (2, 4, 128, 128)).float() * 2 - 1) * 1e6)
    ]
    
    pi = torch.pi
    for name, ext_in in extreme_inputs:
        with torch.no_grad():
            phase = model(ext_in)
            p_min = phase.min().item()
            p_max = phase.max().item()
            is_bounded = (p_min >= -pi - 1e-6) and (p_max <= pi + 1e-6)
            has_nan = torch.isnan(phase).any().item()
            has_inf = torch.isinf(phase).any().item()
            print(f"  {name:<45} -> min: {p_min:9.6f}, max: {p_max:9.6f}, bounded: {is_bounded}, NaN={has_nan}")
            if not is_bounded or has_nan or has_inf:
                failures.append(f"SUITE 3: Phase bounds breached for {name}: min={p_min}, max={p_max}")
                
    # 3.2 Projected Gradient Attack to actively breach [-pi, pi]
    print("  Running adversarial gradient ascent on input to breach phase bounds...")
    adv_input = torch.rand(1, 4, 128, 128, requires_grad=True)
    opt_adv = torch.optim.SGD([adv_input], lr=100.0)
    for step in range(30):
        opt_adv.zero_grad()
        phase = model(adv_input)
        # maximize max phase
        loss_adv = -phase.max()
        loss_adv.backward()
        opt_adv.step()
    
    with torch.no_grad():
        final_phase = model(adv_input)
        f_max = final_phase.max().item()
        f_min = final_phase.min().item()
        print(f"  After 30 steps of gradient ascent -> max phase: {f_max:.7f} (pi = {float(pi):.7f}), min phase: {f_min:.7f}")
        if f_max > float(pi) + 1e-6 or f_min < -float(pi) - 1e-6:
            failures.append(f"SUITE 3: Adversarial attack successfully breached pi bounds! max={f_max}")

    # =========================================================================
    # SUITE 4: Scale Invariance & Fresnel Normalization
    # =========================================================================
    print("\n[SUITE 4] Scale Invariance & Amplitude Normalization (|K| max == 1.0)")
    scales = [1e-8, 1e-4, 0.01, 0.5, 1.0, 2.0, 100.0, 1e6]
    
    base_target = torch.rand(2, 4, 128, 128)
    for s in scales:
        scaled_target = base_target * s
        # Normalization as done in pipeline
        norm_target = normalize_amplitude(scaled_target)
        norm_max = norm_target.amax(dim=(-2, -1))
        all_one = torch.allclose(norm_max, torch.ones_like(norm_max), atol=1e-4)
        print(f"  Target scaled by {s:10.2e} -> normalized max range: [{norm_max.min().item():.5f}, {norm_max.max().item():.5f}], matches 1.0: {all_one}")
        if not all_one:
            failures.append(f"SUITE 4: Target scale invariance failed for scale {s}")

    # Test Fresnel output normalization under diverse phase configurations
    phase_scenarios = [
        ("All-zero phase (flat wavefront)", torch.zeros(2, 1, 128, 128)),
        ("Constant pi phase", torch.full((2, 1, 128, 128), float(pi))),
        ("Random uniform phase [-pi, pi]", (torch.rand(2, 1, 128, 128) * 2 - 1) * float(pi)),
        ("Quadratic phase / Fresnel lens", None)
    ]
    
    # Generate quadratic phase (strong focus)
    coords = torch.linspace(-64, 63, 128)
    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    quad_phase = ((X**2 + Y**2) * 0.1).remainder(2 * float(pi)) - float(pi)
    phase_scenarios[3] = ("Quadratic phase / focusing lens", quad_phase.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1))

    for name, ph in phase_scenarios:
        rec = propagator(ph, normalize=True)
        rec_max = rec.amax(dim=(-2, -1))
        norm_ok = torch.allclose(rec_max, torch.ones_like(rec_max), atol=1e-5)
        rec_has_nan = torch.isnan(rec).any().item()
        rec_has_inf = torch.isinf(rec).any().item()
        print(f"  Fresnel with {name:<35} -> Max: {rec_max.mean().item():.6f}, NaN: {rec_has_nan}, Inf: {rec_has_inf}, All == 1.0: {norm_ok}")
        if not norm_ok or rec_has_nan or rec_has_inf:
            failures.append(f"SUITE 4: Fresnel normalization failed for {name}")

    # =========================================================================
    # SUITE 5: Complex / Extreme Corner Cases (Batch 1, Non-Square, Memory)
    # =========================================================================
    print("\n[SUITE 5] Batch Size 1, Eval/Train Mode Switch, and Gradient Clipping")
    
    # 5.1 Batch Size 1 in training mode (BatchNorm check)
    model.train()
    b1_target = torch.rand(1, 4, 128, 128)
    model.zero_grad()
    b1_phase = model(b1_target)
    b1_rec = propagator(b1_phase, normalize=True)
    b1_loss, _, _, _ = criterion(b1_rec, b1_target)
    b1_loss.backward()
    b1_grad_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    print(f"  Batch size B=1 in train() mode -> Loss: {b1_loss.item():.4f}, Grads finite: {b1_grad_finite}")
    if not b1_grad_finite:
        failures.append("SUITE 5: Gradients non-finite for B=1 in train() mode")

    # 5.2 5-step AdamW optimization convergence test
    print("  Testing 5-step optimization dynamics on synthetic pattern...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target_dyn = torch.zeros(2, 4, 128, 128)
    target_dyn[:, :, 32:96, 32:96] = 1.0 # square pattern
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        p = model(target_dyn)
        r = propagator(p, normalize=True)
        l, l1_val, ssim_val, pcc_val = criterion(r, target_dyn)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(l.item())
        print(f"    Step {step+1}: Loss = {l.item():.4f} (L1: {l1_val.item():.4f}, SSIM: {ssim_val.item():.4f}, PCC: {pcc_val.item():.4f})")
    
    if any(math.isnan(x) or math.isinf(x) for x in losses):
        failures.append("SUITE 5: Loss exploded or became NaN during 5-step training")
    if losses[-1] > losses[0] * 1.5:
        failures.append("SUITE 5: Training diverged significantly in 5 steps")

    # =========================================================================
    # SUMMARY & VERDICT
    # =========================================================================
    print("\n" + "=" * 80)
    print("CHALLENGER STRESS TEST SUITE RESULTS")
    print("=" * 80)
    if failures:
        print(f"FAILURES DETECTED ({len(failures)}):")
        for f in failures:
            print(f"  [X] {f}")
        print("\nVERDICT: REQUEST_CHANGES")
        return False
    else:
        print("ALL ADVERSARIAL STRESS TESTS PASSED SUCCESSFULLY (0 FAILURES)!")
        print("VERDICT: APPROVE")
        return True

if __name__ == "__main__":
    success = run_adversarial_suite()
    sys.exit(0 if success else 1)
