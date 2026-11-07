import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure code/src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from doe_network import (
    WAVELENGTH, PIXEL_SIZE, GRID_SIZE, Z_DISTANCES, device,
    DOENetwork, FresnelPropagation, CompositeLoss, fresnel_propagation
)

def run_tests():
    print("=" * 80)
    print("CHALLENGER M1-2: PERFORMANCE, SCALING & GRADIENT HEALTH VERIFICATION")
    print("=" * 80)
    print(f"PyTorch Version: {torch.__version__}, Running on device: {device}")
    
    # -------------------------------------------------------------------------
    # TEST 1: Parameter Count & Architectural Verification (Bottleneck Resolution)
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Architectural Verification & Bottleneck Resolution")
    model = DOENetwork()
    model.train()
    
    # Test intermediate layer shapes
    dummy_input = torch.randn(2, 4, 128, 128)
    shapes = {}
    
    x0 = model.inc(dummy_input)
    shapes['inc (128x128)'] = tuple(x0.shape)
    
    x1 = model.down1(x0)
    shapes['down1 (64x64)'] = tuple(x1.shape)
    
    x2 = model.down2(x1)
    shapes['down2 (32x32)'] = tuple(x2.shape)
    
    x3 = model.down3(x2)
    shapes['down3 (16x16)'] = tuple(x3.shape)
    
    x4 = model.down4(x3)
    shapes['down4 (bottleneck 8x8)'] = tuple(x4.shape)
    
    u1 = model.conv_up1(torch.cat([model.up1(x4), x3], dim=1))
    shapes['conv_up1 (16x16)'] = tuple(u1.shape)
    
    u2 = model.conv_up2(torch.cat([model.up2(u1), x2], dim=1))
    shapes['conv_up2 (32x32)'] = tuple(u2.shape)
    
    u3 = model.conv_up3(torch.cat([model.up3(u2), x1], dim=1))
    shapes['conv_up3 (64x64)'] = tuple(u3.shape)
    
    u4 = model.conv_up4(torch.cat([model.up4(u3), x0], dim=1))
    shapes['conv_up4 (128x128)'] = tuple(u4.shape)
    
    out = model.final_conv(u4)
    shapes['final_conv (128x128)'] = tuple(out.shape)
    phase = torch.tanh(out) * torch.pi
    shapes['phase_mask'] = tuple(phase.shape)
    
    for stage, shape in shapes.items():
        print(f"  {stage:30s} -> {shape}")
        
    assert shapes['down4 (bottleneck 8x8)'] == (2, 512, 8, 8), \
        f"Bottleneck shape mismatch: expected (2, 512, 8, 8), got {shapes['down4 (bottleneck 8x8)']}"
    print("  -> PASSED: Bottleneck spatial resolution is strictly 8x8 with 512 channels.")
    
    # Parameter count verification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_tensors = list(model.named_parameters())
    num_param_tensors = len(param_tensors)
    
    print(f"\n  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Number of parameter tensors: {num_param_tensors}")
    
    assert total_params == 7849889, f"Unexpected parameter count: {total_params}"
    assert trainable_params == total_params, "Found non-trainable parameters in model"
    assert num_param_tensors == 56, f"Expected exactly 56 parameter tensors, got {num_param_tensors}"
    print("  -> PASSED: Parameter count is exactly 7,849,889 across 56 tensors (~7.85M).")
    
    # -------------------------------------------------------------------------
    # TEST 2: Full End-to-End Autograd Gradient Health Across All 56 Tensors
    # -------------------------------------------------------------------------
    print("\n[TEST 2] End-to-End Autograd Gradient Health Check (56 parameter tensors)")
    torch.manual_seed(42)
    model = DOENetwork()
    model.train()
    propagator = FresnelPropagation()
    criterion = CompositeLoss()
    
    # Input targets [B=4, 4, 128, 128]
    targets = torch.rand(4, 4, 128, 128)
    targets_norm = targets / (targets.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    
    model.zero_grad()
    phase_mask = model(targets_norm)
    rec_norm = propagator(phase_mask, normalize=True)
    loss, l1, ssim_val, pcc_val = criterion(rec_norm, targets_norm)
    
    print(f"  Forward pass successful. Loss: {loss.item():.4f} (L1: {l1.item():.4f}, SSIM: {ssim_val.item():.4f}, PCC: {pcc_val.item():.4f})")
    
    loss.backward()
    
    grad_health = []
    nan_count = 0
    inf_count = 0
    zero_count = 0
    none_count = 0
    
    print("\n  Detailed inspection of all 56 parameter tensors:")
    print(f"  {'#':<3} {'Parameter Name':<42} {'Shape':<22} {'Grad Norm':<12} {'Min':<10} {'Max':<10} {'Status'}")
    print("  " + "-" * 110)
    
    for idx, (name, param) in enumerate(model.named_parameters(), 1):
        if param.grad is None:
            none_count += 1
            status = "NONE"
            norm_str = "N/A"
            min_str = "N/A"
            max_str = "N/A"
        else:
            g = param.grad
            has_nan = torch.isnan(g).any().item()
            has_inf = torch.isinf(g).any().item()
            all_zero = (g == 0).all().item()
            g_norm = g.norm().item()
            g_min = g.min().item()
            g_max = g.max().item()
            
            if has_nan:
                nan_count += 1
                status = "NAN"
            elif has_inf:
                inf_count += 1
                status = "INF"
            elif all_zero:
                zero_count += 1
                status = "ZERO"
            else:
                status = "HEALTHY"
                
            norm_str = f"{g_norm:.4e}"
            min_str = f"{g_min:.2e}"
            max_str = f"{g_max:.2e}"
            
        print(f"  {idx:<3} {name:<42} {str(list(param.shape)):<22} {norm_str:<12} {min_str:<10} {max_str:<10} {status}")
        grad_health.append({
            'idx': idx,
            'name': name,
            'shape': list(param.shape),
            'status': status,
            'norm': norm_str
        })
        
    print("\n  Summary of Gradient Health:")
    print(f"    Total tensors: {num_param_tensors}")
    print(f"    None gradients: {none_count}")
    print(f"    NaN gradients:  {nan_count}")
    print(f"    Inf gradients:  {inf_count}")
    print(f"    All-zero grads: {zero_count}")
    print(f"    Healthy grads:  {num_param_tensors - none_count - nan_count - inf_count - zero_count}")
    
    assert none_count == 0, f"Found {none_count} parameter tensors with None gradient!"
    assert nan_count == 0, f"Found {nan_count} parameter tensors with NaN gradient!"
    assert inf_count == 0, f"Found {inf_count} parameter tensors with Inf gradient!"
    assert zero_count == 0, f"Found {zero_count} parameter tensors with all-zero gradient!"
    print("  -> PASSED: All 56 parameter tensors receive finite, non-zero, non-NaN, non-Inf gradients.")
    
    # -------------------------------------------------------------------------
    # TEST 3: Gradient Health Under Stress & Degenerate Conditions
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Gradient Health Stress Testing on Degenerate Inputs")
    degenerate_cases = [
        ("All Zeros (Black Image)", torch.zeros(2, 4, 128, 128)),
        ("All Ones (White Image)", torch.ones(2, 4, 128, 128)),
        ("Checkerboard Pattern", torch.kron(torch.tensor([[0.0, 1.0], [1.0, 0.0]]).repeat(64, 64), torch.ones(1, 1)).repeat(2, 4, 1, 1)),
        ("Identical Target Channels", torch.rand(2, 1, 128, 128).repeat(1, 4, 1, 1)),
        ("High Dynamic Range Spikes", torch.zeros(2, 4, 128, 128).scatter_(2, torch.randint(0, 128, (2, 4, 10, 128)), 1.0))
    ]
    
    for case_name, degen_target in degenerate_cases:
        model.zero_grad()
        target_norm = degen_target / (degen_target.amax(dim=(-2, -1), keepdim=True) + 1e-8)
        phase = model(target_norm)
        rec = propagator(phase, normalize=True)
        loss, l1, ssim_val, pcc_val = criterion(rec, target_norm)
        loss.backward()
        
        has_nan = any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any().item() for p in model.parameters() if p.grad is not None)
        all_none = any(p.grad is None for p in model.parameters())
        
        print(f"  Stress Case: {case_name:<30} -> Loss={loss.item():.4f}, SSIM={ssim_val.item():.4f}, PCC={pcc_val.item():.4f} | NaN={has_nan}, Inf={has_inf}")
        assert not has_nan, f"NaN gradient detected in stress case: {case_name}"
        assert not has_inf, f"Inf gradient detected in stress case: {case_name}"
        assert not all_none, f"None gradient detected in stress case: {case_name}"
        
    print("  -> PASSED: Model & Loss survive extreme degenerate inputs without numerical breakdown.")
    
    # -------------------------------------------------------------------------
    # TEST 4: Batch Scaling Verification (B = 1, 2, 4, 8, 16)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Batch Scaling & Execution Performance (B = 1, 2, 4, 8, 16)")
    batch_sizes = [1, 2, 4, 8, 16]
    scaling_results = []
    
    print(f"  {'Batch (B)':<10} {'Phase Shape':<20} {'Rec Shape':<20} {'Max Normalization':<20} {'Step Time (ms)':<16} {'Throughput (samp/s)'}")
    print("  " + "-" * 105)
    
    for B in batch_sizes:
        x = torch.rand(B, 4, 128, 128)
        
        # Measure step time
        start_time = time.perf_counter()
        
        model.zero_grad()
        phase = model(x)
        rec = propagator(phase, normalize=True)
        loss, l1, ssim_val, pcc_val = criterion(rec, x)
        loss.backward()
        
        step_time_ms = (time.perf_counter() - start_time) * 1000.0
        throughput = B / (step_time_ms / 1000.0)
        
        # Verify shapes
        assert phase.shape == (B, 1, 128, 128), f"Phase shape mismatch for B={B}: {phase.shape}"
        assert rec.shape == (B, 4, 128, 128), f"Rec shape mismatch for B={B}: {rec.shape}"
        
        # Verify amplitude normalization: max must be 1.0 for each plane of each sample
        max_vals = rec.amax(dim=(-2, -1))
        is_normalized = torch.allclose(max_vals, torch.ones_like(max_vals), atol=1e-5)
        assert is_normalized, f"Amplitude normalization failed for B={B}!"
        
        # Verify gradients for this batch size
        grads_ok = all(
            p.grad is not None and torch.isfinite(p.grad).all() and not (p.grad == 0).all()
            for p in model.parameters()
        )
        assert grads_ok, f"Gradient health check failed for batch size B={B}"
        
        norm_str = "All == 1.000000" if is_normalized else "FAILED"
        print(f"  B={B:<8} {str(list(phase.shape)):<20} {str(list(rec.shape)):<20} {norm_str:<20} {step_time_ms:<16.2f} {throughput:<.2f}")
        scaling_results.append({
            'B': B,
            'time_ms': step_time_ms,
            'throughput': throughput
        })
        
    print("  -> PASSED: Batch scaling verified across B=1, 2, 4, 8, 16.")
    
    # -------------------------------------------------------------------------
    # TEST 5: Mode Transitions (Train vs Eval) and Batch Size 1 Edge Case
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Mode Transitions (Train vs Eval) & Single-Sample Invariance")
    model.eval()
    with torch.no_grad():
        x_single = torch.rand(1, 4, 128, 128)
        phase_eval = model(x_single)
        rec_eval = propagator(phase_eval, normalize=True)
        assert phase_eval.shape == (1, 1, 128, 128)
        assert rec_eval.shape == (1, 4, 128, 128)
        assert phase_eval.min() >= -torch.pi - 1e-6 and phase_eval.max() <= torch.pi + 1e-6
        assert torch.allclose(rec_eval.amax(dim=(-2, -1)), torch.ones(1, 4), atol=1e-5)
    print("  -> PASSED: Evaluation mode with B=1 executes cleanly, maintaining bounds.")

    print("\n" + "=" * 80)
    print("ALL EMPIRICAL CHALLENGER TESTS COMPLETED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
