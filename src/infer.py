#!/usr/bin/env python3
"""Universal CLI Inference Script for MEPhI AI Photonics DOE Generation.

Generates a Diffractive Optical Element (DOE) phase mask from 4 axial plane target images
using a trained U-Net deep learning model (DOENetwork).

CLI Usage:
    # 1. Using an input directory:
    python src/infer.py --input_dir path/to/group_dir --output doe.png

    # 2. Using explicit input plane file paths:
    python src/infer.py --inputs z1.png z2.png z3.png z4.png --output doe.png

    # 3. With Fresnel evaluation and metrics (Pearson CC, SSIM):
    python src/infer.py --input_dir path/to/group_dir --eval

    # 4. Specifying device and custom weights:
    python src/infer.py --input_dir path/to/group_dir --device cuda --weights doe_network.pth
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Setup module search paths
SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

try:
    from doe_network import (
        GRID_SIZE,
        PIXEL_SIZE,
        WAVELENGTH,
        Z_DISTANCES,
        DOENetwork,
        SpectralResAttentionUNet,
        FresnelPropagation,
        calculate_metrics,
    )
except ImportError:
    from src.doe_network import (
        GRID_SIZE,
        PIXEL_SIZE,
        WAVELENGTH,
        Z_DISTANCES,
        DOENetwork,
        SpectralResAttentionUNet,
        FresnelPropagation,
        calculate_metrics,
    )

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path: Path) -> list:
    """Sort filenames with embedded numbers in natural human order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def resolve_input_paths(input_dir: Optional[str], inputs: Optional[List[str]]) -> List[Path]:
    """Resolves and validates exactly 4 input image paths from either --inputs or --input_dir."""
    if inputs:
        paths = [Path(p) for p in inputs]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(f"Specified input image does not exist: {p}")
        return paths

    if input_dir is not None:
        dir_path = Path(input_dir)
        if not dir_path.exists():
            # Support group_000 -> group_0 alias fallback if needed
            normalized_name = re.sub(r"group_0+", "group_", dir_path.name)
            fallback = dir_path.parent / normalized_name
            if fallback.exists():
                dir_path = fallback
            else:
                raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        files = [
            p for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if len(files) != 4:
            raise ValueError(
                f"Expected exactly 4 image files in directory '{input_dir}', but found {len(files)}. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return sorted(files, key=natural_sort_key)

    raise ValueError("Either --input_dir or --inputs must be specified.")


def load_and_preprocess_image(path: Path, target_size: int = GRID_SIZE) -> np.ndarray:
    """Loads an arbitrary image, converts to 8-bit grayscale, resizes, and normalizes to [0.0, 1.0]."""
    try:
        with Image.open(path) as img:
            img_gray = img.convert("L")
            if img_gray.size != (target_size, target_size):
                img_gray = img_gray.resize((target_size, target_size), Image.Resampling.BILINEAR)
            arr = np.array(img_gray, dtype=np.float32) / 255.0
            return arr
    except Exception as e:
        raise RuntimeError(f"Failed to load and preprocess image at '{path}': {e}") from e


def load_all_input_planes(paths: List[Path], target_size: int = GRID_SIZE) -> torch.Tensor:
    """Loads all 4 input planes and stacks them into a [1, 4, target_size, target_size] float tensor."""
    planes_np = [load_and_preprocess_image(p, target_size=target_size) for p in paths]
    stacked = np.stack(planes_np, axis=0)  # [4, H, W]
    tensor = torch.from_numpy(stacked).unsqueeze(0)  # [1, 4, H, W]
    return tensor


def resolve_device(device_arg: str) -> torch.device:
    """Resolves target compute device: auto, cuda, or cpu."""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested via --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    elif device_arg == "cpu":
        return torch.device("cpu")
    elif device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_arg)


def resolve_weights_path(weights_arg: str) -> Path:
    """Locates model weights file from argument or common project locations."""
    p = Path(weights_arg)
    if p.is_file():
        return p
    candidate = CODE_DIR / weights_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Model weights checkpoint not found: {weights_arg}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal CLI inference for MEPhI AI Photonics DOE generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Path to directory containing 4 axial plane images (auto-sorted).",
    )
    input_group.add_argument(
        "--inputs",
        nargs=4,
        type=str,
        metavar=("Z1", "Z2", "Z3", "Z4"),
        help="Explicit paths to 4 image files corresponding to planes z1, z2, z3, z4.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="doe.png",
        help="Path to save the generated DOE phase mask PNG.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="doe_network.pth",
        help="Path to trained model weights checkpoint file.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Perform Fresnel propagation and compute Pearson CC and SSIM reconstruction metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Target device for inference: auto, cuda, or cpu.",
    )

    args = parser.parse_args()

    # 1. Resolve Device
    device = resolve_device(args.device)
    print(f"[AI Photonics] Target compute device: {device.type.upper()}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # 2. Resolve & Load Inputs
    input_paths = resolve_input_paths(args.input_dir, args.inputs)
    print("[AI Photonics] Input planes:")
    for idx, (path, z) in enumerate(zip(input_paths, Z_DISTANCES), start=1):
        print(f"  Plane {idx} (z = {z * 1000:.1f} mm): {path}")

    inputs_tensor = load_all_input_planes(input_paths, target_size=GRID_SIZE).to(device)

    # 3. Load Model & Weights
    weights_path = resolve_weights_path(args.weights)
    print(f"[AI Photonics] Loading model weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    is_spectral = any("bottleneck_fno" in k or "spectral_dec" in k for k in state_dict.keys())
    if is_spectral:
        print("[AI Photonics] Detected Spectral FNO weights; initializing SpectralResAttentionUNet.")
        model = SpectralResAttentionUNet(in_channels=4, base_channels=32).to(device)
    else:
        model = DOENetwork(in_channels=4, base_channels=32).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 4. Pure Inference Timing
    with torch.no_grad():
        if device.type == "cuda":
            # Warm-up pass to initialize CUDA runtime context
            _ = model(inputs_tensor)
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        phase_tensor = model(inputs_tensor)  # [1, 1, 128, 128], range [-pi, +pi]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

    pure_inference_ms = (t_end - t_start) * 1000.0
    print(f"[AI Photonics] Pure model inference time: {pure_inference_ms:.2f} ms")

    # 5. Output Phase Mapping & Saving
    # Phase in [-pi, +pi] -> uint8 [0, 255]: 0 -> -pi, 255 -> +pi
    phase_np = phase_tensor.squeeze().detach().cpu().numpy()
    phase_uint8 = np.clip(
        np.round((phase_np + np.pi) / (2.0 * np.pi) * 255.0),
        0,
        255,
    ).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = Image.fromarray(phase_uint8, mode="L")
    out_img.save(output_path)
    print(f"[AI Photonics] DOE phase mask saved to: {output_path} (128x128 8-bit grayscale PNG)")

    # 6. Evaluation Mode (--eval)
    if args.eval:
        print("\n[AI Photonics] Running Fresnel diffraction propagation evaluation...")
        propagator = FresnelPropagation(
            z_list=Z_DISTANCES,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            grid_size=GRID_SIZE,
        ).to(device)

        with torch.no_grad():
            reconstructed_amp = propagator(phase_tensor, normalize=True)  # [1, 4, 128, 128]

        # Target inputs normalized to maximum 1.0 (matching benchmark protocol)
        targets_norm = inputs_tensor / torch.clamp_min(
            inputs_tensor.amax(dim=(-2, -1), keepdim=True), 1e-12
        )

        plane_metrics = []
        for j, z in enumerate(Z_DISTANCES):
            pcc, ssim = calculate_metrics(
                targets_norm[:, j : j + 1],
                reconstructed_amp[:, j : j + 1],
            )
            plane_metrics.append((j + 1, z * 1000.0, pcc, ssim))

        mean_pcc = sum(m[2] for m in plane_metrics) / len(plane_metrics)
        mean_ssim = sum(m[3] for m in plane_metrics) / len(plane_metrics)

        print("\nFresnel Reconstruction Evaluation (--eval):")
        print("+-------+---------------+------------+---------+")
        print("| Plane | Distance (mm) | Pearson CC |   SSIM  |")
        print("+-------+---------------+------------+---------+")
        for plane_idx, dist_mm, pcc, ssim in plane_metrics:
            print(f"|   {plane_idx}   |     {dist_mm:4.1f}      |   {pcc:.4f}   |  {ssim:.4f} |")
        print("+-------+---------------+------------+---------+")
        print(f"| Mean  |       -       |   {mean_pcc:.4f}   |  {mean_ssim:.4f} |")
        print("+-------+---------------+------------+---------+")


if __name__ == "__main__":
    main()
