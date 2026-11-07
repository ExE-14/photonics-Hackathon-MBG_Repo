"""Comprehensive Unit & Regression Tests for CLI Inference (src/infer.py).

Verifies:
- Argument parsing and input resolution
- Preprocessing of arbitrary dimensions and color formats
- Output image specifications (128x128 8-bit grayscale PNG)
- Fresnel propagation and evaluation metrics calculation
- CLI execution end-to-end for both --input_dir and --inputs modes
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infer import (
    GRID_SIZE,
    load_all_input_planes,
    load_and_preprocess_image,
    natural_sort_key,
    resolve_device,
    resolve_input_paths,
    resolve_weights_path,
)

PYTHON_EXE = sys.executable
SAMPLE_GROUP_DIR = Path(r"E:\Фотоника\data\processed\test_dataset\group_0")
if not SAMPLE_GROUP_DIR.exists():
    SAMPLE_GROUP_DIR = Path(r"E:\Фотоника\data\processed\test_dataset\group_000")


class TestInferCLI(unittest.TestCase):

    def setUp(self):
        self.temp_dir = CODE_DIR / "outputs" / "unit_test_tmp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.temp_dir.exists():
            for f in self.temp_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                self.temp_dir.rmdir()
            except Exception:
                pass

    def test_natural_sort_key(self):
        paths = [
            Path("plane_10.png"),
            Path("plane_1.png"),
            Path("plane_2.png"),
            Path("plane_20.png"),
        ]
        sorted_paths = sorted(paths, key=natural_sort_key)
        self.assertEqual(
            [p.name for p in sorted_paths],
            ["plane_1.png", "plane_2.png", "plane_10.png", "plane_20.png"],
        )

    def test_resolve_input_paths_from_dir(self):
        paths = resolve_input_paths(str(SAMPLE_GROUP_DIR), None)
        self.assertEqual(len(paths), 4)
        for p in paths:
            self.assertTrue(p.is_file())

    def test_resolve_input_paths_explicit(self):
        dir_paths = resolve_input_paths(str(SAMPLE_GROUP_DIR), None)
        explicit_paths = [str(p) for p in dir_paths]
        resolved = resolve_input_paths(None, explicit_paths)
        self.assertEqual(resolved, dir_paths)

    def test_resolve_input_paths_invalid_count(self):
        # Create dir with 3 images
        sub = self.temp_dir / "three_images"
        sub.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            Image.new("L", (128, 128), color=128).save(sub / f"img_{i}.png")
        with self.assertRaises(ValueError):
            resolve_input_paths(str(sub), None)

    def test_resolve_input_paths_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            resolve_input_paths(None, ["nonexistent_1.png", "nonexistent_2.png", "3.png", "4.png"])

    def test_load_and_preprocess_image_formats(self):
        # Test RGB 200x300
        rgb_path = self.temp_dir / "test_rgb.png"
        Image.new("RGB", (200, 300), color=(100, 150, 200)).save(rgb_path)
        arr = load_and_preprocess_image(rgb_path, target_size=128)
        self.assertEqual(arr.shape, (128, 128))
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(0.0 <= arr.min() and arr.max() <= 1.0)

        # Test RGBA 64x64
        rgba_path = self.temp_dir / "test_rgba.png"
        Image.new("RGBA", (64, 64), color=(50, 60, 70, 255)).save(rgba_path)
        arr2 = load_and_preprocess_image(rgba_path, target_size=128)
        self.assertEqual(arr2.shape, (128, 128))

    def test_resolve_device(self):
        cpu_dev = resolve_device("cpu")
        self.assertEqual(cpu_dev.type, "cpu")
        auto_dev = resolve_device("auto")
        self.assertIn(auto_dev.type, ["cuda", "cpu"])

    def test_resolve_weights_path(self):
        weights = resolve_weights_path("doe_network.pth")
        self.assertTrue(weights.is_file())

    def test_cli_end_to_end_dir_and_eval(self):
        out_png = self.temp_dir / "test_out_eval.png"
        cmd = [
            PYTHON_EXE,
            str(SRC_DIR / "infer.py"),
            "--input_dir",
            str(SAMPLE_GROUP_DIR),
            "--output",
            str(out_png),
            "--eval",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertIn("[AI Photonics] Target compute device:", res.stdout)
        self.assertIn("[AI Photonics] Pure model inference time:", res.stdout)
        self.assertIn("Fresnel Reconstruction Evaluation (--eval):", res.stdout)
        self.assertIn("Mean", res.stdout)
        self.assertTrue(out_png.is_file())

        with Image.open(out_png) as img:
            self.assertEqual(img.size, (128, 128))
            self.assertEqual(img.mode, "L")

    def test_cli_end_to_end_inputs_explicit(self):
        out_png = self.temp_dir / "test_out_inputs.png"
        dir_paths = resolve_input_paths(str(SAMPLE_GROUP_DIR), None)
        cmd = [
            PYTHON_EXE,
            str(SRC_DIR / "infer.py"),
            "--inputs",
            str(dir_paths[0]),
            str(dir_paths[1]),
            str(dir_paths[2]),
            str(dir_paths[3]),
            "--output",
            str(out_png),
            "--device",
            "cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.assertIn("[AI Photonics] Target compute device: CPU", res.stdout)
        self.assertIn("[AI Photonics] Pure model inference time:", res.stdout)
        self.assertTrue(out_png.is_file())

        with Image.open(out_png) as img:
            self.assertEqual(img.size, (128, 128))
            self.assertEqual(img.mode, "L")


if __name__ == "__main__":
    unittest.main()
