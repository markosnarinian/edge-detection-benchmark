#!/usr/bin/env python3
"""
CPU-only detector benchmark across every core RF-DETR detection variant and
every official YOLOX variant, using every relevant Raspberry Pi CPU runtime:
PyTorch, ONNX Runtime, OpenVINO, ncnn, TFLite, and ExecuTorch/XNNPACK.

Target hardware: Raspberry Pi 4 or Pi 5 (CPU-only, no accelerator).
See README.md for the two-stage setup, dependency caveats, and Pi guidance.

This script is meant to run unattended on a Raspberry Pi with nobody available
to debug it interactively. Every external operation (uv install, git clone,
checkpoint download, ONNX export, inference) is wrapped so a single failure is
logged clearly and skipped rather than crashing the whole run. Partial results
(e.g. only YOLOX succeeded, or only some resolutions) are still written out.

Resolutions are swept via --img-sizes (comma-separated, default covers 14
practically useful sizes from 256 through 1024). Both model families
require the resolution to be a multiple of 32 — YOLOX's FPN has strides up to
32, and RF-DETR Nano's ViT backbone requires resolution divisible by
patch_size(16) * num_windows(2) = 32 — so any entry that isn't a multiple of
32 is dropped with a warning rather than failing the whole run.

Typical usage on the Pi:

    python3 benchmark.py --images ./my_test_images

    # or, using the built-in COCO-sample / synthetic-image fallback:
    python3 benchmark.py

    # sweep a custom set of resolutions instead of the default range:
    python3 benchmark.py --img-sizes 320,640,960

    # export on a beefier dev machine, copy artifacts to the Pi, then run only
    # the timing phase there (runtime packages are still required):
    python3 benchmark.py --export-only --work-dir ./work
    # ... copy ./results/artifacts to the Pi ...
    python3 benchmark.py --skip-export --artifact-dir ./artifacts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

YOLOX_REPO_URL = "https://github.com/Megvii-BaseDetection/YOLOX.git"
YOLOX_CKPT_URLS = {
    "nano": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
    "tiny": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
    "s": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "m": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
    "l": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
    "x": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
    "darknet53": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_darknet.pth",
}
YOLOX_EXP_FILE = {
    "nano": "exps/default/yolox_nano.py",
    "tiny": "exps/default/yolox_tiny.py",
    "s": "exps/default/yolox_s.py",
    "m": "exps/default/yolox_m.py",
    "l": "exps/default/yolox_l.py",
    "x": "exps/default/yolox_x.py",
    "darknet53": "exps/default/yolov3.py",
}

YOLOX_VARIANTS = tuple(YOLOX_EXP_FILE)
RFDETR_VARIANTS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
}
MODEL_RUNTIMES = {
    "yolox": ("pytorch", "onnxruntime", "openvino", "ncnn"),
    "rfdetr": ("pytorch", "onnxruntime", "openvino", "tflite", "executorch"),
}
CPU_RUNTIMES = tuple(dict.fromkeys(runtime for values in MODEL_RUNTIMES.values() for runtime in values))

# A handful of low-numbered, widely-mirrored COCO val2017 images. Used only
# as a convenience fallback when the user doesn't pass --images. If any of
# these URLs are unreachable (no internet on the Pi, server hiccup, image
# renamed/removed upstream), the script logs it and falls back further to
# synthetically generated test images so the benchmark can still run.
COCO_SAMPLE_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000000139.jpg",
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000000632.jpg",
    "http://images.cocodataset.org/val2017/000000000724.jpg",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

log = logging.getLogger("bench")


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def run_cmd(
    cmd: list[str], cwd: Optional[Path] = None, timeout: int = 1800
) -> tuple[bool, str]:
    """Run a subprocess, capturing combined output. Never raises."""
    log.info("$ %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        return False, f"command not found: {e}"
    except subprocess.TimeoutExpired:
        return False, f"command timed out after {timeout}s: {' '.join(cmd)}"
    except Exception as e:  # defensive: never let a subprocess call crash the run
        return False, f"unexpected error running command: {e}"

    ok = proc.returncode == 0
    if not ok:
        # Keep the tail of the log; full install/build output can be huge.
        tail = "\n".join(proc.stdout.splitlines()[-60:]) if proc.stdout else ""
        log.error(
            "command failed (exit %s): %s\n--- last output lines ---\n%s",
            proc.returncode,
            " ".join(cmd),
            tail,
        )
    return ok, proc.stdout or ""


def uv_install(args: list[str], timeout: int = 1800) -> bool:
    ok, _ = run_cmd(
        ["uv", "pip", "install", "--python", sys.executable] + args,
        timeout=timeout,
    )
    return ok


def download_file(url: str, dest: Path, timeout: int = 60, retries: int = 2) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    for attempt in range(1, retries + 2):
        try:
            log.info("Downloading %s -> %s (attempt %d)", url, dest, attempt)
            req = urllib.request.Request(url, headers={"User-Agent": "edge-bench/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(
                partial, "wb"
            ) as f:
                shutil.copyfileobj(resp, f)
                f.flush()
                os.fsync(f.fileno())
            if partial.stat().st_size == 0:
                raise IOError("downloaded file is empty")
            partial.replace(dest)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, IOError, OSError) as e:
            log.warning("Download failed (%s): %s", url, e)
            partial.unlink(missing_ok=True)
            time.sleep(1.5 * attempt)
    log.error("Giving up on download after %d attempts: %s", retries + 1, url)
    return False


def human_mb(num_bytes: float) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def package_version(*distributions: str) -> str:
    """Return the first installed distribution version without importing it."""
    for distribution in distributions:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unavailable"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_components(path: Path, kind: str) -> list[Path]:
    return [path, path.with_suffix(".bin")] if kind == "ncnn" else [path]


def exporter_versions(family: str, kind: str) -> dict[str, str]:
    distributions = {
        ("yolox", "onnx"): ("torch", "onnx", "onnxsim"),
        ("yolox", "ncnn"): ("torch", "pnnx", "ncnn"),
        ("rfdetr", "onnx"): ("rfdetr", "torch", "onnx"),
        ("rfdetr", "tflite"): ("rfdetr", "torch", "ai-edge-litert", "tensorflow"),
        ("rfdetr", "executorch"): ("rfdetr", "torch", "executorch"),
    }[(family, kind)]
    return {name: package_version(name) for name in distributions}


def runtime_versions(runtimes: list[str]) -> dict[str, str]:
    distributions = {
        "pytorch": ("torch",),
        "onnxruntime": ("onnxruntime",),
        "openvino": ("openvino",),
        "ncnn": ("ncnn",),
        "tflite": ("ai-edge-litert", "tflite-runtime", "tensorflow"),
        "executorch": ("executorch",),
    }
    return {runtime: package_version(*distributions[runtime]) for runtime in runtimes}


def build_artifact_manifest(path: Path, kind: str, config: dict, versions: dict[str, str]) -> dict:
    components = {
        component.name: {"sha256": sha256_file(component), "size": component.stat().st_size}
        for component in artifact_components(path, kind)
    }
    identity_source = json.dumps(
        {"config": config, "exporter_versions": versions, "components": components},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema": 2,
        "config": config,
        "exporter_versions": versions,
        "components": components,
        "artifact_id": hashlib.sha256(identity_source).hexdigest(),
    }


def validate_artifact_manifest(
    path: Path,
    kind: str,
    manifest_path: Path,
    config: dict,
    expected_versions: Optional[dict[str, str]] = None,
) -> tuple[bool, Optional[dict], str]:
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != 2 or manifest.get("config") != config:
            return False, None, "manifest configuration does not match"
        if expected_versions is not None and manifest.get("exporter_versions") != expected_versions:
            return False, None, "exporter versions do not match"
        recorded = manifest.get("components")
        components = artifact_components(path, kind)
        if not isinstance(recorded, dict) or set(recorded) != {item.name for item in components}:
            return False, None, "manifest component list does not match"
        for component in components:
            details = recorded[component.name]
            if not isinstance(details, dict):
                return False, None, f"invalid manifest entry for {component.name}"
            if not component.is_file() or component.stat().st_size != details.get("size"):
                return False, None, f"missing or size-mismatched component {component.name}"
            if sha256_file(component) != details.get("sha256"):
                return False, None, f"hash mismatch for {component.name}"
        identity_source = json.dumps(
            {
                "config": manifest["config"],
                "exporter_versions": manifest["exporter_versions"],
                "components": recorded,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if hashlib.sha256(identity_source).hexdigest() != manifest.get("artifact_id"):
            return False, None, "artifact identity does not match"
        return True, manifest, ""
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
        return False, None, str(e)


DEFAULT_IMG_SIZES = "256,320,384,416,448,512,576,640,704,768,832,896,960,1024"


def parse_img_sizes(raw: str) -> list[int]:
    """Parse a comma-separated --img-sizes value into a sorted, deduplicated
    list of valid resolutions. Both model families require the resolution to be a
    multiple of 32 (YOLOX's FPN strides; RF-DETR Nano's patch_size(16) *
    num_windows(2)); invalid entries are dropped with a warning rather than
    aborting the whole sweep."""
    sizes: set[int] = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            log.warning("Ignoring non-integer --img-sizes entry: %r", tok)
            continue
        if v <= 0:
            log.warning("Ignoring non-positive --img-sizes entry: %d", v)
            continue
        if v % 32 != 0:
            nearest = max(32, round(v / 32) * 32)
            log.warning(
                "Ignoring --img-sizes entry %d: not a multiple of 32, which both "
                "YOLOX (FPN strides) and RF-DETR Nano (patch_size*num_windows) "
                "require. Nearest valid value: %d.",
                v,
                nearest,
            )
            continue
        sizes.add(v)
    return sorted(sizes)


# --------------------------------------------------------------------------
# Test images
# --------------------------------------------------------------------------


def make_synthetic_image(path: Path, size: int = 720, seed: int = 0) -> None:
    """Last-resort fallback so the benchmark can always run: a deterministic
    pseudo-random RGB image. Not representative of real detection accuracy,
    but a perfectly valid input tensor for a *latency* benchmark."""
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype="uint8")
    Image.fromarray(arr, mode="RGB").save(path, quality=90)


def get_test_images(images_dir: Optional[Path], num_fallback: int, work_dir: Path) -> list[Path]:
    if images_dir is not None:
        if not images_dir.is_dir():
            log.error(
                "--images %s is not a directory; falling back to bundled/synthetic images",
                images_dir,
            )
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            found = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
            if found:
                log.info("Using %d local test image(s) from %s", len(found), images_dir)
                return found
            log.warning(
                "--images %s contained no images with extensions %s; falling back",
                images_dir,
                sorted(exts),
            )

    cache_dir = work_dir / "sample_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for url in COCO_SAMPLE_URLS[:num_fallback]:
        dest = cache_dir / Path(url).name
        if dest.exists() and dest.stat().st_size > 0:
            downloaded.append(dest)
            continue
        if download_file(url, dest):
            downloaded.append(dest)

    if downloaded:
        log.info("Using %d downloaded COCO val2017 sample image(s)", len(downloaded))
        return downloaded

    log.warning(
        "No local images provided and COCO sample downloads failed (likely no "
        "internet on this device). Generating %d synthetic test image(s) instead. "
        "Latency numbers are still valid; detection *output* is meaningless.",
        num_fallback,
    )
    synth: list[Path] = []
    for i in range(num_fallback):
        p = cache_dir / f"synthetic_{i}.jpg"
        try:
            make_synthetic_image(p, seed=i)
            synth.append(p)
        except Exception as e:
            log.error("Failed to generate synthetic image %s: %s", p, e)
    if not synth:
        raise RuntimeError(
            "Could not obtain any test images (no --images, downloads failed, "
            "and synthetic image generation failed). Cannot run the benchmark."
        )
    return synth


# --------------------------------------------------------------------------
# Preprocessing
#
# These approximate each model's official preprocessing closely enough to
# exercise realistic tensor shapes and memory-access patterns. Exact
# numerical/accuracy fidelity is NOT required for a latency/throughput
# benchmark; if you need to sanity-check detections themselves, use each
# project's own inference script instead.
# --------------------------------------------------------------------------


def preprocess_yolox(image_path: Path, size: int):
    """Letterbox resize, BGR, no mean/std normalization (matches YOLOX's own
    preproc, which trains on raw 0-255 pixel values)."""
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.BILINEAR)

    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(resized, (0, 0))

    arr = np.asarray(canvas, dtype="float32")  # HWC, RGB, 0-255
    arr = arr[:, :, ::-1]  # -> BGR to match YOLOX training convention
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.ascontiguousarray(arr[None, ...])  # add batch dim
    return arr


def preprocess_rfdetr(image_path: Path, size: int):
    """Resize (no letterbox padding), RGB, ImageNet mean/std normalization,
    matching the general DINOv2/ViT-backbone preprocessing convention."""
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype="float32") / 255.0  # HWC, RGB, 0-1
    mean = np.array(IMAGENET_MEAN, dtype="float32")
    std = np.array(IMAGENET_STD, dtype="float32")
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.ascontiguousarray(arr[None, ...])
    return arr


# --------------------------------------------------------------------------
# Export: YOLOX
# --------------------------------------------------------------------------


def setup_yolox_repo(work_dir: Path) -> Optional[Path]:
    repo_dir = work_dir / "YOLOX"
    if (repo_dir / "yolox").is_dir():
        log.info("YOLOX repo already present at %s", repo_dir)
    else:
        ok, _ = run_cmd(["git", "clone", "--depth", "1", YOLOX_REPO_URL, str(repo_dir)])
        if not ok:
            log.error(
                "Failed to clone YOLOX repo. Check internet connectivity and that "
                "`git` is installed (`sudo apt install git`)."
            )
            return None

    log.info(
        "Installing YOLOX + its Python dependencies (torch, torchvision, "
        "pycocotools, ...). This is the slowest step of the whole script on a "
        "Pi 4 — see README.md for timing guidance and how to avoid it via "
        "--export-only on a faster machine."
    )
    if not uv_install(["-q", "-r", str(repo_dir / "requirements.txt")], timeout=3600):
        log.error(
            "Failed to install YOLOX requirements.txt. This most commonly means "
            "torch failed to install. On a Pi, prefer the official PyTorch CPU "
            "wheels: uv pip install torch --default-index https://download.pytorch.org/whl/cpu "
            "(see README.md)."
        )
        return None
    if not uv_install(["-q", "-e", str(repo_dir)], timeout=1200):
        log.error("Failed to install the YOLOX repo itself.")
        return None
    return repo_dir


def load_yolox_model(
    repo_dir: Path, variant: str, img_size: int, ckpt_path: Path, num_classes: int
):
    sys.path.insert(0, str(repo_dir))
    from yolox.exp import get_exp  # type: ignore
    import torch  # type: ignore

    exp = get_exp(str(repo_dir / YOLOX_EXP_FILE[variant]), None)
    exp.test_size = (img_size, img_size)
    exp.num_classes = num_classes
    model = exp.get_model()
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    current = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    skipped = sorted(set(state) - set(compatible))
    if skipped:
        log.warning(
            "num_classes=%d left %d YOLOX head tensor(s) randomly initialized: %s",
            num_classes,
            len(skipped),
            skipped,
        )
    model.load_state_dict(compatible, strict=False)
    model.eval()
    model.head.decode_in_inference = False
    return model


def replace_yolox_focus_for_export(model) -> int:
    """Replace pnnx-incompatible Focus slicing while preserving exact outputs."""
    import torch  # type: ignore
    from yolox.models.network_blocks import Focus  # type: ignore

    class ExportFocus(torch.nn.Module):
        def __init__(self, focus):
            super().__init__()
            self.unshuffle = torch.nn.PixelUnshuffle(2)
            self.conv = focus.conv
            weight = self.conv.conv.weight
            channels = weight.shape[1] // 4
            indices = torch.arange(channels, device=weight.device)
            # Focus emits TL/BL/TR/BR channel blocks. PixelUnshuffle emits
            # TL/TR/BL/BR per source channel, so adapt the consumer weights.
            permutation = torch.stack(
                (indices, indices + 2 * channels, indices + channels, indices + 3 * channels),
                dim=1,
            ).reshape(-1)
            with torch.no_grad():
                self.conv.conv.weight.copy_(weight[:, permutation].contiguous())

        def forward(self, value):
            return self.conv(self.unshuffle(value))

    replacements = 0
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, Focus):
            continue
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, ExportFocus(module))
        replacements += 1
    return replacements


def flatten_torch_outputs(value):
    import torch  # type: ignore

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    if isinstance(value, dict):
        values = [flatten_torch_outputs(value[key]) for key in sorted(value)]
    elif isinstance(value, (tuple, list)):
        values = [flatten_torch_outputs(item) for item in value]
    else:
        raise TypeError(f"unsupported model output type: {type(value).__name__}")
    import numpy as np

    return np.concatenate(values)


def validate_ncnn_export(param_path: Path, bin_path: Path, input_array, expected_output) -> None:
    """Load with stock ncnn and reject a converted graph whose raw output drifts."""
    import ncnn  # type: ignore
    import numpy as np

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    if net.load_param(str(param_path)) != 0 or net.load_model(str(bin_path)) != 0:
        raise RuntimeError("stock ncnn failed to load the converted param/bin pair")
    input_names = list(net.input_names())
    output_names = list(net.output_names())
    if len(input_names) != 1 or not output_names:
        raise RuntimeError(f"unexpected ncnn I/O names: {input_names}/{output_names}")
    input_mat = ncnn.Mat(input_array[0]).clone()
    extractor = net.create_extractor()
    code = extractor.input(input_names[0], input_mat)
    if code != 0:
        raise RuntimeError(f"ncnn input binding failed: {code}")
    actual_parts = []
    for output_name in output_names:
        code, output = extractor.extract(output_name)
        if code != 0:
            raise RuntimeError(f"ncnn extraction failed for {output_name}: {code}")
        actual_parts.append(np.asarray(output).reshape(-1))
    actual = np.concatenate(actual_parts)
    expected = flatten_torch_outputs(expected_output)
    if actual.shape != expected.shape:
        raise RuntimeError(f"ncnn parity shape mismatch: {actual.shape} != {expected.shape}")
    if not np.allclose(actual, expected, rtol=2e-3, atol=2e-3):
        max_error = float(np.max(np.abs(actual - expected)))
        raise RuntimeError(f"ncnn raw-output parity check failed (max abs error {max_error:.6g})")


def export_yolox_onnx(
    repo_dir: Path, variant: str, img_size: int, ckpt_path: Path, out_path: Path, opset: int, num_classes: int
) -> bool:
    try:
        import torch  # type: ignore
    except Exception as e:
        log.error("Failed to import YOLOX/torch for export: %s", e)
        return False

    try:
        model = load_yolox_model(repo_dir, variant, img_size, ckpt_path, num_classes)

        dummy = torch.randn(1, 3, img_size, img_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = out_path.with_name(out_path.stem + ".partial.onnx")
        partial_path.unlink(missing_ok=True)
        torch.onnx.export(
            model,
            dummy,
            str(partial_path),
            input_names=["images"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
        )

        try:
            import onnxsim  # type: ignore
            import onnx  # type: ignore

            onnx_model = onnx.load(str(partial_path))
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(partial_path))
                log.info("onnx-simplifier applied to %s", out_path.name)
            else:
                log.warning("onnx-simplifier reported failure; keeping unsimplified export")
        except ImportError:
            log.info("onnx-simplifier not installed; skipping simplification (optional)")
        except Exception as e:
            log.warning("onnx-simplifier step failed non-fatally: %s", e)

        if partial_path.stat().st_size <= 0:
            raise RuntimeError("export produced an empty ONNX file")
        partial_path.replace(out_path)
        return True
    except Exception as e:
        log.error("YOLOX ONNX export failed for variant=%s: %s", variant, e)
        log.debug(traceback.format_exc())
        return False


def export_yolox_ncnn(
    repo_dir: Path,
    variant: str,
    img_size: int,
    ckpt_path: Path,
    out_path: Path,
    num_classes: int,
) -> bool:
    """Export a static FP32 ncnn graph through Tencent's current pnnx path."""
    partial_dir = out_path.parent / ".ncnn.partial"
    shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pnnx  # type: ignore
        import torch  # type: ignore

        model = load_yolox_model(repo_dir, variant, img_size, ckpt_path, num_classes)
        dummy = torch.linspace(-1.0, 1.0, steps=3 * img_size * img_size).reshape(
            1, 3, img_size, img_size
        )
        with torch.inference_mode():
            original_output = model(dummy)
        replacements = replace_yolox_focus_for_export(model)
        with torch.inference_mode():
            transformed_output = model(dummy)
        if not torch.allclose(
            torch.from_numpy(flatten_torch_outputs(original_output)),
            torch.from_numpy(flatten_torch_outputs(transformed_output)),
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError("PixelUnshuffle Focus replacement changed PyTorch output")
        log.info("Replaced %d YOLOX Focus module(s) for pnnx export", replacements)
        torchscript_path = partial_dir / "model.pt"
        pnnx.export(model, str(torchscript_path), (dummy,), fp16=False)
        params = list(partial_dir.glob("*.ncnn.param")) + list(partial_dir.glob("*.param"))
        bins = list(partial_dir.glob("*.ncnn.bin")) + list(partial_dir.glob("*.bin"))
        if not params or not bins:
            raise RuntimeError("pnnx completed without producing ncnn .param/.bin files")
        validate_ncnn_export(params[0], bins[0], dummy.numpy(), transformed_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bin_path = out_path.with_suffix(".bin")
        bins[0].replace(bin_path)
        params[0].replace(out_path)
        shutil.rmtree(partial_dir, ignore_errors=True)
        return out_path.stat().st_size > 0 and bin_path.stat().st_size > 0
    except Exception as e:
        log.error("YOLOX ncnn export failed for variant=%s: %s", variant, e)
        log.debug(traceback.format_exc())
        return False


# --------------------------------------------------------------------------
# Export: RF-DETR
# --------------------------------------------------------------------------


def export_rfdetr_onnx(
    variant: str, img_size: int, out_dir: Path, opset: int, num_classes: int
) -> Optional[Path]:
    try:
        import rfdetr  # type: ignore
    except Exception as e:
        log.error("Failed to import rfdetr after install: %s", e)
        return None

    try:
        log.info(
            "Instantiating RF-DETR %s at resolution=%d, num_classes=%d — this "
            "auto-downloads the official pretrained COCO checkpoint on first run "
            "and re-initializes the detection head if num_classes differs from "
            "the checkpoint's trained class count.",
            variant,
            img_size,
            num_classes,
        )
        model_class = getattr(rfdetr, RFDETR_VARIANTS[variant])
        model = model_class(resolution=img_size, num_classes=num_classes)
    except Exception as e:
        log.error(
            "RF-DETR %s (resolution=%d) failed to construct: %s\n"
            "Core RF-DETR detectors require the resolution to be divisible by "
            "patch_size(16) * num_windows(2) = 32 (this should already be "
            "enforced by --img-sizes validation; if you're seeing this, check "
            "for a version mismatch in the installed rfdetr package).",
            variant,
            img_size,
            e,
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / f".rfdetr_{variant}.partial"
    shutil.rmtree(partial_dir, ignore_errors=True)
    partial_dir.mkdir(parents=True)
    try:
        model.export(
            output_dir=str(partial_dir),
            format="onnx",
            shape=(img_size, img_size),
            batch_size=1,
            opset_version=opset,
            verbose=True,
        )
    except Exception as e:
        log.error(
            "RF-DETR export() failed at resolution=%d: %s\n"
            "If this mentions shape/divisibility, note that --img-sizes entries "
            "not divisible by 32 are already filtered out before export runs.",
            img_size,
            e,
        )
        log.debug(traceback.format_exc())
        return None

    onnx_files = sorted(partial_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if onnx_files:
        destination = out_dir / f"rfdetr_{variant}.onnx"
        onnx_files[0].replace(destination)
        shutil.rmtree(partial_dir, ignore_errors=True)
        return destination

    log.error("RF-DETR export() completed without error but no .onnx file was found in %s", out_dir)
    return None


def export_rfdetr_deployment(
    variant: str,
    img_size: int,
    out_path: Path,
    runtime: str,
    num_classes: int,
) -> Optional[Path]:
    """Export RF-DETR's model-specific CPU deployment formats atomically."""
    try:
        import rfdetr  # type: ignore

        model = getattr(rfdetr, RFDETR_VARIANTS[variant])(
            resolution=img_size, num_classes=num_classes, device="cpu"
        )
        partial_dir = out_path.parent / f".{runtime}.partial"
        shutil.rmtree(partial_dir, ignore_errors=True)
        partial_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {
            "output_dir": str(partial_dir),
            "shape": (img_size, img_size),
            "batch_size": 1,
            "verbose": True,
            "output_name": f"rfdetr-{variant}",
        }
        if runtime == "tflite":
            model.export(format="tflite", quantization="fp32", **kwargs)
            candidates = list(partial_dir.rglob("*_fp32.tflite"))
        elif runtime == "executorch":
            model.export(format="executorch", backend="xnnpack", **kwargs)
            candidates = list(partial_dir.rglob("*.pte"))
        else:
            raise ValueError(f"unsupported RF-DETR deployment runtime: {runtime}")
        if not candidates or candidates[0].stat().st_size <= 0:
            raise RuntimeError(f"{runtime} export produced no non-empty artifact")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        candidates[0].replace(out_path)
        shutil.rmtree(partial_dir, ignore_errors=True)
        return out_path
    except Exception as e:
        log.error("RF-DETR %s export failed for variant=%s: %s", runtime, variant, e)
        log.debug(traceback.format_exc())
        return None


# --------------------------------------------------------------------------
# Benchmarking
# --------------------------------------------------------------------------


@dataclass
class ModelResult:
    name: str
    runtime: str = ""
    img_size: int = 0
    status: str = "not_run"  # not_run | ok | failed
    error: str = ""
    onnx_path: str = ""
    artifact_path: str = ""
    file_size_mb: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    std_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    fps: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    effective_threads: str = ""
    num_timed_runs: int = 0
    per_run_ms: list[float] = field(default_factory=list)


class _RssSampler:
    """Background thread polling this process's RSS so we can capture a peak
    even though individual inferences may be too fast to catch on entry/exit."""

    def __init__(self, interval_s: float = 0.02):
        import psutil  # local import: keep psutil optional until actually used

        self._proc = psutil.Process(os.getpid())
        self._interval = interval_s
        self._stop = threading.Event()
        self._peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            self._stop.wait(self._interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=1.0)
        return self._peak


def benchmark_model(
    name: str,
    family: str,
    variant: str,
    runtime: str,
    artifact_path: Optional[Path],
    preprocess_fn: Callable[[Path, int], "object"],
    image_paths: list[Path],
    img_size: int,
    warmup_runs: int,
    timed_runs: int,
    threads: int,
    num_classes: int,
    yolox_repo: Optional[Path] = None,
    yolox_ckpt: Optional[Path] = None,
) -> ModelResult:
    result = ModelResult(
        name=name,
        runtime=runtime,
        img_size=img_size,
        effective_threads="runtime-default" if runtime == "executorch" else str(threads),
        onnx_path=str(artifact_path) if artifact_path and runtime in ("onnxruntime", "openvino") else "",
        artifact_path=str(artifact_path) if artifact_path else "",
    )

    if runtime != "pytorch":
        if artifact_path is None:
            result.status = "failed"
            result.error = f"export/skip-export did not produce a {runtime} artifact"
            return result
        try:
            artifact_bytes = artifact_path.stat().st_size
            if runtime == "ncnn":
                artifact_bytes += artifact_path.with_suffix(".bin").stat().st_size
            result.file_size_mb = human_mb(artifact_bytes)
        except OSError as e:
            result.status = "failed"
            result.error = f"could not stat runtime artifact: {e}"
            return result

    try:
        if runtime == "onnxruntime":
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(artifact_path), sess_options=so, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name
            infer = lambda tensor: session.run(None, {input_name: tensor})
        elif runtime == "openvino":
            import openvino as ov

            core = ov.Core()
            compiled = core.compile_model(
                str(artifact_path), "CPU", {"INFERENCE_NUM_THREADS": threads}
            )
            input_port = compiled.input(0)
            infer = lambda tensor: compiled({input_port: tensor})
        elif runtime == "tflite":
            try:
                from ai_edge_litert.interpreter import Interpreter  # type: ignore
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter  # type: ignore
                except ImportError:
                    from tensorflow.lite import Interpreter  # type: ignore

            interpreter = Interpreter(model_path=str(artifact_path), num_threads=threads)
            interpreter.allocate_tensors()
            input_detail = interpreter.get_input_details()[0]
            input_shape = tuple(int(value) for value in input_detail["shape"])
            if input_shape != (1, img_size, img_size, 3):
                raise RuntimeError(f"unexpected TFLite input shape: {input_shape}")
            if "float32" not in str(input_detail["dtype"]):
                raise RuntimeError(f"unexpected TFLite input dtype: {input_detail['dtype']}")
            input_index = input_detail["index"]

            def infer(tensor):
                interpreter.set_tensor(input_index, tensor)
                interpreter.invoke()
        elif runtime == "executorch":
            import torch
            from executorch.runtime import Runtime  # type: ignore

            program = Runtime.get().load_program(str(artifact_path))
            forward = program.load_method("forward")

            def infer(tensor):
                return forward.execute([tensor])
        elif runtime == "ncnn":
            import ncnn  # type: ignore

            net = ncnn.Net()
            net.opt.use_vulkan_compute = False
            net.opt.num_threads = threads
            net.opt.use_packing_layout = True
            net.opt.use_fp16_storage = False
            net.opt.use_fp16_arithmetic = False
            if net.load_param(str(artifact_path)) != 0:
                raise RuntimeError(f"ncnn could not load {artifact_path}")
            bin_path = artifact_path.with_suffix(".bin")
            if net.load_model(str(bin_path)) != 0:
                raise RuntimeError(f"ncnn could not load {bin_path}")
            input_names = list(net.input_names())
            output_names = list(net.output_names())
            if len(input_names) != 1 or not output_names:
                raise RuntimeError(
                    f"ncnn expected one input and at least one output, got {input_names}/{output_names}"
                )
            input_name = input_names[0]
            if hasattr(ncnn, "set_omp_dynamic"):
                ncnn.set_omp_dynamic(0)
            if hasattr(ncnn, "set_omp_num_threads"):
                ncnn.set_omp_num_threads(threads)

            def infer(tensor):
                extractor = net.create_extractor()
                code = extractor.input(input_name, tensor)
                if code != 0:
                    raise RuntimeError(f"ncnn input binding failed for {input_name}: {code}")
                for output_name in output_names:
                    code, _ = extractor.extract(output_name)
                    if code != 0:
                        raise RuntimeError(f"ncnn extraction failed for {output_name}: {code}")
        elif runtime == "pytorch":
            import torch

            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

            if family == "yolox":
                if yolox_repo is None or yolox_ckpt is None:
                    raise RuntimeError("YOLOX repository/checkpoint unavailable")
                model = load_yolox_model(yolox_repo, variant, img_size, yolox_ckpt, num_classes)
            else:
                import rfdetr  # type: ignore

                wrapper = getattr(rfdetr, RFDETR_VARIANTS[variant])(
                    resolution=img_size, num_classes=num_classes, device="cpu"
                )
                model = wrapper.model.model
                if model is None:
                    raise RuntimeError("RF-DETR did not construct its PyTorch model")
                model.eval()
                model.export()

            def infer(tensor):
                with torch.inference_mode():
                    return model(torch.from_numpy(tensor))
        else:
            raise ValueError(f"unknown runtime: {runtime}")
    except Exception as e:
        result.status = "failed"
        result.error = f"failed to initialize {runtime}: {e}"
        log.error("Runtime initialization failed for %s/%s: %s", name, runtime, e)
        log.debug(traceback.format_exc())
        return result

    try:
        tensors = []
        for p in image_paths:
            try:
                tensors.append(preprocess_fn(p, img_size))
            except Exception as e:
                log.warning("Skipping image %s: preprocessing failed (%s)", p, e)
        if not tensors:
            raise RuntimeError("no images could be preprocessed")
        if runtime == "tflite":
            tensors = [tensor.transpose(0, 2, 3, 1).copy() for tensor in tensors]
        elif runtime == "executorch":
            import torch

            tensors = [torch.from_numpy(tensor).contiguous() for tensor in tensors]
        elif runtime == "ncnn":
            import ncnn  # type: ignore

            # ncnn.Mat(ndarray) is zero-copy and does not retain the ndarray.
            # clone() gives every Mat owned storage before replacing this list.
            tensors = [ncnn.Mat(tensor[0]).clone() for tensor in tensors]
    except Exception as e:
        result.status = "failed"
        result.error = f"preprocessing failed: {e}"
        return result

    def run_once(i: int):
        tensor = tensors[i % len(tensors)]
        infer(tensor)

    sampler: Optional[_RssSampler] = None
    try:
        import psutil

        log.info("%s/%s: warming up (%d runs)...", result.name, runtime, warmup_runs)
        for i in range(warmup_runs):
            run_once(i)

        proc = psutil.Process(os.getpid())
        proc.cpu_percent(interval=None)  # prime the internal counter
        sampler = _RssSampler().start()

        log.info("%s/%s: timing %d runs (%d thread(s))...", result.name, runtime, timed_runs, threads)
        per_run_ms = []
        wall_start = time.perf_counter()
        for i in range(timed_runs):
            t0 = time.perf_counter()
            run_once(i)
            per_run_ms.append((time.perf_counter() - t0) * 1000.0)
        wall_elapsed = time.perf_counter() - wall_start

        peak_rss = sampler.stop()
        sampler = None
        cpu_pct = proc.cpu_percent(interval=None)

        result.per_run_ms = per_run_ms
        result.num_timed_runs = len(per_run_ms)
        result.avg_latency_ms = round(statistics.mean(per_run_ms), 3)
        result.std_latency_ms = round(statistics.pstdev(per_run_ms), 3) if len(per_run_ms) > 1 else 0.0
        result.min_latency_ms = round(min(per_run_ms), 3)
        result.max_latency_ms = round(max(per_run_ms), 3)
        result.fps = round(len(per_run_ms) / wall_elapsed, 3) if wall_elapsed > 0 else None
        result.peak_rss_mb = human_mb(peak_rss)
        result.cpu_percent = round(cpu_pct, 1)
        result.status = "ok"
    except Exception as e:
        result.status = "failed"
        result.error = f"inference loop failed: {e}"
        log.error("Inference loop failed for %s: %s", result.name, e)
        log.debug(traceback.format_exc())
    finally:
        if sampler is not None:
            sampler.stop()

    return result


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    partial.replace(path)


def write_csv(results: list[ModelResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "runtime", "img_size", "run_index", "latency_ms"])
        for r in results:
            for i, ms in enumerate(r.per_run_ms):
                w.writerow([r.name, r.runtime, r.img_size, i, round(ms, 4)])
        f.flush()
        os.fsync(f.fileno())
    partial.replace(path)
    log.info("Wrote per-run timings to %s", path)


def write_html(results: list[ModelResult], path: Path, meta: dict) -> None:
    def cell(value: object) -> str:
        return html.escape("-" if value is None else str(value))

    rows = []
    for r in sorted(results, key=lambda item: (item.name, item.runtime, item.img_size)):
        values = [
            r.name,
            r.runtime,
            f"{r.img_size}×{r.img_size}",
            r.status,
            r.avg_latency_ms,
            r.std_latency_ms,
            r.min_latency_ms,
            r.max_latency_ms,
            r.fps,
            r.peak_rss_mb,
            r.cpu_percent,
            r.effective_threads,
            r.file_size_mb,
            r.num_timed_runs,
        ]
        klass = "failed" if r.status != "ok" else ""
        rows.append(f'<tr class="{klass}">' + "".join(f"<td>{cell(v)}</td>" for v in values) + "</tr>")

    errors = "".join(
        f"<li><strong>{cell(r.name)} / {cell(r.runtime)} @ {r.img_size}×{r.img_size}</strong>: "
        f"{cell(r.error)}</li>"
        for r in results
        if r.status != "ok"
    )
    metadata = "".join(f"<dt>{cell(k)}</dt><dd>{cell(v)}</dd>" for k, v in meta.items())
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CPU detector benchmark</title>
<style>
body{{font:14px system-ui,sans-serif;margin:2rem;color:#18212b}} h1,h2{{margin-top:1.6em}}
dl{{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1rem}} dt{{font-weight:700}}
.table-wrap{{overflow:auto}} table{{border-collapse:collapse;white-space:nowrap;width:100%}}
th,td{{border:1px solid #ccd5df;padding:.45rem .6rem;text-align:right}} th{{background:#eef3f7;position:sticky;top:0}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}} tr.failed{{background:#fff0f0}}
</style></head><body>
<h1>CPU detector benchmark</h1><h2>Run metadata</h2><dl>{metadata}</dl>
<h2>Results</h2><div class="table-wrap"><table><thead><tr>
<th>Model</th><th>Runtime</th><th>Resolution</th><th>Status</th><th>Avg ms</th><th>Std ms</th>
<th>Min ms</th><th>Max ms</th><th>FPS</th><th>Peak RSS MB</th><th>CPU %</th><th>Threads</th><th>Artifact MB</th><th>Runs</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{f'<h2>Errors</h2><ul>{errors}</ul>' if errors else ''}
<h2>Notes</h2><ul><li>Latency excludes model loading, export, and preprocessing.</li>
<li>CPU percentage is per process and is not normalized by core count.</li>
<li>The Threads column reports <code>runtime-default</code> for ExecuTorch because its Python runtime exposes no stable per-program XNNPACK thread control.</li>
<li>Raw timings are in <code>per_run_timings.csv</code>; machine-readable summaries are in <code>results.json</code>.</li></ul>
</body></html>"""
    atomic_write_text(path, document)
    log.info("Wrote HTML report to %s", path)


def write_checkpoint(results: list[ModelResult], path: Path, signature: dict) -> None:
    atomic_write_text(
        path,
        json.dumps({"signature": signature, "results": [r.__dict__ for r in results]}, indent=2),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--img-sizes",
        type=str,
        default=DEFAULT_IMG_SIZES,
        help=(
            "Comma-separated square input resolutions to sweep, covering the full "
            "practically-supported range (default: %(default)s). Both model families "
            "require multiples of 32; invalid entries are dropped with a warning."
        ),
    )
    p.add_argument("--images", type=Path, default=None, help="Directory of local test images")
    p.add_argument(
        "--num-fallback-images",
        type=int,
        default=5,
        help="How many COCO/synthetic fallback images to use if --images is not given (default: 5)",
    )
    p.add_argument(
        "--yolox-variants",
        default=",".join(YOLOX_VARIANTS),
        help="Comma-separated YOLOX variants (default: all)",
    )
    p.add_argument(
        "--rfdetr-variants",
        default=",".join(RFDETR_VARIANTS),
        help="Comma-separated core RF-DETR detection variants (default: all)",
    )
    p.add_argument(
        "--runtimes",
        default=",".join(CPU_RUNTIMES),
        help=(
            "Comma-separated CPU runtimes: pytorch,onnxruntime,openvino,ncnn,"
            "tflite,executorch (defaults to every model-relevant runtime)"
        ),
    )
    p.add_argument("--warmup-runs", type=int, default=10, help="Warmup inferences before timing (default: 10)")
    p.add_argument(
        "--timed-runs",
        type=int,
        default=30,
        help="Timed inferences per model, minimum 20 enforced (default: 30)",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 4,
        help="CPU threads requested from each runtime (default: all detected cores)",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version for export (default: 17)")
    p.add_argument(
        "--num-classes",
        type=int,
        default=15,
        help=(
            "Detection head class count for both models (default: 15). Differs "
            "from the pretrained COCO checkpoints (80/90 classes), so the "
            "classification head is reinitialized to this shape at export time; "
            "the pretrained backbone/neck weights are kept. This only matters "
            "for a latency benchmark insofar as head size scales with class "
            "count — detection *output* is meaningless either way."
        ),
    )
    p.add_argument("--work-dir", type=Path, default=Path("./work"), help="Scratch dir for repo clones/checkpoints")
    p.add_argument("--output-dir", type=Path, default=Path("./results"), help="Where results are written")
    p.add_argument("--skip-yolox", action="store_true", help="Skip the YOLOX model entirely")
    p.add_argument("--skip-rfdetr", action="store_true", help="Skip the RF-DETR model entirely")
    p.add_argument(
        "--export-only", action="store_true", help="Only run export, then exit (no benchmarking)"
    )
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip export; load existing runtime artifacts from --artifact-dir instead",
    )
    p.add_argument(
        "--artifact-dir",
        "--onnx-dir",
        dest="artifact_dir",
        type=Path,
        default=None,
        help=(
            "Root copied from results/artifacts, containing pre-exported runtime "
            "artifacts; --onnx-dir is retained as a compatibility alias"
        ),
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore a matching checkpoint and rerun completed benchmark cells",
    )
    p.add_argument(
        "--allow-unsigned-artifacts",
        action="store_true",
        help=(
            "With --skip-export, explicitly allow legacy artifacts without a valid "
            "hash manifest. Their content is still hashed for resume identity."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    if args.timed_runs < 20:
        log.warning("--timed-runs %d is below the minimum of 20; raising to 20", args.timed_runs)
        args.timed_runs = 20

    work_dir: Path = args.work_dir.resolve()
    out_dir: Path = args.output_dir.resolve()
    default_artifact_dir = out_dir / "artifacts"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Platform: %s | Python: %s | CPUs: %s", platform.platform(), platform.python_version(), os.cpu_count())

    img_sizes = parse_img_sizes(args.img_sizes)
    if not img_sizes:
        log.error("No valid resolutions remain in --img-sizes %r; nothing to do.", args.img_sizes)
        return 1
    log.info("Resolution sweep: %s", ", ".join(f"{s}x{s}" for s in img_sizes))

    def parse_selection(raw: str, allowed: tuple[str, ...], option: str) -> list[str]:
        selected = list(dict.fromkeys(item.strip().lower() for item in raw.split(",") if item.strip()))
        invalid = [item for item in selected if item not in allowed]
        if invalid:
            raise ValueError(f"{option} has unsupported values {invalid}; choose from {allowed}")
        return selected

    try:
        yolox_variants = [] if args.skip_yolox else parse_selection(
            args.yolox_variants, YOLOX_VARIANTS, "--yolox-variants"
        )
        rfdetr_variants = [] if args.skip_rfdetr else parse_selection(
            args.rfdetr_variants, tuple(RFDETR_VARIANTS), "--rfdetr-variants"
        )
        runtimes = parse_selection(args.runtimes, CPU_RUNTIMES, "--runtimes")
    except ValueError as e:
        log.error("%s", e)
        return 2
    models = [
        (f"yolox_{variant}", "yolox", variant, preprocess_yolox)
        for variant in yolox_variants
    ] + [
        (f"rfdetr_{variant}", "rfdetr", variant, preprocess_rfdetr)
        for variant in rfdetr_variants
    ]
    jobs = [
        (name, family, variant, preprocess_fn, runtime, size)
        for name, family, variant, preprocess_fn in models
        for runtime in runtimes
        if runtime in MODEL_RUNTIMES[family]
        for size in img_sizes
    ]
    if not jobs:
        log.error("The selected models and runtimes have no supported combinations; nothing to do.")
        return 2
    log.info("Planned %d model/runtime/resolution benchmark cells", len(jobs))
    job_runtimes = list(dict.fromkeys(job[4] for job in jobs))
    active_yolox_variants = list(dict.fromkeys(job[2] for job in jobs if job[1] == "yolox"))
    active_rfdetr_variants = list(dict.fromkeys(job[2] for job in jobs if job[1] == "rfdetr"))

    # One-time framework/checkpoint setup. Failures remain local to affected cells.
    yolox_repo_dir: Optional[Path] = None
    yolox_checkpoints: dict[str, Optional[Path]] = {variant: None for variant in active_yolox_variants}
    if active_yolox_variants and (not args.skip_export or "pytorch" in job_runtimes):
        try:
            yolox_repo_dir = setup_yolox_repo(work_dir)
            if yolox_repo_dir:
                for variant in active_yolox_variants:
                    checkpoint = work_dir / f"yolox_{variant}.pth"
                    if (checkpoint.exists() and checkpoint.stat().st_size > 0) or download_file(
                        YOLOX_CKPT_URLS[variant], checkpoint
                    ):
                        yolox_checkpoints[variant] = checkpoint
        except Exception as e:
            log.error("Unexpected error during YOLOX setup: %s", e)
            log.debug(traceback.format_exc())

    if active_yolox_variants and "ncnn" in job_runtimes and not args.skip_export:
        if not uv_install(["-q", "ncnn", "pnnx"], timeout=1800):
            log.error("Failed to install ncnn/pnnx; YOLOX ncnn cells may fail.")

    if active_rfdetr_variants and (not args.skip_export or "pytorch" in job_runtimes):
        extras: list[str] = []
        if not args.skip_export:
            if any(runtime in job_runtimes for runtime in ("onnxruntime", "openvino")):
                extras.append("onnx")
            if "tflite" in job_runtimes:
                extras.append("tflite")
            if "executorch" in job_runtimes:
                extras.append("executorch")
        package = f"rfdetr[{','.join(extras)}]" if extras else "rfdetr"
        if not uv_install(["-q", package], timeout=3600):
            log.error("Failed to install %s; RF-DETR cells may fail.", package)

    # Resolve/export every artifact required by the canonical job list. ONNX is
    # shared by ONNX Runtime and OpenVINO; other runtimes use native artifacts.
    artifact_by_key: dict[tuple[str, int, str], Optional[Path]] = {}
    artifact_manifest_by_key: dict[tuple[str, int, str], dict] = {}
    artifact_kind = {
        "onnxruntime": "onnx",
        "openvino": "onnx",
        "ncnn": "ncnn",
        "tflite": "tflite",
        "executorch": "executorch",
    }
    extensions = {"onnx": ".onnx", "ncnn": ".param", "tflite": ".tflite", "executorch": ".pte"}
    artifact_requests = list(
        dict.fromkeys(
            (name, family, variant, size, artifact_kind[runtime])
            for name, family, variant, _, runtime, size in jobs
            if runtime != "pytorch"
        )
    )
    for name, family, variant, size, kind in artifact_requests:
        root = args.artifact_dir.resolve() if args.artifact_dir else default_artifact_dir
        destination = root / name / str(size) / kind / f"{name}{extensions[kind]}"
        if args.skip_export and kind == "onnx" and not destination.exists():
            legacy_onnx = root / name / str(size) / f"{name}.onnx"
            if legacy_onnx.exists():
                destination = legacy_onnx
        manifest_path = destination.with_suffix(".export.json")
        export_config = {
            "model": name,
            "family": family,
            "artifact_kind": kind,
            "img_size": size,
            "num_classes": args.num_classes,
            "opset": args.opset,
        }
        versions = exporter_versions(family, kind)
        artifact: Optional[Path] = None
        manifest: Optional[dict] = None
        valid, loaded_manifest, reason = validate_artifact_manifest(
            destination,
            kind,
            manifest_path,
            export_config,
        )
        if valid:
            log.info("Reusing verified export %s", destination)
            artifact, manifest = destination, loaded_manifest
        elif args.skip_export:
            complete = all(item.is_file() and item.stat().st_size > 0 for item in artifact_components(destination, kind))
            if complete and args.allow_unsigned_artifacts:
                log.warning("Using explicitly allowed unsigned artifact %s (%s)", destination, reason)
                try:
                    manifest = build_artifact_manifest(
                        destination, kind, export_config, {"legacy": "unsigned"}
                    )
                    artifact = destination
                except OSError as e:
                    log.error("Could not hash legacy artifact %s: %s", destination, e)
            else:
                log.warning("Rejecting unverified pre-exported model %s: %s", destination, reason)
        else:
            try:
                if family == "yolox" and kind == "onnx":
                    checkpoint = yolox_checkpoints[variant]
                    if yolox_repo_dir and checkpoint and export_yolox_onnx(
                        yolox_repo_dir, variant, size, checkpoint, destination, args.opset, args.num_classes
                    ):
                        artifact = destination
                elif family == "yolox" and kind == "ncnn":
                    checkpoint = yolox_checkpoints[variant]
                    if yolox_repo_dir and checkpoint and export_yolox_ncnn(
                        yolox_repo_dir, variant, size, checkpoint, destination, args.num_classes
                    ):
                        artifact = destination
                elif kind == "onnx":
                    artifact = export_rfdetr_onnx(
                        variant, size, destination.parent, args.opset, args.num_classes
                    )
                else:
                    artifact = export_rfdetr_deployment(
                        variant, size, destination, kind, args.num_classes
                    )
            except Exception as e:
                log.error("Export failed for %s @ %dx%d: %s", name, size, size, e)
                log.debug(traceback.format_exc())
            if artifact is not None:
                try:
                    manifest = build_artifact_manifest(artifact, kind, export_config, versions)
                    # This commit marker is written last. A crash while replacing
                    # an ncnn param/bin pair therefore cannot publish a mixed pair.
                    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
                except OSError as e:
                    log.error("Could not commit export manifest for %s: %s", artifact, e)
                    artifact, manifest = None, None
        for job_name, _, _, _, runtime, job_size in jobs:
            if job_name == name and job_size == size and artifact_kind.get(runtime) == kind:
                key = (name, size, runtime)
                artifact_by_key[key] = artifact
                if manifest is not None:
                    artifact_manifest_by_key[key] = manifest

    if args.export_only:
        log.info("--export-only set; exiting after export.")
        return 0 if any(artifact_by_key.values()) else 1

    # ---- test images (resolution-independent) ----------------------------
    try:
        image_paths = get_test_images(args.images, args.num_fallback_images, work_dir)
    except Exception as e:
        log.error("Could not obtain any test images: %s", e)
        return 1

    artifact_identities = {
        f"{name}/{runtime}/{size}": artifact_manifest_by_key.get(
            (name, size, runtime), {"artifact_id": "missing"}
        )["artifact_id"]
        for name, _, _, _, runtime, size in jobs
        if runtime != "pytorch"
    }
    artifact_exporter_versions = {
        f"{name}/{runtime}/{size}": artifact_manifest_by_key[(name, size, runtime)][
            "exporter_versions"
        ]
        for name, _, _, _, runtime, size in jobs
        if (name, size, runtime) in artifact_manifest_by_key
    }
    source_identities: dict[str, object] = {}
    if any(family == "rfdetr" and runtime == "pytorch" for _, family, _, _, runtime, _ in jobs):
        source_identities["rfdetr"] = package_version("rfdetr")
    if yolox_repo_dir and any(
        family == "yolox" and runtime == "pytorch" for _, family, _, _, runtime, _ in jobs
    ):
        ok, revision = run_cmd(["git", "rev-parse", "HEAD"], cwd=yolox_repo_dir, timeout=30)
        source_identities["yolox_revision"] = revision.strip() if ok else "unknown"
        source_identities["yolox_checkpoints"] = {
            variant: sha256_file(checkpoint)
            for variant, checkpoint in yolox_checkpoints.items()
            if checkpoint is not None and checkpoint.is_file()
        }
    signature = {
        "jobs": [f"{name}/{runtime}/{size}" for name, _, _, _, runtime, size in jobs],
        "num_classes": args.num_classes,
        "opset": args.opset,
        "threads": args.threads,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "images": str(args.images.resolve()) if args.images else "fallback",
        "artifact_identities": artifact_identities,
        "artifact_exporter_versions": artifact_exporter_versions,
        "runtime_versions": runtime_versions(job_runtimes),
        "source_identities": source_identities,
    }
    checkpoint_path = out_dir / "checkpoint.json"
    results: list[ModelResult] = []
    if not args.no_resume and checkpoint_path.exists():
        try:
            saved = json.loads(checkpoint_path.read_text())
            if saved.get("signature") == signature:
                results = [ModelResult(**item) for item in saved.get("results", [])]
                log.info("Resuming from %s with %d completed cells", checkpoint_path, len(results))
            else:
                log.info("Ignoring checkpoint because its run configuration differs")
        except Exception as e:
            log.warning("Ignoring unreadable checkpoint %s: %s", checkpoint_path, e)

    completed = {(r.name, r.runtime, r.img_size) for r in results}
    for name, family, variant, preprocess_fn, runtime, size in jobs:
        key = (name, runtime, size)
        if key in completed:
            continue
        result = benchmark_model(
            name=name,
            family=family,
            variant=variant,
            runtime=runtime,
            artifact_path=artifact_by_key.get((name, size, runtime)),
            preprocess_fn=preprocess_fn,
            image_paths=image_paths,
            img_size=size,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
            threads=args.threads,
            num_classes=args.num_classes,
            yolox_repo=yolox_repo_dir,
            yolox_ckpt=yolox_checkpoints.get(variant),
        )
        results.append(result)
        completed.add(key)
        write_checkpoint(results, checkpoint_path, signature)

    meta = {
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "img_sizes": ", ".join(f"{s}x{s}" for s in img_sizes),
        "models": ", ".join(dict.fromkeys(job[0] for job in jobs)),
        "runtimes": ", ".join(job_runtimes),
        "runtime_versions": json.dumps(signature["runtime_versions"], sort_keys=True),
        "num_classes": args.num_classes,
        "threads": args.threads,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "num_test_images": len(image_paths),
    }

    write_html(results, out_dir / "index.html", meta)
    write_csv(results, out_dir / "per_run_timings.csv")
    atomic_write_text(
        out_dir / "results.json",
        json.dumps({"meta": meta, "results": [r.__dict__ for r in results]}, indent=2),
    )

    any_ok = any(r.status == "ok" for r in results)
    print(f"\nHTML report: {out_dir / 'index.html'}")
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
