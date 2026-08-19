#!/usr/bin/env python3
"""
CPU-only detector benchmark across every core RF-DETR detection variant and
every official YOLOX variant, using PyTorch, ONNX Runtime, and OpenVINO.

Target hardware: Raspberry Pi 4 (ARM Cortex-A72, CPU-only, no accelerator).
See SETUP.md for install instructions, Pi-specific caveats, and expected runtime.

This script is meant to run unattended on a Raspberry Pi with nobody available
to debug it interactively. Every external operation (pip install, git clone,
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

    # export on a beefier dev machine, copy the .onnx files to the Pi, then
    # only run the timing phase on the Pi (skips the heavy torch/rfdetr/YOLOX
    # install on-device):
    python3 benchmark.py --export-only --work-dir ./work
    # ... copy ./results/onnx to the Pi ...
    python3 benchmark.py --skip-export --onnx-dir ./onnx
"""

from __future__ import annotations

import argparse
import csv
import html
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
CPU_RUNTIMES = ("pytorch", "onnxruntime", "openvino")

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
        # Keep the tail of the log; full pip/build output can be huge.
        tail = "\n".join(proc.stdout.splitlines()[-60:]) if proc.stdout else ""
        log.error(
            "command failed (exit %s): %s\n--- last output lines ---\n%s",
            proc.returncode,
            " ".join(cmd),
            tail,
        )
    return ok, proc.stdout or ""


def pip_install(args: list[str], timeout: int = 1800) -> bool:
    ok, _ = run_cmd([sys.executable, "-m", "pip", "install"] + args, timeout=timeout)
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
        "Pi 4 — see SETUP.md for expected timing and how to avoid it via "
        "--export-only on a faster machine."
    )
    if not pip_install(["-q", "-r", str(repo_dir / "requirements.txt")], timeout=3600):
        log.error(
            "Failed to install YOLOX requirements.txt. This most commonly means "
            "torch failed to install. On a Pi, prefer the official PyTorch CPU "
            "wheels: pip install torch --index-url https://download.pytorch.org/whl/cpu "
            "(see SETUP.md)."
        )
        return None
    if not pip_install(["-q", "-e", str(repo_dir)], timeout=1200):
        log.error("Failed to `pip install -e` the YOLOX repo itself.")
        return None
    return repo_dir


def export_yolox_onnx(
    repo_dir: Path, variant: str, img_size: int, ckpt_path: Path, out_path: Path, opset: int, num_classes: int
) -> bool:
    try:
        sys.path.insert(0, str(repo_dir))
        from yolox.exp import get_exp  # type: ignore
        import torch  # type: ignore
    except Exception as e:
        log.error("Failed to import YOLOX/torch for export: %s", e)
        return False

    try:
        exp_file = repo_dir / YOLOX_EXP_FILE[variant]
        exp = get_exp(str(exp_file), None)
        # Override the experiment's default 416x416 test size so both models
        # are compared at the same resolution for each entry in --img-sizes.
        exp.test_size = (img_size, img_size)
        # Override the pretrained checkpoint's 80-class (COCO) head shape with
        # --num-classes. The backbone/neck stay pretrained; only the head's
        # classification conv is shape-mismatched and gets randomly
        # reinitialized below (fine for a latency-only benchmark).
        exp.num_classes = num_classes

        model = exp.get_model()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        model_state = model.state_dict()
        compatible_state = {
            k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape
        }
        skipped = sorted(set(state_dict) - set(compatible_state))
        if skipped:
            log.warning(
                "num_classes=%d differs from the checkpoint's trained class count; "
                "%d head tensor(s) shape-mismatched and left randomly initialized: %s",
                num_classes,
                len(skipped),
                skipped,
            )
        model.load_state_dict(compatible_state, strict=False)
        model.eval()
        # Fuse decode step into a plain conv graph rather than the training-time
        # decoding path, matching what YOLOX's own tools/export_onnx.py does by
        # default when --decode_in_inference is not passed.
        if hasattr(model.head, "decode_in_inference"):
            model.head.decode_in_inference = False

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
    file_size_mb: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    std_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    fps: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
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
    onnx_path: Optional[Path],
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
        onnx_path=str(onnx_path) if onnx_path else "",
    )

    if runtime != "pytorch":
        if onnx_path is None:
            result.status = "failed"
            result.error = "export/skip-export did not produce an ONNX file"
            return result
        try:
            result.file_size_mb = human_mb(onnx_path.stat().st_size)
        except OSError as e:
            result.status = "failed"
            result.error = f"could not stat ONNX file: {e}"
            return result

    try:
        if runtime == "onnxruntime":
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name
            infer = lambda tensor: session.run(None, {input_name: tensor})
        elif runtime == "openvino":
            import openvino as ov

            core = ov.Core()
            compiled = core.compile_model(
                str(onnx_path), "CPU", {"INFERENCE_NUM_THREADS": threads}
            )
            input_port = compiled.input(0)
            infer = lambda tensor: compiled({input_port: tensor})
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
                sys.path.insert(0, str(yolox_repo))
                from yolox.exp import get_exp  # type: ignore

                exp = get_exp(str(yolox_repo / YOLOX_EXP_FILE[variant]), None)
                exp.test_size = (img_size, img_size)
                exp.num_classes = num_classes
                model = exp.get_model()
                checkpoint = torch.load(str(yolox_ckpt), map_location="cpu")
                state = checkpoint.get("model", checkpoint)
                current = model.state_dict()
                model.load_state_dict(
                    {k: v for k, v in state.items() if k in current and current[k].shape == v.shape},
                    strict=False,
                )
                model.eval()
                model.head.decode_in_inference = False
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
<th>Min ms</th><th>Max ms</th><th>FPS</th><th>Peak RSS MB</th><th>CPU %</th><th>ONNX MB</th><th>Runs</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{f'<h2>Errors</h2><ul>{errors}</ul>' if errors else ''}
<h2>Notes</h2><ul><li>Latency excludes model loading, export, and preprocessing.</li>
<li>CPU percentage is per process and is not normalized by core count.</li>
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
        help="Comma-separated CPU runtimes: pytorch,onnxruntime,openvino (default: all)",
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
        help="Skip export; load existing .onnx files from --onnx-dir instead",
    )
    p.add_argument(
        "--onnx-dir",
        type=Path,
        default=None,
        help=(
            "Root copied from results/onnx, containing "
            "<model>/<resolution>/<model>.onnx; used with --skip-export"
        ),
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore a matching checkpoint and rerun completed benchmark cells",
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
    onnx_dir = out_dir / "onnx"
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
    if not runtimes or not (yolox_variants or rfdetr_variants):
        log.error("No models or runtimes selected; nothing to do.")
        return 1

    models = [
        (f"yolox_{variant}", "yolox", variant, preprocess_yolox)
        for variant in yolox_variants
    ] + [
        (f"rfdetr_{variant}", "rfdetr", variant, preprocess_rfdetr)
        for variant in rfdetr_variants
    ]

    # One-time framework/checkpoint setup. Failures remain local to affected cells.
    yolox_repo_dir: Optional[Path] = None
    yolox_checkpoints: dict[str, Optional[Path]] = {variant: None for variant in yolox_variants}
    if yolox_variants and (not args.skip_export or "pytorch" in runtimes):
        try:
            yolox_repo_dir = setup_yolox_repo(work_dir)
            if yolox_repo_dir:
                for variant in yolox_variants:
                    checkpoint = work_dir / f"yolox_{variant}.pth"
                    if (checkpoint.exists() and checkpoint.stat().st_size > 0) or download_file(
                        YOLOX_CKPT_URLS[variant], checkpoint
                    ):
                        yolox_checkpoints[variant] = checkpoint
        except Exception as e:
            log.error("Unexpected error during YOLOX setup: %s", e)
            log.debug(traceback.format_exc())

    if rfdetr_variants and (not args.skip_export or "pytorch" in runtimes):
        if not pip_install(["-q", "rfdetr[onnx]"], timeout=1800):
            log.error("Failed to install rfdetr[onnx]; RF-DETR cells may fail.")

    # Resolve or export each static ONNX artifact. Existing non-empty artifacts
    # are reused, making the expensive export phase naturally resumable.
    onnx_by_key: dict[tuple[str, int], Optional[Path]] = {}
    need_onnx = args.export_only or any(r != "pytorch" for r in runtimes)
    if need_onnx:
        for name, family, variant, _ in models:
            for size in img_sizes:
                root = (args.onnx_dir.resolve() if args.onnx_dir else onnx_dir)
                destination = root / name / str(size) / f"{name}.onnx"
                manifest_path = destination.with_suffix(".export.json")
                export_signature = {
                    "model": name,
                    "img_size": size,
                    "num_classes": args.num_classes,
                    "opset": args.opset,
                }
                artifact: Optional[Path] = None
                manifest_matches = False
                if manifest_path.exists():
                    try:
                        manifest_matches = json.loads(manifest_path.read_text()) == export_signature
                    except (OSError, json.JSONDecodeError):
                        pass
                if (
                    destination.exists()
                    and destination.stat().st_size > 0
                    and (args.skip_export or manifest_matches)
                ):
                    log.info("Reusing export %s", destination)
                    artifact = destination
                elif args.skip_export:
                    log.warning("Missing pre-exported model: %s", destination)
                else:
                    try:
                        if family == "yolox":
                            checkpoint = yolox_checkpoints[variant]
                            if yolox_repo_dir and checkpoint and export_yolox_onnx(
                                yolox_repo_dir,
                                variant,
                                size,
                                checkpoint,
                                destination,
                                args.opset,
                                args.num_classes,
                            ):
                                artifact = destination
                        else:
                            artifact = export_rfdetr_onnx(
                                variant, size, destination.parent, args.opset, args.num_classes
                            )
                    except Exception as e:
                        log.error("Export failed for %s @ %dx%d: %s", name, size, size, e)
                        log.debug(traceback.format_exc())
                    if artifact is not None:
                        atomic_write_text(manifest_path, json.dumps(export_signature, indent=2))
                onnx_by_key[(name, size)] = artifact

    if args.export_only:
        log.info("--export-only set; exiting after export.")
        return 0 if any(onnx_by_key.values()) else 1

    # ---- test images (resolution-independent) ----------------------------
    try:
        image_paths = get_test_images(args.images, args.num_fallback_images, work_dir)
    except Exception as e:
        log.error("Could not obtain any test images: %s", e)
        return 1

    signature = {
        "models": [name for name, _, _, _ in models],
        "runtimes": runtimes,
        "img_sizes": img_sizes,
        "num_classes": args.num_classes,
        "threads": args.threads,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "images": str(args.images.resolve()) if args.images else "fallback",
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
    for name, family, variant, preprocess_fn in models:
        for runtime in runtimes:
            for size in img_sizes:
                key = (name, runtime, size)
                if key in completed:
                    continue
                result = benchmark_model(
                    name=name,
                    family=family,
                    variant=variant,
                    runtime=runtime,
                    onnx_path=onnx_by_key.get((name, size)),
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
        "models": ", ".join(signature["models"]),
        "runtimes": ", ".join(runtimes),
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
