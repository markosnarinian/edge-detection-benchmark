#!/usr/bin/env python3
"""
CPU-only ONNX Runtime benchmark: RF-DETR Nano vs YOLOX-Nano/Tiny, swept across
multiple input resolutions.

Target hardware: Raspberry Pi 4 (ARM Cortex-A72, CPU-only, no accelerator).
See SETUP.md for install instructions, Pi-specific caveats, and expected runtime.

This script is meant to run unattended on a Raspberry Pi with nobody available
to debug it interactively. Every external operation (pip install, git clone,
checkpoint download, ONNX export, inference) is wrapped so a single failure is
logged clearly and skipped rather than crashing the whole run. Partial results
(e.g. only YOLOX succeeded, or only some resolutions) are still written out.

Resolutions are swept via --img-sizes (comma-separated, default covers the
full practically-supported range: 320-1024 in steps of 32). Both models
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
    # ... copy ./work/onnx/* to the Pi (one subdir per resolution) ...
    python3 benchmark.py --skip-export --onnx-dir ./work/onnx
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
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
}
YOLOX_EXP_FILE = {
    "nano": "exps/default/yolox_nano.py",
    "tiny": "exps/default/yolox_tiny.py",
}

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
    for attempt in range(1, retries + 2):
        try:
            log.info("Downloading %s -> %s (attempt %d)", url, dest, attempt)
            req = urllib.request.Request(url, headers={"User-Agent": "edge-bench/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(
                dest, "wb"
            ) as f:
                shutil.copyfileobj(resp, f)
            if dest.stat().st_size == 0:
                raise IOError("downloaded file is empty")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, IOError, OSError) as e:
            log.warning("Download failed (%s): %s", url, e)
            if dest.exists():
                dest.unlink(missing_ok=True)
            time.sleep(1.5 * attempt)
    log.error("Giving up on download after %d attempts: %s", retries + 1, url)
    return False


def human_mb(num_bytes: float) -> float:
    return round(num_bytes / (1024 * 1024), 2)


DEFAULT_IMG_SIZES = "320,416,512,640,768,896,1024"


def parse_img_sizes(raw: str) -> list[int]:
    """Parse a comma-separated --img-sizes value into a sorted, deduplicated
    list of valid resolutions. Both models require the resolution to be a
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
    repo_dir: Path, variant: str, img_size: int, ckpt_path: Path, out_path: Path, opset: int
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

        model = exp.get_model()
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict)
        model.eval()
        # Fuse decode step into a plain conv graph rather than the training-time
        # decoding path, matching what YOLOX's own tools/export_onnx.py does by
        # default when --decode_in_inference is not passed.
        if hasattr(model.head, "decode_in_inference"):
            model.head.decode_in_inference = False

        dummy = torch.randn(1, 3, img_size, img_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["images"],
            output_names=["output"],
            opset_version=opset,
            do_constant_folding=True,
        )

        try:
            import onnxsim  # type: ignore
            import onnx  # type: ignore

            onnx_model = onnx.load(str(out_path))
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(out_path))
                log.info("onnx-simplifier applied to %s", out_path.name)
            else:
                log.warning("onnx-simplifier reported failure; keeping unsimplified export")
        except ImportError:
            log.info("onnx-simplifier not installed; skipping simplification (optional)")
        except Exception as e:
            log.warning("onnx-simplifier step failed non-fatally: %s", e)

        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as e:
        log.error("YOLOX ONNX export failed for variant=%s: %s", variant, e)
        log.debug(traceback.format_exc())
        return False


# --------------------------------------------------------------------------
# Export: RF-DETR
# --------------------------------------------------------------------------


def export_rfdetr_onnx(img_size: int, out_dir: Path, opset: int) -> Optional[Path]:
    if not pip_install(["-q", "rfdetr[onnx]"], timeout=1800):
        log.error(
            "Failed to install rfdetr[onnx]. Check internet connectivity and "
            "available disk space (`df -h`)."
        )
        return None

    try:
        from rfdetr import RFDETRNano  # type: ignore
    except Exception as e:
        log.error("Failed to import rfdetr after install: %s", e)
        return None

    try:
        log.info(
            "Instantiating RFDETRNano() at resolution=%d — this auto-downloads "
            "the official pretrained COCO checkpoint on first run.",
            img_size,
        )
        model = RFDETRNano(resolution=img_size)
    except Exception as e:
        log.error(
            "RFDETRNano(resolution=%d) failed to construct: %s\n"
            "RF-DETR Nano requires the resolution to be divisible by "
            "patch_size(16) * num_windows(2) = 32 (this should already be "
            "enforced by --img-sizes validation; if you're seeing this, check "
            "for a version mismatch in the installed rfdetr package).",
            img_size,
            e,
        )
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.export(
            output_dir=str(out_dir),
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

    candidate = out_dir / "inference_model.onnx"
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate

    # Defensive fallback: export() may name the file differently across
    # rfdetr versions. Take the newest .onnx file written into out_dir.
    onnx_files = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if onnx_files:
        log.warning(
            "Expected inference_model.onnx but found %s instead; using it.",
            onnx_files[0].name,
        )
        return onnx_files[0]

    log.error("RF-DETR export() completed without error but no .onnx file was found in %s", out_dir)
    return None


# --------------------------------------------------------------------------
# Benchmarking
# --------------------------------------------------------------------------


@dataclass
class ModelResult:
    name: str
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


def benchmark_onnx_model(
    name: str,
    onnx_path: Path,
    preprocess_fn: Callable[[Path, int], "object"],
    image_paths: list[Path],
    img_size: int,
    warmup_runs: int,
    timed_runs: int,
    threads: int,
) -> ModelResult:
    import numpy as np  # noqa: F401  (ensures numpy import errors surface here, not deep in preprocess)
    import psutil

    result = ModelResult(name=name, img_size=img_size, onnx_path=str(onnx_path))

    try:
        result.file_size_mb = human_mb(onnx_path.stat().st_size)
    except OSError as e:
        result.status = "failed"
        result.error = f"could not stat onnx file: {e}"
        return result

    try:
        import onnxruntime as ort
    except Exception as e:
        result.status = "failed"
        result.error = f"onnxruntime import failed: {e}"
        log.error(
            "onnxruntime is not importable (%s). Install with `pip install "
            "onnxruntime` (see SETUP.md for the ARM-specific notes).",
            e,
        )
        return result

    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
    except Exception as e:
        result.status = "failed"
        result.error = f"failed to create ONNX Runtime session: {e}"
        log.error("Session creation failed for %s: %s", onnx_path, e)
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
        session.run(None, {input_name: tensor})

    try:
        log.info("%s: warming up (%d runs)...", result.name, warmup_runs)
        for i in range(warmup_runs):
            run_once(i)

        proc = psutil.Process(os.getpid())
        proc.cpu_percent(interval=None)  # prime the internal counter
        sampler = _RssSampler().start()

        log.info("%s: timing %d runs (%d thread(s))...", result.name, timed_runs, threads)
        per_run_ms = []
        wall_start = time.perf_counter()
        for i in range(timed_runs):
            t0 = time.perf_counter()
            run_once(i)
            per_run_ms.append((time.perf_counter() - t0) * 1000.0)
        wall_elapsed = time.perf_counter() - wall_start

        peak_rss = sampler.stop()
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

    return result


# --------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------


def write_csv(results: list[ModelResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "img_size", "run_index", "latency_ms"])
        for r in results:
            for i, ms in enumerate(r.per_run_ms):
                w.writerow([r.name, r.img_size, i, round(ms, 4)])
    log.info("Wrote per-run timings to %s", path)


def write_markdown(results: list[ModelResult], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# CPU-Only ONNX Runtime Benchmark: RF-DETR Nano vs YOLOX\n")
    lines.append("## Run metadata\n")
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Pivot table: FPS by resolution, one column per model. Quick way to read
    # the speed/resolution tradeoff curve for each architecture at a glance.
    model_names: list[str] = []
    for r in results:
        if r.name not in model_names:
            model_names.append(r.name)
    sizes_present = sorted({r.img_size for r in results})
    by_key = {(r.name, r.img_size): r for r in results}

    if len(sizes_present) > 1:
        lines.append("## FPS by resolution\n")
        header = ["Resolution"] + model_names
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for s in sizes_present:
            row = [f"{s}x{s}"]
            for m in model_names:
                rr = by_key.get((m, s))
                if rr is None:
                    row.append("-")
                elif rr.status != "ok":
                    row.append("FAILED")
                else:
                    row.append(str(rr.fps))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.append("## Results\n")
    header = [
        "Model",
        "Resolution",
        "Status",
        "Avg latency (ms)",
        "Std (ms)",
        "Min / Max (ms)",
        "FPS",
        "Peak RSS (MB)",
        "CPU %",
        "File size (MB)",
        "Timed runs",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in sorted(results, key=lambda r: (r.name, r.img_size)):
        res = f"{r.img_size}x{r.img_size}"
        if r.status != "ok":
            lines.append(
                f"| {r.name} | {res} | FAILED | - | - | - | - | - | - | "
                f"{r.file_size_mb if r.file_size_mb is not None else '-'} | 0 |"
            )
            continue
        lines.append(
            f"| {r.name} | {res} | ok | {r.avg_latency_ms} | {r.std_latency_ms} | "
            f"{r.min_latency_ms} / {r.max_latency_ms} | {r.fps} | {r.peak_rss_mb} | "
            f"{r.cpu_percent} | {r.file_size_mb} | {r.num_timed_runs} |"
        )
    lines.append("")

    failed = [r for r in results if r.status != "ok"]
    if failed:
        lines.append("## Errors\n")
        for r in sorted(failed, key=lambda r: (r.name, r.img_size)):
            lines.append(f"- **{r.name}** @ {r.img_size}x{r.img_size}: {r.error}")
        lines.append("")

    lines.append(
        "## Notes\n\n"
        "- Latency excludes model load and ONNX export time; only the timed "
        "inference loop (after warmup) is measured.\n"
        "- CPU % is per-process (`psutil`), not normalized by core count — a "
        "value near 400% on a quad-core Pi means all 4 cores were saturated.\n"
        "- Raw per-run timings are in the accompanying CSV file for outlier "
        "inspection.\n"
    )
    path.write_text("\n".join(lines))
    log.info("Wrote markdown report to %s", path)


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
            "practically-supported range (default: %(default)s). Both models "
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
        "--yolox-variant", choices=["nano", "tiny"], default="nano", help="Which YOLOX variant to benchmark"
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
        help="onnxruntime intra_op_num_threads (default: all detected cores)",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version for export (default: 17)")
    p.add_argument("--work-dir", type=Path, default=Path("./work"), help="Scratch dir for repo clones/checkpoints")
    p.add_argument("--output-dir", type=Path, default=Path("./bench_output"), help="Where results are written")
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
            "Directory with pre-exported onnx files, one subdirectory per "
            "resolution (e.g. <dir>/640/yolox_nano.onnx); used with --skip-export"
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
    onnx_dir = out_dir / "onnx"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Platform: %s | Python: %s | CPUs: %s", platform.platform(), platform.python_version(), os.cpu_count())

    img_sizes = parse_img_sizes(args.img_sizes)
    if not img_sizes:
        log.error("No valid resolutions remain in --img-sizes %r; nothing to do.", args.img_sizes)
        return 1
    log.info("Resolution sweep: %s", ", ".join(f"{s}x{s}" for s in img_sizes))

    results: list[ModelResult] = []
    # onnx_by_size[size] = {"yolox": Optional[Path], "rfdetr": Optional[Path]}
    onnx_by_size: dict[int, dict[str, Optional[Path]]] = {}

    # ---- one-time YOLOX repo/checkpoint setup (resolution-independent) ----
    yolox_repo_dir: Optional[Path] = None
    yolox_ckpt_path: Optional[Path] = None
    if not args.skip_export and not args.skip_yolox:
        try:
            yolox_repo_dir = setup_yolox_repo(work_dir)
            if yolox_repo_dir is not None:
                yolox_ckpt_path = work_dir / f"yolox_{args.yolox_variant}.pth"
                if not yolox_ckpt_path.exists() and not download_file(
                    YOLOX_CKPT_URLS[args.yolox_variant], yolox_ckpt_path
                ):
                    log.error(
                        "Could not download YOLOX-%s checkpoint; YOLOX will be FAILED at every resolution.",
                        args.yolox_variant,
                    )
                    yolox_ckpt_path = None
        except Exception as e:
            log.error("Unexpected error during YOLOX setup: %s", e)
            log.debug(traceback.format_exc())

    # ---- resolve/export ONNX files, per resolution -------------------
    for size in img_sizes:
        size_dir = onnx_dir / str(size)
        yolox_onnx: Optional[Path] = None
        rfdetr_onnx: Optional[Path] = None

        if args.skip_export:
            src = (args.onnx_dir or onnx_dir) / str(size)
            if not src.is_dir():
                log.warning("--skip-export: no onnx directory for resolution %dx%d (%s)", size, size, src)
            else:
                yx = list(src.glob(f"yolox_{args.yolox_variant}*.onnx"))
                rf = list(src.glob("rfdetr*.onnx")) + list(src.glob("inference_model.onnx"))
                yolox_onnx = yx[0] if yx else None
                rfdetr_onnx = rf[0] if rf else None
                if not args.skip_yolox and not yolox_onnx:
                    log.warning("No matching yolox_%s*.onnx found in %s", args.yolox_variant, src)
                if not args.skip_rfdetr and not rfdetr_onnx:
                    log.warning("No matching rfdetr/inference_model.onnx found in %s", src)
        else:
            if not args.skip_yolox and yolox_repo_dir is not None and yolox_ckpt_path is not None:
                log.info("=== Exporting YOLOX-%s @ %dx%d ===", args.yolox_variant, size, size)
                try:
                    dest = size_dir / f"yolox_{args.yolox_variant}.onnx"
                    if export_yolox_onnx(yolox_repo_dir, args.yolox_variant, size, yolox_ckpt_path, dest, args.opset):
                        yolox_onnx = dest
                    else:
                        log.error(
                            "YOLOX export failed at %dx%d; YOLOX will be FAILED at this resolution.", size, size
                        )
                except Exception as e:
                    log.error("Unexpected error exporting YOLOX at %dx%d: %s", size, size, e)
                    log.debug(traceback.format_exc())

            if not args.skip_rfdetr:
                log.info("=== Exporting RF-DETR Nano @ %dx%d ===", size, size)
                try:
                    rfdetr_onnx = export_rfdetr_onnx(size, size_dir, args.opset)
                except Exception as e:
                    log.error("Unexpected error exporting RF-DETR at %dx%d: %s", size, size, e)
                    log.debug(traceback.format_exc())

        onnx_by_size[size] = {"yolox": yolox_onnx, "rfdetr": rfdetr_onnx}

    if args.export_only:
        log.info("--export-only set; exiting after export.")
        any_ok = False
        for size in img_sizes:
            d = onnx_by_size[size]
            log.info(
                "  %dx%d: yolox=%s rfdetr=%s",
                size,
                size,
                d["yolox"] or "FAILED/SKIPPED",
                d["rfdetr"] or "FAILED/SKIPPED",
            )
            any_ok = any_ok or bool(d["yolox"]) or bool(d["rfdetr"])
        return 0 if any_ok else 1

    # ---- test images (resolution-independent) ----------------------------
    try:
        image_paths = get_test_images(args.images, args.num_fallback_images, work_dir)
    except Exception as e:
        log.error("Could not obtain any test images: %s", e)
        return 1

    # ---- benchmark, per resolution ------------------------------------
    for size in img_sizes:
        d = onnx_by_size[size]

        if not args.skip_yolox:
            name = f"yolox_{args.yolox_variant}"
            if d["yolox"] is None:
                r = ModelResult(
                    name=name,
                    img_size=size,
                    status="failed",
                    error="export/skip-export did not produce an onnx file",
                )
            else:
                r = benchmark_onnx_model(
                    name, d["yolox"], preprocess_yolox, image_paths, size, args.warmup_runs, args.timed_runs, args.threads
                )
            results.append(r)

        if not args.skip_rfdetr:
            if d["rfdetr"] is None:
                r = ModelResult(
                    name="rfdetr_nano",
                    img_size=size,
                    status="failed",
                    error="export/skip-export did not produce an onnx file",
                )
            else:
                r = benchmark_onnx_model(
                    "rfdetr_nano", d["rfdetr"], preprocess_rfdetr, image_paths, size, args.warmup_runs, args.timed_runs, args.threads
                )
            results.append(r)

    if not results:
        log.error("Both models skipped (--skip-yolox and --skip-rfdetr); nothing to do.")
        return 1

    meta = {
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "img_sizes": ", ".join(f"{s}x{s}" for s in img_sizes),
        "threads": args.threads,
        "warmup_runs": args.warmup_runs,
        "timed_runs": args.timed_runs,
        "num_test_images": len(image_paths),
    }

    write_markdown(results, out_dir / "results.md", meta)
    write_csv(results, out_dir / "per_run_timings.csv")
    (out_dir / "results.json").write_text(
        json.dumps({"meta": meta, "results": [r.__dict__ for r in results]}, indent=2)
    )

    any_ok = any(r.status == "ok" for r in results)
    print("\n" + (out_dir / "results.md").read_text())
    return 0 if any_ok else 1


if __name__ == "__main__":
    sys.exit(main())
