# SETUP.md — Running `benchmark.py` on a Raspberry Pi 4

This benchmarks **RF-DETR Nano** vs **YOLOX-Nano/Tiny** on CPU-only ONNX
Runtime. Written for a Pi 4 (Cortex-A72, no accelerator), 64-bit Raspberry Pi
OS. It should also run on a Pi 5 unmodified if you want a second data point.

## 0. Two ways to run this

Exporting to ONNX requires the heavy training-side dependencies (PyTorch, the
YOLOX repo, `rfdetr`). Running the benchmark itself only needs `onnxruntime`,
`numpy`, `Pillow`, and `psutil`. On a Pi 4 the export-side install is by far
the slowest and most failure-prone part of this whole process, so you have a
choice:

- **Path A — everything on the Pi** (simplest, slowest, ~30–60+ min):
  the script installs torch/YOLOX/rfdetr on-device and does export + benchmark
  in one command.
- **Path B — export on your Mac, benchmark on the Pi** (recommended, faster
  overall, avoids the flakiest step running unattended on the Pi): run
  `benchmark.py --export-only` on your dev machine, copy the resulting
  `.onnx` files to the Pi, then run `benchmark.py --skip-export --onnx-dir ...`
  on the Pi with a much lighter dependency set.

Both paths are covered below.

## 1. Prerequisites (Pi 4)

- **Raspberry Pi OS 64-bit (Bookworm or newer)**. This matters: official
  PyTorch CPU wheels for ARM only publish `aarch64` (64-bit) builds. On a
  32-bit (`armhf`) OS, `pip install torch` will try to build from source,
  which can take hours on a Pi 4 and often fails. Check with:
  ```bash
  uname -m
  # must print aarch64, not armv7l
  ```
- **Python 3.9+** (ships by default on current Raspberry Pi OS).
- **At least ~4 GB free disk space** if doing Path A (torch + torchvision +
  build artifacts + checkpoints add up). Path B only needs a few hundred MB
  on the Pi.
- **At least 4 GB RAM recommended**; on a 2 GB Pi 4, increase swap before
  attempting Path A — `pip install torch` can OOM-kill itself mid-build
  otherwise:
  ```bash
  sudo dphys-swapfile swapoff
  sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' /etc/dphys-swapfile
  sudo dphys-swapfile setup
  sudo dphys-swapfile swapon
  ```
- `git` installed (`sudo apt install -y git`).
- A stable internet connection for the checkpoint/package downloads (several
  hundred MB total).

## 2. Path A — install everything on the Pi

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Official PyTorch CPU wheels — do this BEFORE running the script so it's
# fast and predictable; if you skip this, the script's own `pip install -r
# requirements.txt` step inside the YOLOX repo will pull in whatever torch
# version YOLOX pins, which may not have a prebuilt aarch64 wheel.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install onnxruntime psutil pillow numpy onnx onnxsim

python3 benchmark.py --images ./my_test_images -v
# or, to use the built-in COCO-sample/synthetic fallback instead of your own images:
python3 benchmark.py -v
```

The script itself `pip install`s the YOLOX repo's `requirements.txt` and
`rfdetr[onnx]` as it runs — expect a lot of console output during that phase.

## 3. Path B — export on your Mac/PC, benchmark on the Pi (recommended)

**On your dev machine** (any OS/arch — export doesn't need to happen on ARM):

```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision onnx onnxsim
python3 benchmark.py --export-only --work-dir ./work -v
# --work-dir (./work) is scratch space for the YOLOX repo clone/checkpoints;
# the exported .onnx files themselves land under --output-dir/onnx, i.e.
# ./bench_output/onnx by default
```

Copy the exported files to the Pi:

```bash
scp -r ./bench_output/onnx pi@<pi-ip>:~/edge_detector_bench/onnx
```

**On the Pi**, install only the lightweight runtime deps:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install onnxruntime psutil pillow numpy

python3 benchmark.py --skip-export --onnx-dir ./onnx --images ./my_test_images -v
```

## 4. Pi-specific caveats

- **`onnxruntime` wheel**: confirm you get a native `aarch64` wheel, not a
  slow generic/emulated one:
  ```bash
  python3 -c "import onnxruntime, platform; print(platform.machine(), onnxruntime.__version__)"
  ```
  Should print `aarch64`. If `pip install onnxruntime` fails to find a wheel
  for your Python version, try a slightly older/newer Python (3.10–3.12 have
  the broadest prebuilt-wheel coverage as of 2026) or install from
  `piwheels.org` (Raspberry Pi OS's pip is often pre-configured to use it as
  an extra index for ARM wheels).
- **Cooling / throttling**: run `vcgencmd measure_temp` before/after. If the
  Pi is passively cooled and gets hot during the run, later timed runs could
  read slower than early ones purely from thermal throttling, not model
  differences — check the per-run CSV for a rising trend if numbers look odd.
- **Close other processes** before running (SSH in headless rather than
  running a desktop session, if possible) — the CPU%/RSS numbers are
  per-process, but a busy desktop environment competing for the same 4 cores
  will inflate latency for both models equally, muddying the comparison less
  but still worth avoiding.
- **`--threads`**: defaults to `os.cpu_count()` (4 on Pi 4). If you want to
  see how each architecture scales with thread count (useful signal for the
  Part 3 memory-bandwidth-bound hypothesis — see `pi4_vs_pi5_notes.md`), rerun
  with `--threads 1` and `--threads 2` and compare.
- **Resolution sweep / `--img-sizes`**: the benchmark runs across multiple
  resolutions in one invocation (default `320,416,512,640,768,896,1024` —
  the full practically-supported range). Both models require each resolution
  to be a multiple of 32 (YOLOX's FPN strides top out at 32; RF-DETR Nano's
  ViT backbone requires resolution divisible by `patch_size(16) *
  num_windows(2) = 32`) — any entry that isn't gets dropped with a warning
  rather than aborting the whole run. Pass a custom comma-separated list to
  narrow the sweep, e.g. `--img-sizes 320,640,960`.
- **`--num-classes`**: defaults to `15` (both models ship pretrained on COCO's
  80/90 classes). The detection head's classification conv is reinitialized
  to this shape at export time — the pretrained backbone/neck weights are
  kept, only the head is affected — so this changes head compute cost without
  needing a custom-trained checkpoint. Detection *output* is meaningless with
  a randomly-initialized head either way; this is a latency-only benchmark.

## 5. Expected runtime (Pi 4, Path A, from a clean venv)

These are rough order-of-magnitude estimates, not measured — actual timing
depends on your SD card/storage speed, network, and how many resolutions are
in your `--img-sizes` sweep (7 by default).

| Step | Expected time |
|---|---|
| `pip install torch torchvision` (CPU wheels) | 3–8 min |
| YOLOX repo clone + `requirements.txt` install | 5–15 min |
| YOLOX checkpoint download (~8–40 MB, once, reused across resolutions) | <1 min |
| YOLOX ONNX export (per resolution) | <1 min each, ~×7 for the default sweep |
| `pip install rfdetr[onnx]` | 3–10 min |
| RF-DETR checkpoint auto-download (once) + ONNX export (per resolution) | 1–3 min each, ~×7 for the default sweep |
| Benchmark loop (both models × 7 resolutions, defaults: 10 warmup + 30 timed runs each) | 5–30+ min (RF-DETR is expected to be the slower one per-run, and larger resolutions scale up latency substantially) |
| **Total, Path A** | **~30–90 min** for the default 7-resolution sweep, mostly dependency installation + the resolution loop |
| **Total, Path B benchmark-only phase on the Pi** | **~10–40 min** for the default 7-resolution sweep |

Narrow `--img-sizes` to fewer resolutions (e.g. `--img-sizes 416,640,896`) to
cut this down for a quicker first pass.

## 6. Interpreting the output

Everything lands in `--output-dir` (default `./bench_output/`):

- **`results.md`** — an "FPS by resolution" pivot table (quick read of the
  speed/resolution tradeoff curve per model), followed by the full comparison
  table: one row per model × resolution, with avg/min/max/std latency, FPS,
  peak RSS, CPU%, ONNX file size, and how many timed runs completed. A
  model that failed at any stage/resolution shows `FAILED` with its error
  message under the "Errors" section below the table — this is expected
  behavior, not a bug, if e.g. RF-DETR export hits the resolution constraint
  at some size.
- **`per_run_timings.csv`** — one row per timed inference
  (`model, img_size, run_index, latency_ms`). Use this to sanity-check for
  outliers (e.g. a single very slow first run suggesting warmup wasn't
  sufficient, or a rising trend suggesting thermal throttling — see §4).
- **`results.json`** — the same data as `results.md`, machine-readable, handy
  if you want to paste it back for the synthesis step instead of/alongside
  the markdown table.
- **`onnx/`** — the exported `.onnx` files, one subdirectory per resolution
  (e.g. `onnx/640/yolox_nano.onnx`, `onnx/640/inference_model.onnx`); useful
  to inspect file size directly or to reuse with `--skip-export` later.

When you're ready for the synthesis step, share `results.md` (or
`results.json`) and `per_run_timings.csv` back — that's what feeds the Pi 5
prediction in `pi4_vs_pi5_notes.md`'s §5.

## 7. Troubleshooting quick-reference

| Symptom | Likely cause / fix |
|---|---|
| `pip install torch` hangs or takes >30 min | It's building from source — you're likely on 32-bit OS or an unsupported Python version. Switch to 64-bit Raspberry Pi OS and/or use the `--index-url https://download.pytorch.org/whl/cpu` command from §2. |
| YOLOX `requirements.txt` install fails on `pycocotools` | Needs a C compiler: `sudo apt install -y build-essential python3-dev`. |
| RF-DETR export fails mentioning "divisible by" | See §4's resolution-sweep note; `--img-sizes` entries that aren't multiples of 32 are auto-dropped, so this should only happen with a hand-rolled non-multiple-of-32 value. |
| `onnxruntime` import fails or is very slow | Check you have a native `aarch64` wheel (§4). Reinstall in a clean venv if unsure. |
| Whole script exits 1 with both models `FAILED` | Check `bench_output/results.md`'s "Errors" section and the console log above it — every failure is logged with an actionable message rather than a bare traceback. Re-run with `-v` for full debug output. |
| Latency numbers look implausibly fast/slow | Check `platform.machine()` printed at the top of the log is `aarch64` (not running under emulation), and check `vcgencmd measure_temp` for throttling. |
