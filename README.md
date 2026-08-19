# RF-DETR vs YOLOX on Raspberry Pi CPUs

This project benchmarks open RF-DETR and YOLOX object detectors on Raspberry
Pi 4 and Pi 5 CPUs. It measures warmed-up, batch-1 inference across every
supported model/runtime combination and 14 practical square resolutions.

| Family | Variants | Raspberry Pi CPU runtimes |
| --- | --- | --- |
| YOLOX | Nano, Tiny, S, M, L, X, Darknet53 | PyTorch, ONNX Runtime, OpenVINO, ncnn |
| RF-DETR | Nano, Small, Medium, Large | PyTorch, ONNX Runtime, OpenVINO, TFLite, ExecuTorch/XNNPACK |

The default matrix is **672 cells**: 11 models, each applicable runtime, and 14
resolutions from 256 to 1024. TFLite and ExecuTorch are experimental because
their exporters and Linux ARM64 packages are version-sensitive. An unavailable
runtime is recorded as a failed cell instead of stopping the rest of the run.

## The intended workflow

Do not build every deployment graph on the Pi. Use two stages:

1. **Export elsewhere:** generate and verify static deployment artifacts on a
   faster x86-64 or ARM64 machine with ample memory and storage.
2. **Run on the Pi:** copy those artifacts to the Pi and benchmark them without
   installing the training/export toolchains there.

Native PyTorch is the exception: it has no exported graph, so PyTorch cells
still install the model packages and download checkpoints on the Pi.

The wizard is the recommended interface. It records the selected matrix in
`benchmark-plan.json`, uses isolated export environments, verifies the transfer,
and recreates the same matrix on the Pi.

## Prerequisites and dependency constraints

Both machines need a clone of this repository, Python, Git, [uv](https://docs.astral.sh/uv/),
internet access, and enough free storage. The Pi should run 64-bit Raspberry Pi
OS. At least 4 GB RAM, active cooling, and swap for the largest cells are
recommended.

Getting compatible packages is part of the benchmark. The wizard creates
virtual environments and installs packages with `uv`, but it cannot create an ARM64 wheel that a
project does not publish or make incompatible package versions coexist.

The dependency groups in `pyproject.toml` cover the stable stage prerequisites:

```bash
uv sync --group export  # artifact export dependencies
uv sync --group run     # Raspberry Pi runtime dependencies
uv sync --group all     # both stages in one environment
```

The wizard installs only the selected packages into its isolated environments,
so running these commands first is optional. PyTorch, TFLite, ExecuTorch, and
the model packages remain selection-specific because their available wheels and
compatible versions vary by machine.

- Use Python 3.10–3.12 generally.
- RF-DETR TFLite export currently requires Python 3.12 and `rfdetr[tflite]`.
- ExecuTorch requires mutually compatible PyTorch, ExecuTorch, and RF-DETR
  builds. A `.pte` exported with one ABI may not load with another.
- OpenVINO ARM64 wheel support varies by OpenVINO release, Raspberry Pi OS, and
  Python version.
- ncnn export needs `pnnx` and `ncnn`; the Pi needs a compatible ncnn Python
  runtime.
- Stable ONNX/ncnn, TFLite, and ExecuTorch exports use separate virtual
  environments because their dependency constraints may conflict.

Start with ONNX Runtime, OpenVINO, and ncnn where applicable. Once those work,
add PyTorch and the experimental TFLite/ExecuTorch groups. Package failures stay
visible in the final report, so a missing experimental wheel does not erase
successful stable-runtime measurements.

Static graphs repeat model weights at every resolution. A full export can use
tens of gigabytes; put the artifact directory on spacious storage.

## Stage 1: export on the faster machine

Run from this repository on the export machine:

```bash
uv run python wizard.py export
```

The wizard asks for models, runtimes, resolutions, class count, and ONNX opset.
It then:

1. Creates reusable environments under `work/wizard-envs/`.
2. Separates stable, TFLite, and ExecuTorch exporters.
3. Runs `benchmark.py --export-only` for each applicable group.
4. Verifies every artifact and SHA-256 manifest.
5. Writes `benchmark-plan.json` beside the artifacts.

If an export is interrupted, rerun the wizard with the same choices. Valid
artifacts are reused, and export continues with missing cells.

Preview all setup and export commands without executing them:

```bash
uv run python wizard.py export --dry-run
```

### Transfer the artifacts

The wizard prints an `scp` example. Copy the **entire** artifact directory,
including:

- `benchmark-plan.json`
- every `.export.json` manifest
- ONNX, TFLite, and ExecuTorch files
- both files in every ncnn `.param`/`.bin` pair

For example:

```bash
scp -r results/artifacts pi@<pi-address>:~/edge-benchmark/artifacts
```

Do not copy only the model files: the Pi stage rejects unsigned, incomplete, or
hash-mismatched artifacts by default.

## Stage 2: run on the Pi

From this repository on the Pi:

```bash
uv run python wizard.py pi
```

Enter the transferred artifact directory. The wizard:

1. Reads and validates `benchmark-plan.json`.
2. Hash-checks every transferred artifact before setup.
3. Creates a reusable Pi environment under `work/wizard-envs/pi-runtime/`.
4. Offers to install the selected runtime packages.
5. Runs `benchmark.py --skip-export` with the exact exported matrix.

You can provide the plan directly:

```bash
uv run python wizard.py pi --plan ./artifacts/benchmark-plan.json
```

Preview the Pi setup and benchmark command without executing it:

```bash
uv run python wizard.py pi \
  --plan ./artifacts/benchmark-plan.json \
  --dry-run
```

If the Pi loses power, runs out of memory, or is interrupted, run the Pi wizard
again with the same choices. The benchmark resumes after the last completed
model/runtime/resolution cell.

## Manual two-stage workflow

The wizard is preferred, but the equivalent commands are useful for debugging.

### Export stable artifacts

```bash
uv venv --python 3.12 export-stable
uv pip install --python export-stable/bin/python --group export
uv pip install --python export-stable/bin/python torch torchvision \
  --default-index https://download.pytorch.org/whl/cpu

export-stable/bin/python benchmark.py \
  --export-only \
  --runtimes onnxruntime,openvino,ncnn \
  --artifact-dir ./results/artifacts
```

Export RF-DETR TFLite and ExecuTorch in separate environments, writing to the
same artifact root:

```bash
# TFLite: Python 3.12 environment
uv venv --python 3.12 export-tflite
uv pip install --python export-tflite/bin/python numpy pillow psutil
export-tflite/bin/python benchmark.py --export-only --skip-yolox \
  --runtimes tflite --artifact-dir ./results/artifacts

# ExecuTorch: separate ABI-compatible environment
uv venv --python 3.12 export-executorch
uv pip install --python export-executorch/bin/python numpy pillow psutil
export-executorch/bin/python benchmark.py --export-only --skip-yolox \
  --runtimes executorch --artifact-dir ./results/artifacts
```

`benchmark.py` installs the selected RF-DETR extras and the cloned YOLOX package
when export requires them. A failed setup/export is logged without deleting
artifacts completed by another group.

### Run pre-exported stable artifacts on the Pi

```bash
uv venv --python 3.12 pi-benchmark
uv pip install --python pi-benchmark/bin/python --group run

pi-benchmark/bin/python benchmark.py \
  --skip-export \
  --artifact-dir ./artifacts \
  --runtimes onnxruntime,openvino,ncnn
```

For TFLite, install `ai-edge-litert` when an ARM64 wheel is available, otherwise
try `tflite-runtime`; TensorFlow Lite is the final interpreter fallback. For
ExecuTorch, install an ARM64 runtime compatible with the export environment.
For native PyTorch cells, install PyTorch and include `pytorch` in `--runtimes`;
the benchmark then prepares YOLOX/RF-DETR and their checkpoints on the Pi.

```bash
# Add only the optional runtimes selected for this Pi environment.
uv pip install --python pi-benchmark/bin/python ai-edge-litert  # or: tflite-runtime
uv pip install --python pi-benchmark/bin/python executorch
uv pip install --python pi-benchmark/bin/python torch torchvision \
  --default-index https://download.pytorch.org/whl/cpu
```

Run `uv run python benchmark.py --help` for every model, runtime, resolution, output,
thread, and resume option.

## Default matrix and practical subsets

Defaults:

- YOLOX: `nano,tiny,s,m,l,x,darknet53`
- RF-DETR: `nano,small,medium,large`
- YOLOX runtimes: `pytorch,onnxruntime,openvino,ncnn`
- RF-DETR runtimes: `pytorch,onnxruntime,openvino,tflite,executorch`
- Resolutions: `256,320,384,416,448,512,576,640,704,768,832,896,960,1024`
- 10 warmups and 30 timed inferences per cell
- 15 output classes

Every resolution is divisible by 32, as required by both graph families. The
full matrix may take many hours or days on a Pi 4. Start with a smaller sweep:

```bash
uv run python benchmark.py \
  --yolox-variants nano,tiny,s \
  --rfdetr-variants nano,small \
  --runtimes onnxruntime \
  --img-sizes 320,416,512,640,768
```

## Results, failure handling, and resume safety

The ignored `results/` directory contains:

- `index.html` — self-contained report with successful and failed cells
- `results.json` — machine-readable summaries and metadata
- `per_run_timings.csv` — every timed inference
- `checkpoint.json` — durable per-cell resume state
- `artifacts/<model>/<resolution>/onnx/` — ONNX shared by ONNX Runtime/OpenVINO
- `artifacts/<model>/<resolution>/{ncnn,tflite,executorch}/` — native graphs and
  their manifests

The benchmark applies these crash-safety measures:

1. Downloads use a partial file, `fsync`, and atomic rename.
2. Exports are built in temporary locations. A SHA-256 manifest is committed
   only after all components exist; ncnn `.param` and `.bin` are one pair.
3. Existing artifacts are reused only after configuration, size, and hash
   validation. `--skip-export` applies the same checks.
4. `checkpoint.json` is atomically replaced after every completed cell,
   including recorded failures.
5. HTML, JSON, and CSV reports are atomically replaced.

Resume identity includes the exact job matrix, artifact hashes, ONNX opset,
source/exporter/runtime versions, image source, threads, warmups, and run count.
Changing any of these starts a fresh run rather than mixing measurements.

Use `--no-resume` after fixing a failed dependency or when fresh measurements
are required. `--allow-unsigned-artifacts` exists for deliberate legacy imports,
but signed wizard exports are safer. A native-runtime segfault still ends the
current process; restarting resumes from the last completed cell. No software
measure protects against storage corruption, so copy long-running results off
the Pi periodically.

## Interpreting results

The benchmark times raw model execution after warmup. Model loading, image
preprocessing, export, and detection postprocessing are outside the timed
interval. Compare runtimes only within the same model and resolution.

- Use active cooling and check for thermal throttling.
- Close unrelated processes and keep runtime versions consistent across
  devices.
- `--threads` controls PyTorch, ONNX Runtime, OpenVINO, ncnn, and TFLite.
  ExecuTorch currently reports `runtime-default` because its Python API has no
  stable per-program XNNPACK thread control.
- CPU percentage is process-wide and can approach 400% on a four-core Pi.
- Artifact size is omitted only for native PyTorch, which uses checkpoints.

The default 15-class heads are reinitialized from COCO checkpoints. This keeps
head compute representative, but predictions are not meaningful: this is a
**performance benchmark**, not an accuracy evaluation.

## Scope

The benchmark includes detector models available without an account. RF-DETR
Plus XLarge/2XLarge require a separately licensed package and Roboflow account,
so they are excluded. Segmentation and keypoint models perform different tasks.

TensorRT requires an NVIDIA GPU, CoreML requires Apple platforms, and QNN
targets Qualcomm hardware. MegEngine has no credible current Pi packaging path,
and nebullvm is abandoned. These are not relevant Raspberry Pi CPU runtimes.

See [`pi4_vs_pi5_notes.md`](pi4_vs_pi5_notes.md) for hardware context when
comparing Pi 4 and Pi 5 results.
