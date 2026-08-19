# Running the benchmark

The benchmark targets 64-bit Raspberry Pi OS on a Pi 4 or Pi 5. It measures
raw batch-1 CPU inference after warmup; model loading, preprocessing, export,
and postprocessing are outside the timed interval.

## Test matrix

Defaults test:

- RF-DETR: `nano,small,medium,large`
- YOLOX: `nano,tiny,s,m,l,x,darknet53`
- runtimes: `pytorch,onnxruntime,openvino`
- resolutions: `256,320,384,416,448,512,576,640,704,768,832,896,960,1024`
- 10 warmups and 30 timed inferences per cell

Every size is divisible by 32, as required by these detector graphs. The full
matrix is 462 benchmark cells and is deliberately exhaustive; larger variants
at larger resolutions may be too slow or memory-heavy for a Pi 4. Start with a
small representative matrix, then expand it unattended:

```bash
python3 benchmark.py \
  --yolox-variants nano,tiny,s \
  --rfdetr-variants nano,small \
  --runtimes onnxruntime \
  --img-sizes 320,416,512,640,768

# Full matrix (potentially many hours/days on a Pi 4)
python3 benchmark.py
```

Static ONNX graphs repeat model weights at every resolution, so a full export
can consume tens of gigabytes. Keep `results/onnx` on spacious storage, or run
smaller variant/resolution groups in separate output directories.

The scope is detection models available without an account. RF-DETR Plus
XLarge/2XLarge are excluded because they require the separately licensed Plus
package and a Roboflow account. Segmentation and keypoint variants solve
different tasks and are not comparable detector replacements. TensorRT is
excluded because the target has no NVIDIA GPU. YOLOX ncnn is excluded because
upstream provides no Python runtime and its conversion requires a manual graph
rewrite. RF-DETR's experimental TFLite/ExecuTorch formats are not shared by
YOLOX; CoreML does not run on Raspberry Pi. PyTorch, ONNX Runtime, and OpenVINO
are therefore the complete comparable CPU/Linux Python runtime set.

## Install

Use 64-bit Raspberry Pi OS and Python 3.10–3.12. At least 4 GB RAM and active
cooling are recommended; large model/resolution combinations may need swap.

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install onnx onnxsim onnxruntime openvino psutil pillow numpy
```

The script installs `rfdetr[onnx]` and the cloned YOLOX package when export or
native PyTorch is requested. This is convenient but heavy on a Pi. Exporting
ONNX on a faster machine is recommended.

## Export elsewhere, benchmark on the Pi

On the export machine:

```bash
python3 benchmark.py --export-only
scp -r results/onnx pi@<pi-ip>:~/edge-benchmark/onnx
```

On the Pi, ONNX Runtime and OpenVINO need no training-side packages:

```bash
python3 -m pip install onnxruntime openvino psutil pillow numpy
python3 benchmark.py --skip-export --onnx-dir ./onnx \
  --runtimes onnxruntime,openvino
```

Native PyTorch runs still need the YOLOX/RF-DETR packages and checkpoints.

## Crash safety and resume behavior

The benchmark applies the useful low-complexity resilience measures:

1. Downloads use a `.partial` file, `fsync`, and atomic rename. Interrupted
   downloads are never mistaken for valid checkpoints.
2. ONNX exports are written to a temporary file/directory and atomically moved
   into place only after a non-empty artifact exists.
3. Existing non-empty ONNX artifacts are reused, so export resumes at the first
   missing model/resolution. A sidecar signature prevents reuse after changing
   the class count or ONNX opset.
4. `results/checkpoint.json` is atomically replaced after every completed
   model/runtime/resolution cell. A rerun with the same matrix skips completed
   cells, including recorded failures. Use `--no-resume` after fixing a failed
   dependency or when fresh measurements are required.
5. Final HTML, JSON, and CSV files are also atomically replaced, avoiding
   half-written reports after power loss.

These measures preserve all completed work after Ctrl-C, reboot, OOM kill, or
a native-runtime process crash. The inference currently runs in the main
process: a segfault ends that invocation, but restarting resumes from the last
completed cell. Running every cell in a child process could contain segfaults
and enforce memory/time limits, but would add substantial process startup and
model reload overhead to a latency benchmark. It is not justified unless real
Pi runs show repeated native crashes. No software measure protects against
filesystem/media corruption; copy `results/` off-device for long runs.

Resume matching includes models, runtimes, resolutions, class count, thread
count, run counts, and image source. A changed configuration starts a new run
instead of mixing incomparable measurements. Because the same directory holds
one checkpoint, use a separate `--output-dir` for matrices you want to retain.

## Output

The default ignored `results/` directory contains:

- `index.html` — self-contained human-readable report
- `results.json` — machine-readable summary and metadata
- `per_run_timings.csv` — every timed inference
- `checkpoint.json` — durable resume state
- `onnx/<model>/<resolution>/<model>.onnx` — reusable static-shape exports,
  with `.export.json` configuration sidecars

Failed cells remain visible with their error instead of aborting the matrix.
The command exits successfully when at least one cell succeeds.

## Measurement cautions

- Use active cooling and check for thermal throttling during long sweeps.
- Close unrelated processes and use the same runtime versions across devices.
- `--threads` defaults to all detected cores and is applied to each runtime.
- CPU percentage is process-wide and may approach 400% on a four-core Pi.
- The default 15-class head is reinitialized from COCO checkpoints. This keeps
  head compute representative of the intended deployment, but predictions are
  not meaningful; this is a performance benchmark, not an accuracy benchmark.
- Compare runtimes within the same model/resolution. ONNX size is shown only
  for ONNX Runtime/OpenVINO rows because native PyTorch uses checkpoints.
