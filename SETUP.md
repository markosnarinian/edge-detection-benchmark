# Running the benchmark

The benchmark targets 64-bit Raspberry Pi OS on a Pi 4 or Pi 5. It measures
raw batch-1 CPU inference after warmup; model loading, preprocessing, export,
and postprocessing are outside the timed interval.

## Test matrix

Defaults test:

- RF-DETR: `nano,small,medium,large`
- YOLOX: `nano,tiny,s,m,l,x,darknet53`
- YOLOX runtimes: `pytorch,onnxruntime,openvino,ncnn`
- RF-DETR runtimes: `pytorch,onnxruntime,openvino,tflite,executorch`
  (`tflite` is FP32; `executorch` uses the FP32 XNNPACK backend)
- resolutions: `256,320,384,416,448,512,576,640,704,768,832,896,960,1024`
- 10 warmups and 30 timed inferences per cell

Every size is divisible by 32, as required by these detector graphs. The full
matrix is 672 benchmark cells and is deliberately exhaustive; larger variants
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

Static deployment graphs repeat model weights at every resolution, so a full
export can consume tens of gigabytes. Keep `results/artifacts` on spacious
storage, or run smaller groups in separate output directories.

The scope is detection models available without an account. RF-DETR Plus
XLarge/2XLarge are excluded because they require the separately licensed Plus
package and a Roboflow account. Segmentation and keypoint variants solve
different tasks and are not comparable detector replacements.

The runtime set was audited against both upstream repositories and Pi ARM64
availability. TensorRT is excluded because it requires an NVIDIA GPU; CoreML
requires Apple platforms; and QNN targets Qualcomm hardware. MegEngine is an
alternate, poorly maintained YOLOX implementation without a credible current
Pi packaging path, while nebullvm is abandoned. ncnn is included through its
official ARM64 Python bindings and current pnnx converter. RF-DETR TFLite and
ExecuTorch/XNNPACK are included but marked experimental because upstream's
converters and Linux ARM64 packaging remain version-sensitive.
OpenVINO supports ARM64 CPUs, but wheel/OS support varies by release and does
not cover every Raspberry Pi OS/Python combination; an unavailable installation
is reported as a failed runtime cell rather than silently omitted.

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
python3 -m pip install ncnn pnnx
```

The script installs the required RF-DETR extras and cloned YOLOX package when
export or native PyTorch is requested. TFLite export currently needs Python
3.12 and `rfdetr[tflite]`; ExecuTorch needs a Torch/ExecuTorch ABI-compatible
`rfdetr[executorch]` environment. Those dependency sets may conflict, so it can
be necessary to export TFLite and ExecuTorch in separate virtual environments
and copy both verified artifact directories to the Pi. Exporting artifacts on
a compatible faster machine is strongly recommended.

## Export elsewhere, benchmark on the Pi

The interactive wizard is the easiest way to keep both machines on the same
matrix and isolate version-sensitive exporters:

```bash
# On the export machine
python3 wizard.py export

# Copy the artifact directory it prints, then on the Pi
python3 wizard.py pi
```

The export directory contains `benchmark-plan.json`. Transfer the whole
directory, including that file and all `.export.json` hash manifests. The Pi
stage reads the plan, creates an isolated environment, offers to install the
selected runtime packages, then launches the crash-resumable benchmark. Use
`python3 wizard.py --help` for dry-run and non-interactive options.

The equivalent manual process follows.

On the export machine:

```bash
python3 benchmark.py --export-only
scp -r results/artifacts pi@<pi-ip>:~/edge-benchmark/artifacts
```

On the Pi, install only the selected runtime packages:

```bash
python3 -m pip install onnxruntime openvino ncnn psutil pillow numpy
python3 benchmark.py --skip-export --artifact-dir ./artifacts \
  --runtimes onnxruntime,openvino,ncnn
```

Native PyTorch runs still need the YOLOX/RF-DETR packages and checkpoints.
For pre-exported RF-DETR TFLite, install `ai-edge-litert` when an ARM64 wheel
is available, or `tflite-runtime`; the benchmark also falls back to TensorFlow's
interpreter. ExecuTorch requires a Python runtime built for the Pi with the same
ABI-compatible Torch version used during export. If those experimental packages
are unavailable, their cells are reported as failures without stopping stable
runtime measurements.

## Crash safety and resume behavior

The benchmark applies the useful low-complexity resilience measures:

1. Downloads use a `.partial` file, `fsync`, and atomic rename. Interrupted
   downloads are never mistaken for valid checkpoints.
2. Runtime exports are written to a temporary file/directory. A SHA-256
   manifest is written atomically only after every component is present; for
   ncnn, the manifest commits the `.param` and `.bin` as one verified pair.
3. Existing runtime artifacts are reused only after their configuration,
   component sizes, and SHA-256 hashes validate. Manifests also record exporter
   versions. This applies to `--skip-export`; use `--allow-unsigned-artifacts`
   only to import legacy artifacts deliberately. Their current content is still
   hashed into the resume identity.
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

Resume matching includes the exact supported job list, class count, ONNX opset,
thread and run counts, image source, source/runtime/exporter versions, and every
artifact content identity. Replacing a graph or runtime therefore starts a new
run instead of mixing incompatible measurements. Because the same directory
holds one checkpoint, use a separate `--output-dir` for matrices you want to
retain.

## Output

The default ignored `results/` directory contains:

- `index.html` — self-contained human-readable report
- `results.json` — machine-readable summary and metadata
- `per_run_timings.csv` — every timed inference
- `checkpoint.json` — durable resume state
- `artifacts/<model>/<resolution>/onnx/` — ONNX shared by ONNX Runtime/OpenVINO
- `artifacts/<model>/<resolution>/{ncnn,tflite,executorch}/` — native static
  graphs for the applicable model, with hash manifests beside every artifact

Failed cells remain visible with their error instead of aborting the matrix.
The command exits successfully when at least one cell succeeds.

## Measurement cautions

- Use active cooling and check for thermal throttling during long sweeps.
- Close unrelated processes and use the same runtime versions across devices.
- `--threads` defaults to all detected cores and is applied to PyTorch, ONNX
  Runtime, OpenVINO, ncnn, and TFLite. ExecuTorch/XNNPACK currently has no
  stable per-program Python thread-setting API, so its report rows explicitly
  show `runtime-default` rather than claiming controlled thread parity.
- CPU percentage is process-wide and may approach 400% on a four-core Pi.
- The default 15-class head is reinitialized from COCO checkpoints. This keeps
  head compute representative of the intended deployment, but predictions are
  not meaningful; this is a performance benchmark, not an accuracy benchmark.
- Compare runtimes within the same model/resolution. Artifact size is omitted
  only for native PyTorch, which uses the source checkpoint.
