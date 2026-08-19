# RF-DETR vs YOLOX on Raspberry Pi CPUs

This project benchmarks open RF-DETR and YOLOX object detectors on Raspberry
Pi 4 and Pi 5 CPUs. It measures warmed-up, batch-1 model inference across every
supported model/runtime combination and 14 practical square resolutions from
256 to 1024 pixels.

The default matrix contains **672 benchmark cells**:

| Family | Variants | Raspberry Pi CPU runtimes |
| --- | --- | --- |
| YOLOX | Nano, Tiny, S, M, L, X, Darknet53 | PyTorch, ONNX Runtime, OpenVINO, ncnn |
| RF-DETR | Nano, Small, Medium, Large | PyTorch, ONNX Runtime, OpenVINO, TFLite, ExecuTorch/XNNPACK |

TFLite and ExecuTorch support is experimental because their exporters and
Linux ARM64 packages are version-sensitive. An unavailable runtime is recorded
as a failed cell instead of stopping the rest of the benchmark.

## Recommended workflow

Export deployment graphs on a faster machine, then benchmark the exported
artifacts on the Pi. The interactive wizard keeps both machines on exactly the
same model, runtime, resolution, class-count, and ONNX-opset configuration.

Requirements:

- A clone of this repository on the export machine and the Pi
- 64-bit Raspberry Pi OS on the Pi
- Python 3.10–3.12, Git, internet access, and substantial free disk space
- Python 3.12 on the export machine when TFLite is selected

Static graphs repeat model weights at every resolution. A full export can use
tens of gigabytes, so choose an artifact location with enough free space.

### 1. Export on the faster machine

```bash
python3 wizard.py export
```

The wizard:

1. Asks which variants, runtimes, and resolutions to include.
2. Creates reusable isolated environments under `work/wizard-envs/`.
3. Separates stable, TFLite, and ExecuTorch exports to avoid dependency
   conflicts.
4. Verifies every artifact and SHA-256 manifest.
5. Writes `benchmark-plan.json` into the artifact directory.

It prints an `scp` example when export finishes. Copy the **entire** artifact
directory, including `benchmark-plan.json`, `.export.json` manifests, and ncnn
`.param`/`.bin` pairs, to the Pi.

### 2. Run on the Pi

From this repository on the Pi:

```bash
python3 wizard.py pi
```

Enter the transferred artifact directory when prompted. The wizard validates
the transfer, creates an isolated Pi environment, offers to install the chosen
runtimes, and starts `benchmark.py --skip-export` with the exported plan.

To provide the plan directly:

```bash
python3 wizard.py pi --plan ./artifacts/benchmark-plan.json
```

Interrupting the run is safe. Run the Pi wizard again with the same settings to
resume from the last completed model/runtime/resolution cell.

### Preview without installing or exporting

```bash
python3 wizard.py export --dry-run
python3 wizard.py pi --plan ./artifacts/benchmark-plan.json --dry-run
```

Use `python3 wizard.py --help` for all wizard options.

## Run the benchmark directly

The wizard is recommended, but `benchmark.py` can be called directly. For
example, this runs a smaller representative matrix:

```bash
python3 benchmark.py \
  --yolox-variants nano,tiny \
  --rfdetr-variants nano,small \
  --runtimes onnxruntime,ncnn \
  --img-sizes 320,416,512,640
```

See [`SETUP.md`](SETUP.md) for manual package installation, export-on-one-machine
instructions, all command-line behavior, runtime caveats, and the complete
default matrix.

## Results and resume safety

Generated files are placed under the ignored `results/` directory:

- `index.html` — self-contained report for reviewing results and failures
- `results.json` — machine-readable metadata and summaries
- `per_run_timings.csv` — every timed inference
- `checkpoint.json` — atomically updated resume state
- `artifacts/` — reusable static graphs and SHA-256 manifests

Downloads, checkpoints, and final reports are replaced atomically. Exports are
built in temporary locations and become reusable only after a verified manifest
is committed; the manifest treats ncnn `.param` and `.bin` files as one pair.
Resume identity includes the exact job matrix, artifact hashes, opset, source,
exporter and runtime versions, thread count, and run counts. Replacing an
artifact or runtime therefore starts a fresh run instead of mixing incompatible
measurements.

## What the numbers mean

The benchmark reports raw model execution after warmup. Model loading, image
preprocessing, export, and detection postprocessing are outside the timed
interval. Compare runtimes only within the same model and resolution.

The default 15-class detection heads are reinitialized from COCO checkpoints.
This preserves representative head compute, but predictions are not meaningful:
this is a **performance benchmark**, not an accuracy evaluation.

RF-DETR Plus XLarge/2XLarge are excluded because they require a separately
licensed package and account. Segmentation and keypoint models are excluded
because they perform different tasks. GPU- or platform-specific runtimes such
as TensorRT, CoreML, and QNN are not applicable to Raspberry Pi CPUs.

## Further reading

- [`SETUP.md`](SETUP.md) — detailed setup, runtime availability, manual export,
  crash safety, and measurement cautions
- [`pi4_vs_pi5_notes.md`](pi4_vs_pi5_notes.md) — hardware context for comparing
  Pi 4 and Pi 5 results
