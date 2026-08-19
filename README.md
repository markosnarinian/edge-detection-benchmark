# RF-DETR vs YOLOX: CPU edge-detector benchmark

`benchmark.py` compares all core RF-DETR detection variants (Nano, Small,
Medium, Large) with all seven official YOLOX variants (Nano, Tiny, S, M, L, X,
Darknet53). Every model is measured with the CPU/Linux runtimes both families
support on Raspberry Pi: PyTorch, ONNX Runtime, and OpenVINO for both families;
ncnn for YOLOX; and experimental TFLite and ExecuTorch/XNNPACK for RF-DETR.

The default sweep covers 14 practical square resolutions from 256 to 1024.
Use the variant, runtime, and resolution flags to run a smaller matrix first:

```bash
python3 benchmark.py \
  --yolox-variants nano,tiny \
  --rfdetr-variants nano,small \
  --runtimes onnxruntime,ncnn \
  --img-sizes 320,416,512,640
```

Results are written as an HTML report to `results/index.html`, with JSON, raw
CSV timings, resumable checkpoints, and reusable runtime artifacts alongside it.
The generated `results/` directory is intentionally ignored by Git.

For a guided two-machine workflow, run:

```bash
python3 wizard.py
```

Choose **export** on the faster machine. The wizard creates isolated export
environments, records the chosen matrix in `benchmark-plan.json`, and tells you
which artifact directory to copy. After copying that entire directory, run the
wizard again on the Pi and choose **pi** to install runtimes and start or resume
the benchmark.

See [`SETUP.md`](SETUP.md) for installation, the complete test matrix,
resumability behavior, and recommended staged runs. Hardware context for
interpreting Pi 4/Pi 5 differences is in
[`pi4_vs_pi5_notes.md`](pi4_vs_pi5_notes.md).
