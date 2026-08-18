# RF-DETR Nano vs YOLOX: CPU-only edge detector benchmark

Evaluating a YOLOX replacement (unmaintained since 2022) for CPU-only object
detection on a Raspberry Pi. Production target is a Pi 5; only a Pi 4 is
available for benchmarking right now.

## Files

- **[`pi4_vs_pi5_notes.md`](pi4_vs_pi5_notes.md)** — hardware research: what
  actually differs between Pi 4 (Cortex-A72) and Pi 5 (Cortex-A76) at the
  microarchitecture/memory level, and why RF-DETR (transformer-heavy) and
  YOLOX (conv-heavy) are not expected to scale by the same multiplier between
  the two boards.
- **[`benchmark.py`](benchmark.py)** — the benchmark script. Exports official
  pretrained RF-DETR Nano and YOLOX-Nano/Tiny checkpoints to ONNX and measures
  CPU-only ONNX Runtime latency/FPS/memory/CPU%/file size on a fixed set of
  test images. **Not yet run** — this needs to execute on the actual Pi 4.
- **[`SETUP.md`](SETUP.md)** — exact install commands, Pi-specific caveats,
  expected runtime, and how to interpret `benchmark.py`'s output files.
