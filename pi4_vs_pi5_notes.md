# Raspberry Pi 4 vs Pi 5: Hardware Differences Relevant to CPU-Only Detector Inference

Scope: this note is about the specific hardware deltas between Pi 4 (BCM2711) and Pi 5
(BCM2712) that could plausibly cause **RF-DETR Nano** (transformer/attention-heavy,
DINOv2-style ViT backbone) and **YOLOX-Nano/Tiny** (conv-heavy, anchor-free CNN) to
scale by *different* multipliers when moving from Pi 4 to Pi 5 — not just "Pi 5 is
faster." Everything below is desk research; none of it is measured on this project's
actual checkpoints. Treat the "expected differential" callouts as hypotheses to check
against Part 3's real numbers, not conclusions.

## 1. CPU microarchitecture: Cortex-A72 (Pi 4) vs Cortex-A76 (Pi 5)

| | Pi 4 / BCM2711 | Pi 5 / BCM2712 |
|---|---|---|
| Core | 4x Cortex-A72 | 4x Cortex-A76 |
| ISA | ARMv8.0-A | ARMv8.2-A |
| Clock | 1.5 GHz (1.8 GHz on later Pi 4 Rev 1.4/1.5 boards) | 2.4 GHz |
| Decode width | 3-wide, in-order fetch/decode front end | 4-wide decode, deeper/wider out-of-order core |
| NEON/FP pipelines | Narrower SIMD throughput, effectively 1 FP/NEON pipe's worth of useful issue rate | 2x NEON execution pipelines — roughly double SIMD issue throughput per clock vs A72 |
| Int8 dot product (SDOT/UDOT) | **Not supported** — ARMv8.0 baseline only | **Supported** (ARMv8.2 optional extension, implemented in A76) |
| FP16 vector arithmetic | Not supported natively | Supported |
| L1 cache | 48 KB I / 32 KB D per core | Larger/improved (per-core, exact sizes vary by implementation but generally larger than A72) |
| L2 cache | 1 MB **shared** across the 4-core cluster | 512 KB **private per-core** |
| L3 cache | None | 2 MB shared |

Net effect cited by Raspberry Pi Ltd. and reviewers: roughly **2–3x** uplift for
general CPU-bound workloads, driven by clock speed *and* per-clock efficiency (IPC),
not clock speed alone — Cortex-A76 does more work per cycle even before you account
for the 60% clock increase.

**Why this matters unevenly for the two model types:**

- The SDOT/UDOT int8 dot-product instructions are the most architecturally
  interesting delta for ML workloads, but **they only help if the ONNX Runtime
  execution path is actually issuing int8 dot-product ops** — i.e., if the model is
  int8-quantized. Both models in this benchmark are exported and run as plain FP32
  ONNX graphs (per the task spec), so this specific ISA advantage is **not expected
  to show up in Part 2's fp32 numbers on either model.** It becomes directly relevant
  only if a later phase quantizes to int8 (QOperator/QDQ) for further Pi
  optimization — at which point it could disproportionately help *both* models, but
  especially the attention/GEMM-heavy parts of RF-DETR, since transformer inference
  is dominated by large GEMMs that map cleanly onto dot-product hardware, whereas
  YOLOX's depthwise/grouped convolutions get comparatively less benefit from GEMM-style
  int8 acceleration.
- The doubled NEON issue throughput on A76 helps **both** architectures' FP32 compute
  (convolutions and matmuls both bottleneck on MAC throughput), so this alone
  shouldn't shift the relative ranking much — call it a shared multiplier.
- The cache reorganization (1 MB shared L2 on Pi 4 → 512 KB private L2 + 2 MB shared
  L3 on Pi 5) plus roughly double memory bandwidth (below) is the more likely source
  of a *differential* speedup, because CNNs and transformers have different
  arithmetic intensity profiles (see §3).

Sources:
[Raspberry Pi Documentation – Processors](https://www.raspberrypi.com/documentation/computers/processors.html),
[PiCockpit: Raspberry Pi 4 vs Raspberry Pi 5](https://picockpit.com/raspberry-pi/raspberry-pi-4-vs-raspberry-pi-5/),
[BLIIOT: ARM Cortex-A72 vs Cortex-A76 Processors](https://bliiot.com/info-detail/arm-cortex-a72-vs-cortex-a76-processors),
[WikiChip: Cortex-A76](https://en.wikichip.org/wiki/arm_holdings/microarchitectures/cortex-a76),
[cpu-monkey: Pi5 BCM2712 vs Pi4 BCM2711](https://www.cpu-monkey.com/en/compare_cpu-raspberry_pi_5_b_broadcom_bcm2712-vs-raspberry_pi_4_b_broadcom_bcm2711),
[Raspberry Pi docs: bcm2712.adoc](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/processors/bcm2712.adoc).

## 2. Memory: LPDDR4 (Pi 4) vs LPDDR4X (Pi 5)

| | Pi 4 | Pi 5 |
|---|---|---|
| Memory type | LPDDR4, rated up to 3200 MT/s (real configured/effective rate is lower in practice — some sources measure effective throughput closer to ~2000 MT/s once controller/refresh overhead is accounted for) | LPDDR4X-4267 |
| Bus width | 32-bit single channel | 32-bit single channel (same width — the gain is from higher transfer rate + a redesigned memory subsystem/interconnect, not a wider bus) |
| Reported real-world bandwidth uplift | baseline | Reviewers report **~2x** effective bandwidth improvement Pi 4→Pi 5; theoretical peak numbers as high as ~17–34 GB/s appear depending on measurement methodology, so treat absolute figures loosely and rely on the ~2x relative multiplier as the more defensible number |

**Why this matters unevenly:** this is the single most important documented reason to
expect RF-DETR and YOLOX to *not* scale by the same multiplier.

- CNNs like YOLOX have high **data reuse** per weight (a convolution kernel is
  reapplied across many spatial positions), which gives them relatively high
  arithmetic intensity (FLOPs per byte moved) and makes them more **compute-bound**
  than **memory-bound** on a system this small — i.e., they're already reasonably
  efficient on Pi 4's narrower memory subsystem, so they have comparatively less to
  gain from a memory bandwidth increase alone.
- Vision transformer backbones (RF-DETR Nano's DINOv2-derived backbone, and the
  attention/FFN blocks generally) have **lower arithmetic intensity** — self-attention
  and the large linear projections around it move large activation tensors through
  memory relative to the compute performed on them. A 2026 mobile-inference latency
  study states this directly: *"the arithmetic intensity of ViTs is generally lower
  than that of CNNs, indicating a greater tendency for memory-bound performance on
  mobile devices"* (arXiv:2510.25166, "A Study on Inference Latency for Vision
  Transformers on Mobile Devices"). That's exactly the profile that benefits
  disproportionately from a memory-bandwidth-bound bottleneck being loosened — which
  is what Pi 5's LPDDR4X + larger/faster cache hierarchy does.
- Practical implication: **RF-DETR Nano is the model most likely to show a
  larger-than-2–3x speedup going Pi4→Pi5**, precisely because it's the one whose
  bottleneck (memory-bound attention/FFN activations) is the one Pi 5 relieves the
  most. YOLOX, being smaller and more compute/cache-friendly already, is more likely
  to track closer to the "generic" 2–3x CPU uplift figure. This is a hypothesis to
  check against Part 3's real data, not a guarantee — RF-DETR's absolute latency
  could still remain much worse than YOLOX's even after a larger relative gain,
  since the starting gap (per the user's own prior on RF-DETR's CPU-ARM speed) is
  large.

Sources:
[Zbotic: Raspberry Pi 5 Benchmarks](https://zbotic.in/raspberry-pi-5-benchmarks-real-performance-numbers-vs-pi-4/),
[Elektor: Pi 5 vs Pi 4 comparison](https://www.elektormagazine.com/news/raspberry-pi-5-vs-raspberry-pi-4-a-comparison),
[arXiv:2510.25166 — A Study on Inference Latency for Vision Transformers on Mobile Devices](https://arxiv.org/html/2510.25166),
[TinyWeights: Running LLMs on Raspberry Pi 5](https://tinyweights.dev/posts/run-llms-raspberry-pi-5/) (independent corroboration that Pi 5 CPU inference for attention-heavy models is characterized as memory-bandwidth-bound).

## 3. Compute-bound vs memory-bound: why "2–3x faster" is not one number

The commonly cited "Pi 5 is 2–3x faster than Pi 4" figure comes from general
CPU/IO-bound benchmarks (compression, compilation, generic scalar/SIMD workloads),
not from detector-specific measurements. For this benchmark specifically:

- If YOLOX-Nano/Tiny is **compute-bound** on both boards (likely, given its small
  size — 0.91M params / 1.08 GFLOPs for Nano, 5.06M params / 6.45 GFLOPs for Tiny at
  416x416, per the official YOLOX model zoo), its Pi4→Pi5 speedup should track close
  to the clock-speed x IPC improvement (~1.6x clock x some IPC factor), i.e., broadly
  in the commonly cited 2–3x range, and should **not** be dominated by the memory
  bandwidth change.
- If RF-DETR Nano is **partially memory-bound** on Pi 4 (plausible given its
  attention layers and given the user's own observation that RF-DETR is "known to be
  slow on CPU-only ARM" — slow-on-ARM symptoms for transformer models are often a
  memory-bandwidth/cache-miss story more than a raw-FLOPs story on cores this small),
  its Pi4→Pi5 speedup could plausibly **exceed** the generic 2–3x figure, because it's
  relieved on two axes simultaneously: faster core (compute) *and* roughly 2x memory
  bandwidth + a real L3 cache that Pi 4 doesn't have at all.
- Caveat in the other direction: RF-DETR's ONNX export (per Roboflow's own docs) is
  a plain dense graph run through onnxruntime's generic CPU EP/MLAS kernels, which
  are primarily convolution/GEMM-optimized; attention-specific ops (softmax,
  reshape/transpose-heavy patterns) may not be as well-kernel-optimized on ARM as
  YOLOX's convolutions are. This could suppress RF-DETR's Pi 5 gains regardless of
  raw memory bandwidth, i.e., the software/kernel-optimization gap could offset some
  of the hardware-side relative advantage. There is no public benchmark data
  specifically measuring RF-DETR (or a directly comparable ONNX ViT detector) on Pi 4
  vs Pi 5 — this section is reasoned inference from more general ViT-vs-CNN mobile
  inference literature and generic Pi 4 vs Pi 5 benchmarks, not a documented,
  model-specific result. Flag this explicitly for the reader: **confidence is
  moderate-low on the magnitude of the differential, moderate-high on the direction**
  (RF-DETR should close some relative ground on Pi 5, but is very unlikely to
  overtake YOLOX in absolute terms at 640x640 given the starting gap).

## 4. Other confounds worth flagging (lower documentation confidence)

- **Thermal/power throttling**: Pi 5 runs hotter under sustained multi-core load and
  is more commonly paired with active cooling; Pi 4 more often runs passively cooled
  at a lower power ceiling. A 20-run steady-state benchmark is short enough that this
  is unlikely to matter much on either board, but a colder/better-cooled Pi 5 could
  read faster than a thermally-constrained one, independent of the CPU architecture
  discussion above. Not modeled further here.
- **OS/kernel differences**: both boards typically run current Raspberry Pi OS
  (Bookworm+), so scheduler/toolchain differences should be minimal, but confirm both
  benchmark runs use the same OS image/Python/onnxruntime build — do not compare a
  Pi 4 run on an older OS against a Pi 5 run on a newer one.
- **onnxruntime ARM build quality**: whether the `onnxruntime` wheel installed on
  each board picks up NEON-optimized MLAS kernels (vs a slower generic fallback) can
  matter more than the hardware delta being discussed here. Confirm via `python -c
  "import onnxruntime; print(onnxruntime.get_device())"` and check that the wheel is
  a native `aarch64` build, not running under emulation.

## 5. Bottom line for Part 3 synthesis

When the real Pi 4 numbers come back:

1. Compute YOLOX's and RF-DETR's Pi4 latency ratio (RF-DETR ms / YOLOX ms).
2. The generic hardware-only prediction is: both models get faster by roughly the
   same 2–3x "generic CPU uplift" factor on Pi 5, in which case the *ratio* between
   them should stay roughly constant.
3. The differential-scaling hypothesis above says RF-DETR should close *some* of the
   relative gap (i.e., the Pi5 ratio should be smaller than the Pi4 ratio) because
   it's more memory-bandwidth-bound than YOLOX, but is unlikely to close all of it
   given YOLOX's much smaller compute footprint and the kernel-optimization caveat
   in §3.
4. If the Pi 4 numbers show RF-DETR is memory-bound in practice (e.g., latency
   doesn't scale linearly with thread count beyond 2-3 threads, or scales worse than
   YOLOX's does when you vary `intra_op_num_threads` — worth capturing in Part 2's
   run if time permits) that would be direct local evidence supporting a larger
   differential gain on Pi 5, strengthening the prediction above.
