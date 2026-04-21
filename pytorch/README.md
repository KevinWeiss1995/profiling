# profiling-demo

A small, opinionated PyTorch training-step profiling demo. Targets GH200 /
Hopper specifically but runs anywhere with a modern NVIDIA GPU.

The point: show, on a real training loop, how a few lines of Python expand
into hundreds of GPU kernels — and then show three common situations
(bigger batches, bf16 autocast, a slow dataloader) that change what that
picture looks like.

Every scenario is one flag. Every scenario produces a self-contained Nsight
Systems trace with NVTX annotations so the timeline is actually readable.

---

## What you get

Four scenarios, identical code path, just different config:

| scenario      | batch | dtype | data path        | teaches                                          |
| ------------- | ----- | ----- | ---------------- | ------------------------------------------------ |
| `baseline`    | 8     | fp32  | on-device        | `.backward()` expands into hundreds of kernels   |
| `large-batch` | 64    | fp32  | on-device        | launch overhead amortization, SM occupancy       |
| `amp`         | 8     | bf16  | on-device        | Tensor Cores, activation memory bandwidth        |
| `starved`     | 8     | fp32  | slow CPU + copy  | dataloader bottleneck: GPU idle between steps    |

Each run emits, under `out/`:

- `<scenario>.nsys-rep` (if using nsys) — open in `nsys-ui`.
- `<scenario>.trace.json` (if using `torch.profiler`) — drop into
  `chrome://tracing` or `ui.perfetto.dev`.
- `<scenario>.report.json` — machine-readable: median step ms, peak memory,
  kernel count, top-10 kernels by CUDA time.

---

## Install

### Option A — apptainer + NGC container (recommended on HPC)

```bash
# Pull the NGC PyTorch container once (~8-12GB, aarch64 for GH200)
apptainer pull pytorch_25.03.sif docker://nvcr.io/nvidia/pytorch:25.03-py3

# Build a venv INSIDE the container so Python versions match
apptainer exec --nv pytorch_25.03.sif \
    python -m venv --system-site-packages .venv

apptainer exec --nv pytorch_25.03.sif \
    .venv/bin/pip install -e . --no-deps

# Run anything via:
apptainer exec --nv pytorch_25.03.sif \
    bash -c 'source .venv/bin/activate && ./scripts/run_all.sh'
```

The `--system-site-packages` trick lets the venv inherit the container's
torch/nsys while allowing a local editable install of this package on top.

### Option B — native venv

On a workstation with CUDA already installed:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

On aarch64 / GH200 the `--index-url` is critical: `pip`'s default resolver
will otherwise try to build from source or pull a CPU wheel.

Python ≥ 3.10. PyTorch ≥ 2.3.

---

## Quickstart

Run one scenario, no profiler — just get a wall-clock baseline and sanity
check:

```bash
python -m profiling_demo --scenario baseline --profiler none --steps 20
```

Run one scenario with `torch.profiler` (works without Nsight installed):

```bash
python -m profiling_demo --scenario amp --profiler torch --out-dir out
# open out/amp.trace.json in chrome://tracing or ui.perfetto.dev
cat out/amp.report.json
```

Run one scenario under nsys:

```bash
./scripts/run_nsys.sh baseline
# open out/baseline.nsys-rep in nsys-ui
```

Run the whole sweep and print a comparison table:

```bash
./scripts/run_all.sh                  # torch.profiler, fast
PROFILER=nsys ./scripts/run_all.sh    # four nsys traces, slower
PROFILER=none ./scripts/run_all.sh    # just compare wall-clock medians
```

Example summary (actual numbers vary by GPU):

```
scenario     bs  dtype          compiled  step ms  peak MiB  kernels
-----------  --  -------------  --------  -------  --------  -------
baseline      8  fp32           no          18.80     4880       7440
large-batch  64  fp32           no         126.22    29872       7901
amp           8  torch.bfloat16 no          15.61     4158      11471
starved       8  fp32           no         ~40                   7440
```

`starved` lands near `baseline + slow_host_sleep_s` per step (20 ms default).
If yours is much higher, your dataloader is slower than just the sleep —
another lesson hiding in the trace.

---

## What to look for in each trace

See [docs/walkthrough.md](docs/walkthrough.md) for a per-scenario guide: which
NVTX range to zoom into, which kernel names to grep for, and what the
takeaway is supposed to be.

Short version:

- **baseline**: dense forest of short kernels. Count them with
  `nsys stats --report cuda_gpu_kern_sum out/baseline.nsys-rep`. Expect
  hundreds per step — that's the core "high-level code, many kernels" lesson.
- **large-batch**: same forest, each tree taller. Launch overhead becomes
  a smaller fraction of total time; memory use scales roughly linearly.
- **amp**: GEMM kernel names change to Tensor Core variants (look for
  `sm90_xmma_gemm_bf16`, `hmma`, or similar); activations shrink.
- **starved**: huge gaps between kernel bursts labelled `dataloader_sleep`
  on the NVTX row. GPU idle = wasted money. Fix is almost always "more
  dataloader workers + `pin_memory=True`".

---

## Why NVTX matters here

Without NVTX, an nsys trace of a PyTorch training step is just an
undifferentiated blur of hundreds of kernels. With the ranges in
[`src/profiling_demo/step.py`](src/profiling_demo/step.py) and
[`src/profiling_demo/cli.py`](src/profiling_demo/cli.py) you get a clean
decomposition per iteration:

```
iter/0
  data                 (on-device: tiny; starved: mostly sleep)
    dataloader_sleep   (starved only)
    host_collate       (starved only)
    h2d_copy           (starved only)
  forward
  loss
  backward
  optimizer
```

That alone is usually enough to identify whether you're bottlenecked on
forward, backward, optimizer, or the data pipeline.

We also bound the profiler capture with `cudaProfilerStart/Stop`, so the
recorded window is exactly the profiled steps. Warmup is excluded. Traces
stay small; the interesting part is front-and-center when you open the file.

---

## GH200 aside

Everything above works on any modern NVIDIA GPU. A few Grace + Hopper
specifics worth knowing once you start digging:

- **NVLink-C2C** gives Grace ↔ Hopper roughly an order of magnitude more
  bandwidth than PCIe Gen5. Pinned host memory matters less than on a
  PCIe-attached H100: non-pinned `.to(device, non_blocking=True)` is still
  fast. The `starved` scenario pins anyway, for portability.
- **TF32 is on by default** in modern PyTorch, so the "fp32" baseline
  is really using Hopper's TF32 Tensor Cores (kernel names like
  `sm90_xmma_gemm_f32f32_tf32f32_...`). The `amp` scenario's win is
  therefore over TF32, not true-fp32.
- **SDPA** on fp32 causal dispatches to a CUTLASS mem-efficient kernel
  (`fmha_cutlassF_f32_aligned_64x64_rf_sm80`), not flash-3. Flash-3's
  Hopper path targets fp16/bf16; switch to AMP to see it activate.

---

## Extending

The code is deliberately small. Useful extensions:

1. **`torch.compile` + CUDA graphs**: the plumbing exists
   (`--compile` reaches through `build_for_scenario`), but it needs a
   working Triton on your container. Usually one `LD_LIBRARY_PATH` tweak
   away. Would show dramatic kernel fusion.
2. **Starved variants**: try `--slow-host-sleep-s 0.005` vs `0.050` and
   watch the timeline change shape. Good exercise for understanding the
   crossover between GPU-bound and CPU-bound.
3. **Multi-worker `DataLoader`**: replace `starved`'s serial iterator with
   a real `DataLoader(num_workers=4, pin_memory=True)` and show the gaps
   closing without touching any GPU code.
4. **Nsight Compute**: pick the dominant GEMM from the nsys trace and
   profile it with `ncu --kernel-name <name>` to see SM occupancy and
   memory-pipeline breakdowns.

---

## Layout

```
src/profiling_demo/
  model.py          decoder-only transformer block stack
  data.py           synthetic on-GPU data + slow-host starved path
  step.py           NVTX-annotated train_step
  profile_utils.py  NVTX / cudaProfilerStart+Stop / torch.profiler helpers
  scenarios.py      the four scenario configs
  cli.py            argparse entry point

scripts/
  run_nsys.sh       one scenario under nsys
  run_all.sh        all four + comparison table

docs/
  walkthrough.md    per-scenario nsys reading guide
```
