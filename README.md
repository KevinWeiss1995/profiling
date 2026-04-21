# profiling-demo

A small, opinionated PyTorch training-step profiling demo. Targets GH200 /
Hopper specifically but runs anywhere with a modern NVIDIA GPU.

The point: show, on a real training loop, how a few lines of Python expand
into hundreds of GPU kernels — and then show four common optimizations
(bigger batches, bf16 autocast, `torch.compile`) tightening that picture.

Every scenario is one flag. Every scenario produces a self-contained Nsight
Systems trace with NVTX annotations so the timeline is actually readable.

---

## What you get

Four scenarios, identical code path, just different config:

| scenario      | batch | dtype | compiled | teaches                                          |
| ------------- | ----- | ----- | -------- | ------------------------------------------------ |
| `baseline`    | 8     | fp32  | no       | "`.backward()` = hundreds of kernels"            |
| `large-batch` | 64    | fp32  | no       | launch overhead amortization, SM occupancy       |
| `amp`         | 8     | bf16  | no       | Tensor Cores, activation memory bandwidth        |
| `compiled`    | 8     | fp32  | yes      | Inductor kernel fusion + CUDA graphs             |

Each run emits, under `out/`:

- `<scenario>.nsys-rep` (if using nsys) — open in `nsys-ui`.
- `<scenario>.trace.json` (if using `torch.profiler`) — drop into
  `chrome://tracing` or `ui.perfetto.dev`.
- `<scenario>.report.json` — machine-readable: median step ms, peak memory,
  kernel count, top-10 kernels by CUDA time.
- `<scenario>.summary.json` — torch.profiler summary only.

---

## Install

On your GH200 / CUDA box:

```bash
# pick the wheel index matching your CUDA version; examples:
# CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6:
pip install torch --index-url https://download.pytorch.org/whl/cu126

pip install -e .
```

For Nsight Systems, install the `nsight-systems` package from NVIDIA (most
CUDA dev containers already have it; `nsys --version` to check).

Python ≥ 3.10. PyTorch ≥ 2.3. Tested against the 25.x PyTorch container
lineage on sm_90 (H100) and sm_90a (GH200 / H200).

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

Example summary row output:

```
scenario     bs  dtype          compiled  step ms  peak MiB  kernels
-----------  --  -------------  --------  -------  --------  -------
baseline      8  fp32           no          42.10     1832       ~450
large-batch  64  fp32           no         118.40     8104       ~450
amp           8  torch.bfloat16 no          21.70      974       ~450
compiled      8  fp32           yes         18.20     1820        ~40
```

(Actual numbers will vary by GPU; those are illustrative.)

---

## What to look for in each trace

See [docs/walkthrough.md](docs/walkthrough.md) for a per-scenario guide: which
NVTX range to zoom into, which kernel names to grep for, and what the
takeaway is supposed to be.

Short version:

- **baseline**: dense forest of short kernels. Count them with
  `nsys stats --report cuda_gpu_kern_sum out/baseline.nsys-rep`.
- **large-batch**: same forest, each tree taller. Launch overhead becomes
  a smaller fraction of total time.
- **amp**: GEMM kernel names change to bf16 Tensor Core variants; backward
  shrinks noticeably (half the activation bytes).
- **compiled**: the forest collapses. Look for `triton_` kernels and
  `cudaGraphLaunch` replacing streams of `cudaLaunchKernel`.

---

## Why NVTX matters here

Without NVTX, an nsys trace of a PyTorch training step is just an
undifferentiated blur of 500 kernels. With the ranges in
[`src/profiling_demo/step.py`](src/profiling_demo/step.py) you get a clean
decomposition per iteration:

```
iter/0
  forward
  loss
  backward
  optimizer
```

That alone is often enough to identify whether you're bottlenecked on
forward, backward, or optimizer — and whether the host is keeping up with
the device.

We also bound the profiler capture with `cudaProfilerStart/Stop`, so the
recorded window is exactly the profiled steps. Warmup (including compile /
graph capture) is excluded. Traces stay small; the interesting part is
front and center when you open the file.

---

## GH200 aside

Everything above works on any modern NVIDIA GPU. A few Grace + Hopper
specifics worth knowing once you start digging:

- **NVLink-C2C** gives Grace ↔ Hopper roughly an order of magnitude more
  bandwidth than PCIe Gen5. Pinned host memory matters less than on a
  PCIe-attached H100: non-pinned `.to(device, non_blocking=True)` is still
  fast. The `--slow-host-data` hook in `data.py` is wired up for the day
  you want to contrast this with a PCIe machine.
- **`mode="reduce-overhead"`** on torch.compile pairs especially well with
  Hopper. The CPU-side launch path is already fast on Grace, but CUDA
  graphs remove it entirely. You'll see `cudaGraphLaunch` completely
  replace the per-kernel launch stream.
- **Unified memory** (`cudaMallocManaged`) is actually usable for
  activation spill / offload on GH200 in a way it isn't on x86+PCIe. Not
  demoed here — it's a good follow-up: add a scenario that intentionally
  overcommits GPU memory and lets UM handle it.

---

## Extending

The code is deliberately small. Useful extensions:

1. **Starved-GPU demo**: flip `--slow-host-data` and add a sleep in
   [`src/profiling_demo/data.py`](src/profiling_demo/data.py)'s
   `host_batch_iterator`. You'll see the GPU go idle between steps — the
   classic "DataLoader bottleneck" picture.
2. **CUDA graphs without torch.compile**: capture the training step into
   a `torch.cuda.CUDAGraph` manually. Good demo of what
   `mode="reduce-overhead"` does under the hood.
3. **Nsight Compute**: pick one kernel from the nsys trace (usually the
   dominant GEMM) and profile it with `ncu --kernel-name <name>` to see
   SM utilization, memory throughput, and cycle breakdowns.
4. **FP8**: on Hopper, swap autocast for `transformer_engine`'s fp8 recipe.
   The GEMM kernels change names again and memory drops further.

---

## Layout

```
src/profiling_demo/
  model.py          decoder-only transformer block stack
  data.py           synthetic on-GPU data; optional slow-host hook
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
