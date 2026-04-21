# Nsys walkthrough

This is the "what to look for" companion to the four scenarios. Each section
assumes you've opened `out/<scenario>.nsys-rep` in `nsys-ui` and scrolled to
the first `iter/0` NVTX range.

If you've never used nsys before: three rows matter most.

1. **CUDA HW &rarr; Kernels** — what actually ran on the GPU, when.
2. **CUDA API** — host-side `cudaLaunchKernel`, memcpy, sync calls.
3. **NVTX** — our `forward` / `backward` / `optimizer` ranges.

Stacking those three rows is the whole game: you see high-level Python
annotations align to a dense stripe of kernel launches align to the actual
kernel execution timeline. One `.backward()` call really does turn into
hundreds of kernels.

> Tip: right-click a kernel → "Show in Events View" → "Correlated CUDA API"
> to jump straight from a device-side kernel to the exact `cudaLaunchKernel`
> that triggered it.

---

## 1. `baseline` — fp32, bs=8

**What you're looking at:** a dense stream of short kernels. This is the
canonical "high-level code to kernel" picture.

**Look for:**

- In the **NVTX** row, expand an `iter/N` range. You'll see the phase
  breakdown: `forward` (biggest), `loss` (small), `backward` (~2x forward),
  `optimizer` (Adam: many tiny kernels).
- In the **Kernels** row during `forward`, find `sm90_xmma_gemm_...` (Hopper
  Tensor Core GEMMs via cuBLAS) interleaved with `vectorized_elementwise_kernel`
  (LayerNorm, residual add, GELU).
- In the **CUDA API** row during `backward`, you'll see a visibly denser
  cluster of `cudaLaunchKernel` calls than during forward. That's autograd
  dispatching grad kernels for every op.
- During `optimizer`, you'll see the fused AdamW kernel(s). Without
  `fused=True` you'd see 4+ tiny kernels per parameter group; with it you
  see ~1 big one.

**Teaching moment:** count the kernels in a single iteration via

```
nsys stats --report cuda_gpu_kern_sum out/baseline.nsys-rep | head -n 20
```

This is usually in the many-hundreds range. That's what "a few lines of
Python" expanded to.

---

## 2. `large-batch` — fp32, bs=64

**What you're looking at:** the exact same kernel set as baseline, but each
kernel is longer and the gaps between them shrink as a fraction of total time.

**Look for:**

- Compare `forward` duration vs baseline. Not 8x longer (despite 8x more
  tokens) — GEMMs get better SM occupancy at larger batch sizes.
- Launch overhead (`cudaLaunchKernel` width on the API row) is roughly the
  same per call, but it's now a smaller fraction of each kernel's wall time.
- `peak_memory_mib` in the report JSON jumps roughly linearly with batch
  size; that's your cost for the throughput win.

**Teaching moment:** compare median step times.

```
jq .median_step_ms out/baseline.report.json out/large-batch.report.json
```

Tokens/sec = `bs * seq_len / (median_step_ms / 1000)`. Large-batch should
give ~3–6x the throughput of baseline on Hopper-class GPUs.

---

## 3. `amp` — bf16 autocast, bs=8

**What you're looking at:** same overall shape as baseline, but GEMMs now
run in bf16 on Tensor Cores and the backward pass visibly shrinks.

**Look for:**

- Kernel names in `forward` change from `...gemm_f32f32_f32f32...` /
  `sm90_xmma_gemm_...` to bf16 variants (look for `bf16`, `s16816`, or
  `hmma` in the name depending on the cuBLAS heuristic picked).
- LayerNorm, GELU, softmax stay in fp32 under autocast's default rules —
  you'll still see those vectorized elementwise kernels unchanged.
- The `loss` phase stays fp32 (we explicitly `.float()` the logits before
  `cross_entropy`) — visible as an fp32-styled kernel.
- Peak memory drops meaningfully vs baseline (activations are bf16).

**Teaching moment:** this is a memory-bandwidth win, not (only) a compute
win. On Hopper the Tensor Cores already crush fp32 matmul; bf16's bigger
advantage for a step this size is that activations are half the size, so
the backward pass reads less memory.

```
nsys stats --report cuda_gpu_mem_size_sum out/baseline.nsys-rep
nsys stats --report cuda_gpu_mem_size_sum out/amp.nsys-rep
```

---

## 4. `compiled` — torch.compile, bs=8

**What you're looking at:** dramatically fewer, longer kernels. On the API
row, the forest of `cudaLaunchKernel` calls is replaced (or mostly replaced)
by a single `cudaGraphLaunch` per step.

**Look for:**

- Kernel names containing `triton_` — these are fused kernels that
  torch.compile's Inductor backend generated. One Triton kernel often
  replaces 5–10 eager ones (e.g. LayerNorm + residual + next op's pre-GEMM
  reshape all fuse).
- Under NVTX `iter/N`, the total wall time per step drops; more of it is
  now "useful" GEMM time.
- On the **CUDA API** row you should see `cudaGraphLaunch` calls in place
  of streams of `cudaLaunchKernel`. That's `mode="reduce-overhead"` doing
  its job — Inductor captured each forward/backward into a CUDA graph.
- First few profiled steps may still be re-compiling / re-capturing if
  shapes shifted — that's why `--warmup` defaults to ≥3 for this scenario.

**Teaching moment:** compare total kernel count from the torch.profiler
summary. Compiled typically shows a 3–10x reduction in total kernel launches
per step for this model. That reduction is what closes the Python overhead
gap and is what makes small-batch inference / training viable on Hopper.

```
jq .torch_profiler.total_cuda_kernels out/baseline.report.json \
                                      out/compiled.report.json
```

---

## Reading a trace you've never seen before: a mini workflow

1. Zoom into one `iter/N` NVTX range (right-click → "Zoom to").
2. Flatten the timeline: show only Kernels + CUDA API + NVTX rows.
3. Find the longest kernel in the step → that's usually your bottleneck
   candidate. Hover: is it GEMM? elementwise? reduction?
4. Pick a suspicious gap on the Kernels row, look at the CUDA API row
   directly above — is the host too slow (Python / launch overhead) or is
   there a sync point (`cudaStreamSynchronize`, `cudaMemcpy`)?
5. Compare to the adjacent scenario's trace side-by-side (`nsys-ui` supports
   two windows). The diff is usually the story.
