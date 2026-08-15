# cuda2metal

## About

cuda2metal converts a defined subset of CUDA C GPU kernels into Metal Shading
Language (MSL), so kernels written for NVIDIA GPUs (like the ones in
Karpathy's [llm.c](https://github.com/karpathy/llm.c)) can run on Apple
Silicon. Every claim it makes about a kernel "working" is backed by actually
compiling the output with Apple's own Metal compiler — not just a
plausible-looking text rewrite.

## The problem this solves

CUDA and Metal are two different languages for writing the same kind of
program: code that runs on a GPU, with thousands of threads executing the same
function in parallel. NVIDIA GPUs run CUDA. Apple GPUs (in every M-series Mac,
iPad, and iPhone) run Metal instead — CUDA code simply does not run on Apple
hardware, full stop. If you have a CUDA kernel and want it to run on a Mac, someone
has to rewrite it in Metal's language by hand.

The good news: most of that rewriting is *mechanical*, not creative. CUDA and
Metal kernels solve the same problem — "which thread am I, and what's my
piece of the data?" — with different keyword names for the same concepts.
cuda2metal automates that mechanical part for a well-defined subset of CUDA,
and refuses to touch anything outside that subset rather than guessing.

## Why llm.c specifically

[llm.c](https://github.com/karpathy/llm.c) is Andrej Karpathy's project that
trains GPT-2 from scratch with no PyTorch — it has a pure CPU (C) training
path and a CUDA GPU training path, but nothing for Apple Silicon. That gap —
a real project with real kernels, CPU and NVIDIA-GPU only — is exactly what
motivated this tool: making it mechanical to carry individual `__global__`
kernels from a project like that over to a Mac's GPU.

## The core idea: CUDA and Metal name the same things differently

Both languages give every GPU thread a way to figure out "which thread am I."
CUDA splits threads into a grid of *blocks*; Metal splits them into a grid of
*threadgroups*. Same concept, different names:

| Concept | CUDA | Metal (MSL) |
|---|---|---|
| My index within the whole grid | `blockIdx.x * blockDim.x + threadIdx.x` | `thread_position_in_grid` |
| Which block/threadgroup am I in | `blockIdx.x` | `threadgroup_position_in_grid` |
| My index within my block/threadgroup | `threadIdx.x` | `thread_position_in_threadgroup` |
| How many threads per block/threadgroup | `blockDim.x` | `threads_per_threadgroup` |
| How many blocks/threadgroups total | `gridDim.x` | `threadgroups_per_grid` |
| Fast memory shared by one block's threads | `__shared__` | `threadgroup` |
| "Wait here until every thread in my block/threadgroup catches up" | `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Fast/approximate math function names | `fmaxf`, `expf`, `tanhf`, ... | `fmax`, `exp`, `tanh`, ... (no trailing `f`) |

cuda2metal reads a CUDA kernel, recognizes these patterns, and rewrites them
to their Metal names — plus a few structural changes MSL requires that CUDA
doesn't (explained below with a real example).

## A worked example

Input (a real kernel, `examples/gelu_forward_kernel1.cu`, trimmed from llm.c):

```c
__global__ void gelu_forward_kernel1(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float xi = inp[i];
        out[i] = 0.5f * xi * (1.0f + tanhf(0.7978845608f * xi));
    }
}
```

Run `python3 cuda2metal.py examples/gelu_forward_kernel1.cu`, and you get:

```metal
kernel void gelu_forward_kernel1(
    device float* out [[buffer(0)]],
    device const float* inp [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
){
    int i = gid.x;
    if (i < N) {
        float xi = inp[i];
        out[i] = 0.5f * xi * (1.0f + tanh(0.7978845608f * xi));
    }
}
```

What changed, and why each change had to happen:

1. **`__global__ void` → `kernel void`** — that's just Metal's keyword for "this
   function is a GPU entry point," same role as CUDA's `__global__`.
2. **Every parameter gets `[[buffer(N)]]`** — Metal has no separate "just pass
   this argument" mechanism like CUDA does; every kernel argument, including
   plain numbers, has to be told which numbered buffer slot the host code will
   bind it to. `N` (the loop bound, not a pointer) becomes `constant int&
   N [[buffer(2)]]` — passed by reference into constant (read-only) memory,
   MSL's way of passing a scalar.
3. **Pointers get `device` or `device const`** — Metal requires every pointer
   to say *which kind* of GPU memory it points into (this repo's own
   `metal/smith_waterman.metal` does the same thing). CUDA doesn't require
   this because it only really has one kind of GPU memory to worry about
   for basic pointers.
4. **`blockIdx.x * blockDim.x + threadIdx.x` → a single `gid.x` read** — this
   three-part expression is CUDA's standard idiom for "my absolute position in
   the whole grid of threads." Metal computes that directly and hands it to
   you as one value, `thread_position_in_grid`, injected as an extra function
   parameter (`uint3 gid [[thread_position_in_grid]]`) rather than assembled
   from three separate builtins.
5. **`tanhf` → `tanh`** — pure renaming, both do the same thing.

None of this changes what the kernel *computes* — it's the same GELU formula,
same thread-to-data mapping, just spelled in Metal's vocabulary. That's the
whole job of this tool: do the renaming/restructuring correctly and
automatically, and refuse to touch the small number of cases (below) where a
correct 1:1 translation isn't actually possible.

## What it explicitly refuses to translate

Some CUDA features have no direct Metal equivalent — translating them "as if"
they did would produce code that compiles but computes the wrong answer, which
is worse than not translating at all. cuda2metal detects these and stops with
an error instead of guessing:

- **Warp-level intrinsics** (`__shfl*`, `__ballot*`, `__reduce_*`) — Metal's
  `simd_*` functions cover similar ground but with different semantics per
  operation; each one needs a human to check it's actually equivalent.
- **`__device__` helper functions, templates, cooperative groups, dynamic
  parallelism** — bigger structural CUDA features, out of scope for a
  kernel-body translator.
- **Custom packed/vector types** like llm.c's `Packed128`/`x128` (used for
  loading 4 bf16 values as one vectorized instruction) — these encode
  NVIDIA-specific memory-access tricks with no Metal equivalent; they need a
  hand port, not a rename.

If your kernel uses any of these, you'll get a clear error naming the
unsupported construct — not silently wrong Metal code.

## Two more things it handles (beyond the example above)

- **Shared memory**, both fixed-size (`__shared__ float buf[256];`) and
  dynamic (`extern __shared__ float buf[];`) become `threadgroup` memory —
  the dynamic case becomes an extra kernel parameter, and the *host* code
  (not this tool) has to tell Metal how big that buffer is at dispatch time
  via `setThreadgroupMemoryLength(size, index)`.
- **`double` → `float`** — Apple GPUs have no double-precision hardware at
  all, so this isn't a stylistic choice, it's a real drop in numeric
  precision. Every place this happens gets a `// cuda2metal warning:` comment
  written directly into the generated file, so it's impossible to miss.

## Usage

```sh
python3 cuda2metal.py path/to/kernel.cu > kernel.metal
xcrun metal -c kernel.metal -o kernel.air   # sanity-compile; needs macOS + Xcode's Metal toolchain
```

This tool only translates the kernel itself. You still write the host-side
Metal API code by hand (creating buffers, encoding the dispatch, etc.) — the
same way `src/metal_runtime.cpp` does it elsewhere in this repo. Bind buffers
in the same order the original CUDA parameters were declared (`buffer(0)`,
`buffer(1)`, ...).

## Proof it actually works: the tests compile real kernels

`examples/gelu_forward_kernel1.cu` and `examples/softmax_forward_kernel2.cu`
are real kernels taken from llm.c's `dev/cuda/` teaching kernels — one
elementwise (no shared memory), one a shared-memory max-reduction (exercises
`__shared__`, `__syncthreads()`, and the `device` address-space propagation
rule). `tests/test_translate.py` doesn't just diff against a saved "golden"
output file — a golden file could itself be subtly wrong. Instead it
regenerates both `.metal` files from the original `.cu` source and hands them
to `xcrun metal`, Apple's real compiler, on every run:

```sh
python3 tests/test_translate.py
```

```
PASS  gelu_forward_kernel1.cu -> compiles as valid MSL
PASS  softmax_forward_kernel2.cu -> compiles as valid MSL

2/2 examples translate and compile.
```

If cuda2metal is ever changed in a way that breaks the generated Metal code,
this test fails with the compiler's actual error message, not a silent diff.

## Relationship to the rest of this repo

This is a separate concern from metalsw's Smith-Waterman aligner and its
publication — it lives in this directory rather than a new repo purely for
convenience (reusing the same Mac's Metal toolchain for verification). It
shares no code with `src/`/`metal/`, and isn't part of `metalsw_gpu`'s build
or any claim made in `PAPER.md`.
