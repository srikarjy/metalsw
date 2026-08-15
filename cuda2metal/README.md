# cuda2metal

A small, honest source-to-source translator from a subset of CUDA C kernels to
Metal Shading Language (MSL). Not a general CUDA-to-Metal compiler — a real but
narrowly scoped tool, verified by actually compiling its output with `xcrun
metal` (Apple's real Metal compiler), not just eyeballing the text.

Motivated by [llm.c](https://github.com/karpathy/llm.c) (Karpathy) having both a
plain-CPU-C and a CUDA training path with no Metal path — this exists to make
porting individual `__global__` kernels from something like llm.c to Apple
Silicon mechanical instead of manual, for the parts of a kernel that are pure
boilerplate translation (thread indexing, shared memory, barriers, math
intrinsic names).

## What it handles

- `__global__` kernel functions (only — host launch code, `<<<...>>>`, `cudaMalloc`,
  etc. are untouched; write that by hand against Metal's API, same as
  `src/metal_runtime.cpp` does elsewhere in this repo).
- Thread-indexing builtins, per axis: `blockIdx`/`blockDim`/`threadIdx`/`gridDim`,
  including the common `blockIdx.x * blockDim.x + threadIdx.x` global-index idiom
  (collapsed to a single `thread_position_in_grid` read).
- `__shared__` (fixed-size array) and `extern __shared__` (dynamic-size) memory
  -> `threadgroup` parameters.
- `__syncthreads()` -> `threadgroup_barrier(mem_flags::mem_threadgroup)`.
- A small math-intrinsic rename table: `fmaxf/fminf/expf/tanhf/sqrtf/coshf/sinhf/
  powf/logf/M_PI`.
- Local pointer variables aliased from a `device`-space kernel parameter get the
  `device` address-space qualifier propagated onto them (MSL requires this
  explicitly; C++/CUDA infers it and the naive rename would fail to compile
  otherwise — this was caught by actually compiling the softmax example, not by
  inspection).
- `double` -> `float`, because MSL has no double precision. This is a real
  precision change, not a rename — every occurrence is flagged with a
  `// cuda2metal warning:` comment in the generated file, not silently applied.

## What it explicitly does NOT handle

Raises a `TranslateError` and refuses to emit code, rather than guessing:

- Warp-level intrinsics (`__shfl*`, `__ballot*`, `__reduce_*`) — Metal's `simd_*`
  equivalents have different semantics per-op and need a case-by-case port.
- `__device__` helper functions, templates, cooperative groups, dynamic
  parallelism.
- Custom packed/vector types (e.g. llm.c's `Packed128`/`x128`, used for
  bf16-packed vectorized loads) — these encode CUDA-specific tricks with no 1:1
  MSL equivalent and need a hand port.

If you feed it a kernel using any of the above, it will fail loudly at
translation time, not produce something that looks right and silently computes
the wrong answer.

## Usage

```sh
python3 cuda2metal.py path/to/kernel.cu > kernel.metal
xcrun metal -c kernel.metal -o kernel.air   # sanity-compile, macOS + Metal toolchain only
```

Buffers must be bound on the host side in the same order the original CUDA
parameters were declared (`buffer(0)`, `buffer(1)`, ...); any `extern __shared__`
parameter becomes a `threadgroup` buffer at the next free index, and the host
must call `setThreadgroupMemoryLength(size, index)` for it before dispatch.

## Examples

`examples/gelu_forward_kernel1.cu` and `examples/softmax_forward_kernel2.cu` are
real kernels lifted from `llm.c`'s `dev/cuda/` teaching kernels (chosen because
they're self-contained, one elementwise and one shared-memory-reduction — no
`Packed128` vector types, which are out of scope, see above). `tests/test_translate.py`
regenerates both from source and compiles the output with `xcrun metal`,
proving the translated MSL is real, valid Metal code — not just re-checking
against a stored golden file that could itself be wrong.

Run the tests:

```sh
python3 tests/test_translate.py
```

## Relationship to the rest of this repo

This is a separate concern from metalsw's Smith-Waterman aligner and its
publication — it lives in this directory rather than a new repo purely for
convenience of reusing the Mac's Metal toolchain for verification. It shares no
code with `src/`/`metal/` and isn't part of `metalsw_gpu`'s build or the paper's
claims.
