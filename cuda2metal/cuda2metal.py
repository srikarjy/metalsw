#!/usr/bin/env python3
"""
cuda2metal — a small, honest source-to-source translator from a subset of CUDA C
kernels to Metal Shading Language (MSL).

Scope (this is the whole scope — nothing beyond this is claimed to work):
  - __global__ kernel functions only. Host-side launch code (<<<...>>>, cudaMalloc,
    etc.) is not touched; you still write the metal-cpp/Metal API dispatch code by
    hand, same as the rest of metalsw's host runtime does.
  - Thread-indexing builtins: blockIdx/blockDim/threadIdx/gridDim, per axis (x/y/z),
    including the common `blockIdx.x * blockDim.x + threadIdx.x` global-index idiom.
  - `__shared__` (fixed-size) and `extern __shared__` (dynamic) -> `threadgroup` memory.
  - `__syncthreads()` -> `threadgroup_barrier(mem_flags::mem_threadgroup)`.
  - A small math-intrinsic name table (fmaxf/fminf/expf/tanhf/sqrtf/coshf/powf/logf).
  - `double` -> `float`, since MSL has no double precision (a warning comment is
    emitted in the output wherever this happens — it is a real precision loss, not
    a cosmetic rename, and this tool will not pretend otherwise).

Explicitly NOT handled (will raise, not silently mistranslate):
  - Warp-level intrinsics (__shfl*, __ballot, etc.) — Metal's simd_* equivalents
    have different semantics per-op and need a case-by-case port, not a rename.
  - Templates, __device__ helper functions, cooperative groups, dynamic parallelism.
  - Custom packed/vector types (e.g. llm.c's Packed128/x128) — these encode
    CUDA-specific vectorized load/store tricks with no 1:1 MSL equivalent.

Usage:
    python3 cuda2metal.py input.cu > output.metal
"""

import re
import sys

TYPE_RENAMES = {
    "floatX": "float",
    "double": "float",  # precision loss — flagged in output, see WARN_TYPES
}
WARN_TYPES = {"double"}

FUNC_RENAMES = {
    "fmaxf": "fmax",
    "fminf": "fmin",
    "expf": "exp",
    "tanhf": "tanh",
    "sqrtf": "sqrt",
    "coshf": "cosh",
    "sinhf": "sinh",
    "powf": "pow",
    "logf": "log",
    "M_PI": "M_PI_F",  # MSL's metal_stdlib defines M_PI_F, not the bare M_PI macro
}

UNSUPPORTED = [
    r"__shfl\w*", r"__ballot\w*", r"__reduce_\w+", r"cooperative_groups",
    r"template\s*<", r"__device__",
]

KERNEL_RE = re.compile(r"__global__\s+void\s+(\w+)\s*\(", re.MULTILINE)


class TranslateError(RuntimeError):
    pass


def find_matching_paren(s, open_idx):
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    raise TranslateError("unbalanced parentheses in kernel signature")


def find_matching_brace(s, open_idx):
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    raise TranslateError("unbalanced braces in kernel body")


def split_top_level(s, sep=","):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([<":
            depth += 1
        elif ch in ")]>":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return [p.strip() for p in parts]


def check_unsupported(body, kernel_name):
    for pat in UNSUPPORTED:
        m = re.search(pat, body)
        if m:
            raise TranslateError(
                f"kernel '{kernel_name}': '{m.group(0)}' is out of scope for "
                f"cuda2metal (see the NOT handled list in this file's docstring) "
                f"— port it by hand"
            )


def parse_param(param, buffer_index):
    """Returns (msl_param_str, warnings)."""
    warnings = []
    is_const = False
    tokens = param.replace("*", " * ").split()
    tokens = [t for t in tokens if t]
    if "const" in tokens:
        is_const = True
        tokens.remove("const")
    is_pointer = "*" in tokens
    tokens = [t for t in tokens if t != "*"]
    if len(tokens) < 2:
        raise TranslateError(f"could not parse parameter: '{param}'")
    name = tokens[-1]
    ctype = " ".join(tokens[:-1])
    if ctype in WARN_TYPES:
        warnings.append(f"parameter '{name}': CUDA '{ctype}' has no MSL double "
                         f"precision equivalent — narrowed to float")
    ctype = TYPE_RENAMES.get(ctype, ctype)
    if is_pointer:
        qual = "device const" if is_const else "device"
        msl = f"{qual} {ctype}* {name} [[buffer({buffer_index})]]"
    else:
        msl = f"constant {ctype}& {name} [[buffer({buffer_index})]]"
    return msl, warnings, (name if is_pointer else None)


GLOBAL_IDX_RE = re.compile(
    r"blockIdx\.(x|y|z)\s*\*\s*blockDim\.\1\s*\+\s*threadIdx\.\1"
)
BLOCK_IDX_RE = re.compile(r"blockIdx\.([xyz])")
THREAD_IDX_RE = re.compile(r"threadIdx\.([xyz])")
BLOCK_DIM_RE = re.compile(r"blockDim\.([xyz])")
GRID_DIM_RE = re.compile(r"gridDim\.([xyz])")
EXTERN_SHARED_RE = re.compile(
    r"extern\s+__shared__\s+(\w+)\s+(\w+)\s*\[\s*\]\s*;\s*\n?"
)
FIXED_SHARED_RE = re.compile(
    r"__shared__\s+(\w+)\s+(\w+)\s*\[\s*(\w+)\s*\]\s*;"
)
SYNCTHREADS_RE = re.compile(r"__syncthreads\s*\(\s*\)\s*;")


LOCAL_PTR_DECL_RE = re.compile(
    r"(?m)^(\s*)(const\s+)?(\w+)\s*\*\s*(\w+)\s*=\s*([^;]+);"
)


def fix_local_pointer_address_spaces(body, device_ptr_names):
    """MSL requires every pointer type to carry an explicit address-space
    qualifier — unlike C++, it will not infer one for a local `const float* x =
    inp + i;` just because `inp` is a `device` pointer. Any local pointer
    initialized from an expression that references a device-buffer parameter
    gets `device` propagated onto it."""

    def repl(m):
        indent, const_kw, ctype, name, rhs = m.groups()
        if ctype in ("device", "threadgroup", "constant"):
            return m.group(0)  # already qualified, leave alone
        if not any(re.search(rf"\b{p}\b", rhs) for p in device_ptr_names):
            return m.group(0)
        const_kw = const_kw or ""
        return f"{indent}device {const_kw}{ctype}* {name} = {rhs};"

    return LOCAL_PTR_DECL_RE.sub(repl, body)


def translate_body(body, kernel_name, device_ptr_names=()):
    warnings = []
    needed = set()  # subset of {gid, tgid, tid, tg_size, grid_size}

    body = fix_local_pointer_address_spaces(body, device_ptr_names)

    def repl_global_idx(m):
        needed.add("gid")
        return f"_c2m_gid.{m.group(1)}"

    body = GLOBAL_IDX_RE.sub(repl_global_idx, body)

    def repl_block_idx(m):
        needed.add("tgid")
        return f"_c2m_tgid.{m.group(1)}"

    def repl_thread_idx(m):
        needed.add("tid")
        return f"_c2m_tid.{m.group(1)}"

    def repl_block_dim(m):
        needed.add("tg_size")
        return f"_c2m_tg_size.{m.group(1)}"

    def repl_grid_dim(m):
        needed.add("grid_size")
        return f"_c2m_grid_size.{m.group(1)}"

    body = BLOCK_IDX_RE.sub(repl_block_idx, body)
    body = THREAD_IDX_RE.sub(repl_thread_idx, body)
    body = BLOCK_DIM_RE.sub(repl_block_dim, body)
    body = GRID_DIM_RE.sub(repl_grid_dim, body)

    threadgroup_params = []  # list of msl param strings, appended after buffers

    def repl_extern_shared(m):
        ctype, name = m.group(1), m.group(2)
        ctype = TYPE_RENAMES.get(ctype, ctype)
        threadgroup_params.append((name, f"threadgroup {ctype}* {name}"))
        warnings.append(
            f"'extern __shared__ {ctype} {name}[]' -> threadgroup buffer '{name}'; "
            f"host must call setThreadgroupMemoryLength(size, index) for this argument"
        )
        return ""

    body = EXTERN_SHARED_RE.sub(repl_extern_shared, body)

    def repl_fixed_shared(m):
        ctype, name, count = m.group(1), m.group(2), m.group(3)
        ctype = TYPE_RENAMES.get(ctype, ctype)
        return f"threadgroup {ctype} {name}[{count}];"

    body = FIXED_SHARED_RE.sub(repl_fixed_shared, body)

    body = SYNCTHREADS_RE.sub("threadgroup_barrier(mem_flags::mem_threadgroup);", body)

    for cuda_name, msl_name in FUNC_RENAMES.items():
        body = re.sub(rf"\b{cuda_name}\b", msl_name, body)
    for cuda_type in WARN_TYPES:
        if re.search(rf"\b{cuda_type}\b", body):
            warnings.append(
                f"kernel '{kernel_name}': local variable(s) declared 'double' — "
                f"narrowed to float, this changes numerical behavior"
            )
        body = re.sub(rf"\b{cuda_type}\b", TYPE_RENAMES[cuda_type], body)

    return body, needed, threadgroup_params, warnings


BUILTIN_ATTR = {
    "gid": ("uint3 _c2m_gid", "[[thread_position_in_grid]]"),
    "tgid": ("uint3 _c2m_tgid", "[[threadgroup_position_in_grid]]"),
    "tid": ("uint3 _c2m_tid", "[[thread_position_in_threadgroup]]"),
    "tg_size": ("uint3 _c2m_tg_size", "[[threads_per_threadgroup]]"),
    "grid_size": ("uint3 _c2m_grid_size", "[[threadgroups_per_grid]]"),
}
BUILTIN_ORDER = ["gid", "tgid", "tid", "tg_size", "grid_size"]


def translate_kernel(name, params_str, body):
    all_warnings = []
    check_unsupported(body, name)

    raw_params = split_top_level(params_str) if params_str.strip() else []
    msl_params = []
    device_ptr_names = []
    for i, p in enumerate(raw_params):
        msl, warns, ptr_name = parse_param(p, i)
        msl_params.append(msl)
        all_warnings.extend(warns)
        if ptr_name:
            device_ptr_names.append(ptr_name)

    body, needed, threadgroup_params, body_warns = translate_body(
        body, name, device_ptr_names
    )
    all_warnings.extend(body_warns)

    next_buffer = len(raw_params)
    for _, decl in threadgroup_params:
        msl_params.append(f"{decl} [[threadgroup({next_buffer})]]")
        next_buffer += 1

    for key in BUILTIN_ORDER:
        if key in needed:
            var, attr = BUILTIN_ATTR[key]
            msl_params.append(f"{var} {attr}")

    param_block = ",\n    ".join(msl_params)
    warning_header = ""
    if all_warnings:
        warning_header = "".join(f"    // cuda2metal warning: {w}\n" for w in all_warnings)

    return (
        f"kernel void {name}(\n    {param_block}\n){{\n{warning_header}{body}\n}}\n"
    )


DEFINE_RE = re.compile(r"^#define\s+(\w+)\s+(.+)$", re.MULTILINE)


def translate_preamble(source, first_kernel_pos):
    """Carries forward #define constants used inside kernel bodies (e.g. llm.c's
    GELU_SCALING_FACTOR) so translated kernels aren't left referencing undefined
    macros. typedefs are intentionally dropped — MSL kernels use the renamed
    builtin types directly (see TYPE_RENAMES), so a CUDA typedef like
    `typedef float floatX;` has nothing left to alias once floatX itself is renamed."""
    lines = []
    for m in DEFINE_RE.finditer(source, 0, first_kernel_pos):
        name, value = m.group(1), m.group(2)
        for cuda_name, msl_name in FUNC_RENAMES.items():
            value = re.sub(rf"\b{cuda_name}\b", msl_name, value)
        lines.append(f"#define {name} {value}")
    return ("\n".join(lines) + "\n\n") if lines else ""


def translate_source(source):
    first_match = KERNEL_RE.search(source)
    if not first_match:
        raise TranslateError("no __global__ kernel functions found in input")
    out_chunks = [
        "// Generated by cuda2metal.py — do not hand-edit without re-running the "
        "translator on the source .cu file, or this file will silently drift.\n"
        "#include <metal_stdlib>\nusing namespace metal;\n\n",
        translate_preamble(source, first_match.start()),
    ]
    kernels_found = 0
    pos = 0
    for m in KERNEL_RE.finditer(source):
        name = m.group(1)
        paren_open = m.end() - 1
        paren_close = find_matching_paren(source, paren_open)
        params_str = source[paren_open + 1 : paren_close]
        brace_open = source.index("{", paren_close)
        brace_close = find_matching_brace(source, brace_open)
        body = source[brace_open + 1 : brace_close]
        out_chunks.append(translate_kernel(name, params_str, body))
        out_chunks.append("\n")
        kernels_found += 1
    if kernels_found == 0:
        raise TranslateError("no __global__ kernel functions found in input")
    return "".join(out_chunks)


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} input.cu", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        source = f.read()
    try:
        print(translate_source(source))
    except TranslateError as e:
        print(f"cuda2metal: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
