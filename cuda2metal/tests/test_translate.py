#!/usr/bin/env python3
"""
Test harness for cuda2metal. Not just golden-string comparison: every example is
run through xcrun metal (the real Apple Metal compiler) to prove the output is
actually valid MSL, not just plausible-looking text. Requires macOS + the Metal
toolchain (same requirement as the rest of metalsw's GPU path) — this will not
run on the project's Linux devcontainer, same caveat as metalsw_gpu itself.

Usage: python3 tests/test_translate.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import cuda2metal  # noqa: E402

EXAMPLES = sorted((ROOT / "examples").glob("*.cu"))


def main():
    if not shutil.which("xcrun") or subprocess.run(
        ["xcrun", "-f", "metal"], capture_output=True
    ).returncode != 0:
        print("SKIP: xcrun metal not available (need macOS + Metal toolchain)")
        return 0

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for cu_file in EXAMPLES:
            name = cu_file.stem
            try:
                msl = cuda2metal.translate_source(cu_file.read_text())
            except cuda2metal.TranslateError as e:
                failures.append(f"{name}: translation failed: {e}")
                continue
            metal_path = tmp / f"{name}.metal"
            air_path = tmp / f"{name}.air"
            metal_path.write_text(msl)
            result = subprocess.run(
                ["xcrun", "metal", "-c", str(metal_path), "-o", str(air_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                failures.append(f"{name}: xcrun metal failed:\n{result.stderr}")
            else:
                print(f"PASS  {name}.cu -> compiles as valid MSL")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"\n{len(EXAMPLES)}/{len(EXAMPLES)} examples translate and compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
