#!/usr/bin/env python3
"""Build project: run cmake + make under build/.

Default jobs: env BUILD_JOBS, else os.cpu_count() (fallback 8).
Override: python build.py --jobs N
"""

import subprocess
import sys
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.absolute()
BUILD_DIR = PROJECT_DIR / "build"


def default_jobs() -> int:
    env = os.environ.get("BUILD_JOBS")
    if env:
        return int(env)
    return os.cpu_count() or 8


def build(jobs: int) -> int:
    BUILD_DIR.mkdir(exist_ok=True)

    # cmake configure (only if CMakeCache.txt doesn't exist)
    cache_file = BUILD_DIR / "CMakeCache.txt"
    if not cache_file.exists():
        print("=== CMake Configure ===")
        ret = subprocess.call(
            ["cmake", "-B", str(BUILD_DIR), "-S", str(PROJECT_DIR)],
            cwd=str(PROJECT_DIR),
        )
        if ret != 0:
            print("CMake configure failed!")
            return ret

    print(f"=== Make (jobs={jobs}) ===")
    ret = subprocess.call(
        ["make", f"-j{jobs}"],
        cwd=str(BUILD_DIR),
    )
    if ret != 0:
        print("Build failed!")
    else:
        print("Build succeeded.")
    return ret


if __name__ == "__main__":
    jobs = default_jobs()
    for i, arg in enumerate(sys.argv):
        if arg in ("-j", "--jobs") and i + 1 < len(sys.argv):
            jobs = int(sys.argv[i + 1])
    sys.exit(build(jobs=jobs))
