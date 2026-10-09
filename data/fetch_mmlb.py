#!/usr/bin/env python3
"""Fetch the official MMLongBench-Doc question set and evaluation code.

Clones https://github.com/mayubo2333/MMLongBench-Doc (code Apache-2.0; data CC BY-NC 4.0, research use) into
data/vendor/MMLongBench-Doc and links data/mmlongbench_doc/samples.json.  The 135 PDF documents themselves are NOT
needed here: they are served, already indexed, by the public knowledge bases (one package per document).

Usage: python data/fetch_mmlb.py
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor" / "MMLongBench-Doc"
OUT = HERE / "mmlongbench_doc"
REPO = "https://github.com/mayubo2333/MMLongBench-Doc"


def main() -> None:
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    if not VENDOR.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(VENDOR)], check=True)
    samples = VENDOR / "data" / "samples.json"
    if not samples.exists():
        raise SystemExit(f"samples.json not found under {VENDOR}/data — repository layout changed?")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy(samples, OUT / "samples.json")
    print(f"ok -> {OUT / 'samples.json'}  (official eval code: {VENDOR / 'eval'})")
    print("MMLongBench-Doc data is CC BY-NC 4.0 (research use only).")


if __name__ == "__main__":
    main()
