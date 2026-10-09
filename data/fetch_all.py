#!/usr/bin/env python3
"""Fetch the three official question sets into data/ (no documents, no run outputs).

  M3DocVQA        <- MultimodalQA dev split (github.com/allenai/multimodalqa)
  MultiHop-RAG    <- huggingface.co/datasets/yixuantt/MultiHopRAG  (ODC-BY)
  MMLongBench-Doc <- github.com/mayubo2333/MMLongBench-Doc         (data CC BY-NC 4.0, research use)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for script in ("fetch_m3docvqa.py", "fetch_mhr.py", "fetch_mmlb.py"):
    print(f"== {script}")
    subprocess.run([sys.executable, str(HERE / script)] + sys.argv[1:], check=False)
