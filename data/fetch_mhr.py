#!/usr/bin/env python3
"""Build the MultiHop-RAG question set from the official release (Tang & Yang, 2024; ODC-BY).

Downloads MultiHopRAG.json from Hugging Face (yixuantt/MultiHopRAG; use --mirror for a mirror host, or --src to point
at a file you downloaded yourself) and writes

  data/multihop_rag/mhr_full.json   {"questions": [{qid, query, answer, question_type, evidence_list}]}   qid = md5(query)[:16]
  data/multihop_rag/mhr_sets.json   {"full": [...], "mid": [...], "smoke": [...]}   (paper subsets; from data/subsets)
  data/multihop_rag/md_index.json   evidence url/title -> indexed article name (metadata shipped in data/subsets)

The 609 news articles are NOT downloaded: the indexed corpus is served by the public knowledge base.
Usage: python data/fetch_mhr.py [--mirror https://hf-mirror.com] [--src /path/to/MultiHopRAG.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "multihop_rag"
PATH = "/datasets/yixuantt/MultiHopRAG/resolve/main/MultiHopRAG.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default="https://huggingface.co")
    ap.add_argument("--src", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src = Path(a.src) if a.src else HERE / "vendor" / "MultiHopRAG.json"
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", a.mirror.rstrip("/") + PATH)
        urllib.request.urlretrieve(a.mirror.rstrip("/") + PATH, src)
    raw = json.loads(src.read_text(encoding="utf-8"))
    qs = [{"qid": hashlib.md5(q["query"].encode("utf-8")).hexdigest()[:16], "query": q["query"], "answer": q["answer"],
           "question_type": q["question_type"], "evidence_list": q.get("evidence_list") or []} for q in raw]
    (OUT / "mhr_full.json").write_text(json.dumps({"questions": qs}, ensure_ascii=False, indent=1), encoding="utf-8")
    sets = json.loads((HERE / "subsets" / "mhr_sets.json").read_text(encoding="utf-8"))
    sets["full"] = [q["qid"] for q in qs]
    (OUT / "mhr_sets.json").write_text(json.dumps(sets, ensure_ascii=False), encoding="utf-8")
    shutil.copy(HERE / "subsets" / "mhr_md_index.json", OUT / "md_index.json")
    print(f"mhr_full.json: {len(qs)} questions; sets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()))


if __name__ == "__main__":
    main()
