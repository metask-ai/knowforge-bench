#!/usr/bin/env python3
"""Build the M3DocVQA question set from the official MultimodalQA release.

M3DocVQA (Cho et al., 2024) is the open-domain re-framing of MultimodalQA's dev split: every one of the 2,441 dev
questions is asked against the union of all supporting Wikipedia documents (3,368 PDFs).  This script downloads the
MultimodalQA dev questions and the text/image/table metadata from https://github.com/allenai/multimodalqa and writes

  data/m3docvqa/m3d_full.json    {"questions": [{qid, question, answers, qtype, modalities, supporting_doc_ids, supporting_titles}], "pool": [...]}
  data/m3docvqa/m3d_mid.json     the paper's stratified 300-question subset  (qids in data/subsets/m3d_mid_qids.json)
  data/m3docvqa/m3d_smoke.json   30-question smoke subset

The PDFs are NOT downloaded: the indexed corpus is served by the public knowledge base.
Usage: python data/fetch_m3docvqa.py [--src-dir <dir with MMQA_*.jsonl(.gz) already downloaded>]
"""
from __future__ import annotations

import argparse
import gzip
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "m3docvqa"
RAW = "https://github.com/allenai/multimodalqa/raw/master/dataset/"
FILES = ["MMQA_dev.jsonl.gz", "MMQA_texts.jsonl.gz", "MMQA_images.jsonl.gz", "MMQA_tables.jsonl.gz"]


def _lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def fetch(src_dir: Path) -> None:
    src_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dst = src_dir / name
        if dst.exists() or (src_dir / name[:-3]).exists():
            continue
        print("downloading", name)
        urllib.request.urlretrieve(RAW + name, dst)


def find(src_dir: Path, name: str) -> Path:
    for cand in (src_dir / name, src_dir / name[:-3]):
        if cand.exists():
            return cand
    raise SystemExit(f"missing {name} in {src_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default=str(HERE / "vendor" / "multimodalqa"))
    a = ap.parse_args()
    src = Path(a.src_dir)
    fetch(src)
    qs = list(_lines(find(src, "MMQA_dev.jsonl.gz")))
    pool_ids = {sc["doc_id"] for q in qs for sc in (q.get("supporting_context") or [])}
    mapping: dict[str, dict] = {}
    for name in ("MMQA_texts.jsonl.gz", "MMQA_images.jsonl.gz", "MMQA_tables.jsonl.gz"):
        for r in _lines(find(src, name)):
            rid = r.get("id")
            if rid in pool_ids and rid not in mapping:
                mapping[rid] = {"url": r.get("url"), "title": r.get("title")}
    out = {
        "benchmark": "M3DocVQA", "subset": "full-dev", "pool_size": len(mapping),
        "questions": [{
            "qid": q["qid"], "question": q["question"],
            "answers": [x["answer"] for x in q["answers"]],
            "qtype": q["metadata"]["type"], "modalities": q["metadata"]["modalities"],
            "supporting_doc_ids": [sc["doc_id"] for sc in q["supporting_context"]],
            "supporting_titles": [mapping.get(sc["doc_id"], {}).get("title") for sc in q["supporting_context"]],
        } for q in qs],
        "pool": [{"doc_id": d, **mapping[d]} for d in sorted(mapping)],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "m3d_full.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"m3d_full.json: {len(out['questions'])} questions, {len(mapping)} documents")
    for sub in ("mid", "smoke"):
        keep = set(json.loads((HERE / "subsets" / f"m3d_{sub}_qids.json").read_text()))
        part = {**out, "subset": sub, "questions": [q for q in out["questions"] if q["qid"] in keep]}
        (OUT / f"m3d_{sub}.json").write_text(json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"m3d_{sub}.json: {len(part['questions'])} questions")


if __name__ == "__main__":
    main()
