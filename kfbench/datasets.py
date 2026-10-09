"""Question sets and per-question records for the three benchmarks.

Files (fetched/built by data/fetch_*.py into `data_dir`):
  m3docvqa/m3d_full.json            {"questions": [{qid, question, answers, qtype, modalities, supporting_doc_ids, ...}]}
  m3docvqa/m3d_mid.json, m3d_smoke.json   same format, stratified subsets (mid = 300, smoke = 30)
  multihop_rag/mhr_full.json        {"questions": [{qid, query, answer, question_type, evidence_list}]}
  multihop_rag/mhr_sets.json        {"full": [qid...], "mid": [...], "smoke": [...]}
  multihop_rag/md_index.json        {"by_url": {url: md_stem}, "by_title": {title: md_stem}}
  mmlongbench_doc/samples.json      official MMLongBench-Doc samples
  kb_packages.json                  package UUIDs per benchmark (see config.example.toml)
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .prompts import GUARDRAILS

_WS = re.compile(r"\s+")


def _norm_title(t: str) -> str:
    return _WS.sub(" ", (t or "").strip().lower())


def _listish(v) -> list:
    if isinstance(v, list):
        return v
    try:
        x = ast.literal_eval(str(v))
        return list(x) if isinstance(x, (list, tuple)) else [x]
    except Exception:  # noqa: BLE001
        return [v] if v not in (None, "") else []


def load(bench: str, data_dir: Path, subset: str, packages: dict) -> list[dict]:
    """Returns items: {qid, question(with guardrail), raw_question, package_uuid, gold..., supporting_doc_ids}."""
    items: list[dict] = []
    if bench == "m3d":
        fn = {"full": "m3d_full.json", "mid": "m3d_mid.json", "smoke": "m3d_smoke.json"}[subset]
        qs = json.loads((data_dir / "m3docvqa" / fn).read_text(encoding="utf-8"))["questions"]
        pkg = next(iter(packages["m3docvqa"].values()))
        for q in qs:
            items.append({"qid": q["qid"], "raw_question": q["question"], "question": q["question"] + GUARDRAILS["m3d"],
                          "package_uuid": pkg, "answers": _listish(q.get("answers")), "qtype": q.get("qtype"),
                          "modalities": _listish(q.get("modalities")), "supporting_doc_ids": _listish(q.get("supporting_doc_ids"))})
    elif bench == "mhr":
        d = data_dir / "multihop_rag"
        qs = json.loads((d / "mhr_full.json").read_text(encoding="utf-8"))["questions"]
        keep = set(json.loads((d / "mhr_sets.json").read_text(encoding="utf-8"))[subset])
        idx = json.loads((d / "md_index.json").read_text(encoding="utf-8"))
        pkg = next(iter(packages["multihop_rag"].values()))
        for q in qs:
            if q["qid"] not in keep:
                continue
            sup = []
            for e in q.get("evidence_list") or []:
                doc = idx["by_url"].get(e.get("url") or "") or idx["by_title"].get(_norm_title(e.get("title", "")))
                if doc and doc not in sup:
                    sup.append(doc)
            items.append({"qid": q["qid"], "raw_question": q["query"], "question": q["query"] + GUARDRAILS["mhr"], "package_uuid": pkg,
                          "answer": q["answer"], "question_type": q["question_type"], "evidence_list": q.get("evidence_list") or [],
                          "supporting_doc_ids": sup})
    elif bench == "mmlb":
        qs = json.loads((data_dir / "mmlongbench_doc" / "samples.json").read_text(encoding="utf-8"))
        pkgs = packages["mmlongbench_doc"]
        for i, q in enumerate(qs):
            info = pkgs.get(q["doc_id"])
            if not info:
                continue
            qid = f"full-{i:04d}"
            if subset == "smoke" and i % 45 != 0:
                continue
            if subset == "mid" and i % 11 != 0:
                continue
            items.append({"qid": qid, "raw_question": q["question"], "question": q["question"] + GUARDRAILS["mmlb"],
                          "package_uuid": info["package_uuid"] if isinstance(info, dict) else info,
                          "doc_id": q["doc_id"], "doc_type": q.get("doc_type"), "gold": q.get("answer"), "answer": q.get("answer"),
                          "answer_format": q.get("answer_format"), "evidence_pages": q.get("evidence_pages"),
                          "evidence_sources": q.get("evidence_sources")})
    else:
        raise SystemExit(f"unknown bench {bench}")
    return items


def _doc_of(item: dict) -> str:
    sf = str(item.get("source_file") or "").replace("\\", "/").split("/")[-1]
    return sf.rsplit(".", 1)[0] if sf else ""


def _stem(name: str) -> str:
    name = str(name or "").replace("\\", "/").split("/")[-1]
    return name.rsplit(".", 1)[0] if name else ""


def search_hits(search_log: list, root_names: dict | None = None) -> list[dict]:
    """Flatten every search call into per-hit rows.  Image/table hits carry their document through `root_node_id`
    (servers >= skill 0.5.0); `root_names` (root_node_id -> file name, from /sources) turns it into the document id.
    Without it media hits get an empty `doc` and are excluded from recall, which can only lower R@k."""
    root_names = root_names or {}
    hits = []
    for e in search_log or []:
        data = e.get("data") or {}
        base = {"call": e.get("call", 0), "query": e.get("query", ""), "bridge": bool(e.get("bridge"))}
        for it in data.get("results") or []:
            hits.append({**base, "fragment_id": it.get("fragment_id"), "kind": "text", "score": it.get("score"),
                         "from_graph": bool(it.get("from_graph")), "doc": _doc_of(it), "page": it.get("page"),
                         "vec": it.get("vec_score"), "raw": it.get("raw_score")})
        for key in ("images", "tables", "examples"):
            for it in data.get(key) or []:
                if key == "examples":
                    doc = _doc_of(it)
                else:
                    doc = _stem(root_names.get(str(it.get("root_node_id") or ""), ""))
                hits.append({**base, "fragment_id": it.get("fragment_id"), "kind": it.get("kind") or key.rstrip("s"),
                             "score": it.get("score"), "from_graph": False, "doc": doc,
                             "vec": it.get("vec_score"), "raw": it.get("raw_score")})
    return hits


def record(bench: str, item: dict, res: dict, root_names: dict | None = None) -> dict:
    hits = search_hits(res.get("search_log") or [], root_names)
    rec: dict[str, Any] = {k: v for k, v in item.items() if k not in ("question",)}
    rec["question"] = item["raw_question"]
    rec.update({"response": res.get("response", ""), "search_hits": hits, "bridge": res.get("bridge") or {}, "fit": res.get("fit") or {},
                "deep_fids": res.get("deep_fids") or [], "tools": [e["name"] for e in res.get("tool_events") or []],
                "briefs": [(e.get("brief") or "")[:160] for e in res.get("tool_events") or []],
                "final_check": res.get("final_check") or {}, "mode": res.get("status"), "error": res.get("error"),
                "elapsed_s": res.get("elapsed_s"), "timing": res.get("timing") or {}, "llm_calls": res.get("llm_calls"),
                "kb_calls": res.get("kb_calls"), "read_media_calls": res.get("read_media_calls", 0)})
    sup = set(item.get("supporting_doc_ids") or [])
    if sup:
        found = {h["doc"] for h in hits if h.get("doc")}
        first = {h["doc"] for h in hits if h.get("doc") and h.get("call") == 0}
        rec["doc_hit"] = sorted(sup & found)
        rec["doc_hit_rate"] = len(sup & found) / len(sup)
        rec["doc_hit_first"] = len(sup & first) / len(sup)
    if bench == "mmlb":
        pages = set()
        for im in (res.get("images") or {}).values():
            m = re.search(r"page_(\d+)_", im.get("path") or "")
            if m:
                pages.add(int(m.group(1)))
        for h in hits:
            if h.get("page"):
                pages.add(int(h["page"]))
        rec["found_pages"] = sorted(pages)
    return rec
