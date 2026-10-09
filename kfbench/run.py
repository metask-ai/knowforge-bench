"""knowforge-bench CLI.

  python -m kfbench.run subscribe [--bench all|m3d|mmlb|mhr]                                # once: add the benchmark KBs to your account
  python -m kfbench.run answer    --bench m3d|mmlb|mhr --mode M5|P1|P0|S4 [--loc 4 --deep 4] [--set full|mid|smoke]
                                  [--limit N] [--qids a,b] [--workers 4] --out results/<file>.json
  python -m kfbench.run retrieval --bench m3d|mhr --set ... --out results/<file>.json      # layer one: search only, no model
  python -m kfbench.run score     --bench m3d|mmlb|mhr --answers results/<file>.json        # scoring scripts (mmlb needs [judge])

Results are one JSON object per question keyed by qid; reruns skip answered questions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config as C
from . import datasets as D
from .kb_client import KB, subscribe as kb_subscribe
from .llm import LLM
from .pipeline import Pipeline


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def _load_out(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def cmd_answer(a) -> None:
    cfg = C.load(a.config)
    items = D.load(a.bench, cfg.data_dir, a.set, cfg.kb.packages)
    if a.qids:
        keep = {x.strip() for x in a.qids.split(",") if x.strip()}
        items = [q for q in items if q["qid"] in keep]
    if a.limit:
        items = items[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = _load_out(out)
    todo = [q for q in items if not results.get(q["qid"], {}).get("response")]
    log(f"bench={a.bench} mode={a.mode} loc={a.loc} deep={a.deep} set={a.set} {len(items)} q / todo {len(todo)} -> {out}")
    llm = LLM(cfg.model)
    vision = LLM(cfg.vision) if cfg.vision.ok() else None
    stage_llms = {name: LLM(ep) for name, ep in cfg.stages.items()}
    log(f"text model: {cfg.model.model} @ {cfg.model.base_url}" + (f" | stage overrides: {', '.join(f'{k}={v.model}' for k, v in cfg.stages.items())}" if cfg.stages else "")
        + (f" | vision: {cfg.vision.model}" if vision else " | vision: DISABLED (read_media off; configure [vision] to enable)"))
    kbs: dict[str, KB] = {}
    pipes: dict[str, Pipeline] = {}
    sem = asyncio.Semaphore(max(1, a.workers))
    lock = asyncio.Lock()
    t_start = time.time()
    done = [0]

    item_needs_docs = a.bench in ("m3d", "mhr")

    async def one(q: dict) -> None:
        async with sem:
            kb = kbs.setdefault(q["package_uuid"], KB(cfg.kb.base_url, q["package_uuid"], rate_sleep_s=cfg.kb.rate_sleep_s,
                                                     api_key=cfg.kb.api_key))
            pipe = pipes.setdefault(q["package_uuid"], Pipeline(kb, llm, vision, a.bench, a.mode, a.loc, a.deep, not a.no_bridge, stage_llms=stage_llms))
            try:
                res = await pipe.answer(q["question"])
            except Exception as exc:  # noqa: BLE001
                res = {"response": "", "tool_events": [], "elapsed_s": 0.0, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:300]}
            rec = D.record(a.bench, q, res, await kb.a(kb.root_names) if item_needs_docs else None)
            async with lock:
                results[q["qid"]] = rec
                out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
                done[0] += 1
                rate = (time.time() - t_start) / done[0]
                tail = (res.get("response") or "").strip().splitlines()[-1:] or [""]
                hit = f" dochit={rec['doc_hit_rate']:.0%}" if "doc_hit_rate" in rec else ""
                log(f"[{done[0]}/{len(todo)}] {q['qid'][:10]} {rec['elapsed_s']}s tools={len(rec['tools'])}{hit} "
                    f"avg={rate:.0f}s/q eta≈{rate * (len(todo) - done[0]) / 3600:.1f}h | {tail[0][:70]}")

    async def main():
        await asyncio.gather(*(one(q) for q in todo))

    asyncio.run(main())
    log(f"done -> {out} | llm calls {llm.calls} | prompt tok {llm.prompt_tokens} | completion tok {llm.completion_tokens}")


def cmd_retrieval(a) -> None:
    """Layer one: one hybrid search per question, document recall against gold evidence.  No model involved."""
    cfg = C.load(a.config, need_model=False) if Path(a.config).exists() else None
    kb_base = os.getenv("KFBENCH_KB_BASE_URL") or (cfg.kb.base_url if cfg else "")
    kb_key = os.getenv("KFBENCH_KB_API_KEY") or (cfg.kb.api_key if cfg else "")
    packages = cfg.kb.packages if cfg else json.loads(Path(a.packages).read_text(encoding="utf-8"))
    data_dir = cfg.data_dir if cfg else Path("data")
    items = D.load(a.bench, data_dir, a.set, packages)
    if a.limit:
        items = items[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = _load_out(out)
    kbs: dict[str, KB] = {}
    agg = {"n": 0, "r1": 0.0, "r3": 0.0, "r8": 0.0, "all_found": 0}
    for i, q in enumerate(items):
        if q["qid"] in results:
            rec = results[q["qid"]]
        else:
            kb = kbs.setdefault(q["package_uuid"], KB(kb_base, q["package_uuid"], api_key=kb_key))
            t0 = time.time()
            data = kb.search(q["raw_question"], a.topk, True)
            hits = D.search_hits([{"call": 0, "query": q["raw_question"], "data": data}], kb.root_names())
            docs = []          # text hits in rank order, then media hits (as in the in-process harness: any hit counts)
            for h in hits:
                if h["doc"] and h["doc"] not in docs:
                    docs.append(h["doc"])
            rec = {"qid": q["qid"], "supporting_doc_ids": q.get("supporting_doc_ids") or [], "docs_ranked": docs, "search_s": round(time.time() - t0, 3)}
            results[q["qid"]] = rec
            out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        sup = set(rec["supporting_doc_ids"])
        if sup:   # questions without gold evidence (e.g. MultiHop-RAG null queries) are excluded from recall
            agg["n"] += 1
            for k, key in ((1, "r1"), (3, "r3"), (8, "r8")):
                agg[key] += len(sup & set(rec["docs_ranked"][:k])) / len(sup)
            agg["all_found"] += int(sup <= set(rec["docs_ranked"]))
        if (i + 1) % 50 == 0 or i + 1 == len(items):
            n = agg["n"] or 1
            log(f"[{i + 1}/{len(items)}] n={agg['n']} R@1 {100 * agg['r1'] / n:.1f} R@3 {100 * agg['r3'] / n:.1f} "
                f"R@8 {100 * agg['r8'] / n:.1f} all-found {100 * agg['all_found'] / n:.1f} (text + media hits; media attribution needs server skill >= 0.5.0)")


def _package_uuids(packages: dict, bench: str) -> list[str]:
    keys = {"m3d": "m3docvqa", "mhr": "multihop_rag", "mmlb": "mmlongbench_doc"}
    out: list[str] = []
    for b, key in keys.items():
        if bench not in ("all", b):
            continue
        for v in (packages.get(key) or {}).values():
            u = v.get("package_uuid") if isinstance(v, dict) else v
            if u and u not in out:
                out.append(u)
    return out


def cmd_subscribe(a) -> None:
    """Add the benchmark knowledge bases to the account behind [kb] api_key (one call; safe to repeat)."""
    cfg = C.load(a.config, need_model=False)
    uuids = _package_uuids(cfg.kb.packages, a.bench)
    if not uuids:
        raise SystemExit(f"no package UUIDs for --bench {a.bench} in the packages file")
    d = kb_subscribe(cfg.kb.base_url, cfg.kb.api_key, uuids)
    log(f"subscribe {a.bench}: {len(uuids)} packages | added {len(d.get('added') or [])} | "
        f"already present {len(d.get('skipped') or [])}")


def cmd_score(a) -> None:
    here = Path(__file__).parent / "score"
    if a.bench == "m3d":
        subprocess.run([sys.executable, str(here / "m3d_score.py"), "--answers", a.answers] + a.extra, check=False)
    elif a.bench == "mhr":
        subprocess.run([sys.executable, str(here / "mhr_score.py"), "--answers", a.answers] + a.extra, check=False)
    else:
        cfg = C.load(a.config)
        if not cfg.judge.ok():
            raise SystemExit("MMLongBench-Doc scoring needs a [judge] endpoint in the config")
        env = {**os.environ, "KFBENCH_JUDGE_BASE_URL": cfg.judge.base_url, "KFBENCH_JUDGE_MODEL": cfg.judge.model,
               "KFBENCH_JUDGE_API_KEY": cfg.judge.api_key, "KFBENCH_MMLB_REPO": str(cfg.data_dir / "vendor" / "MMLongBench-Doc")}
        judged = a.answers.replace("_answers.json", "_judged.json") if a.answers.endswith("_answers.json") else a.answers + ".judged.json"
        subprocess.run([sys.executable, str(here / "judge_incremental.py"), "--answers", a.answers, "--judged", judged] + a.extra, check=False, env=env)


def main():
    ap = argparse.ArgumentParser(prog="kfbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("subscribe")
    b.add_argument("--config", default="config.toml")
    b.add_argument("--bench", default="all", choices=["all", "m3d", "mmlb", "mhr"])
    b.set_defaults(fn=cmd_subscribe)
    p = sub.add_parser("answer")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--bench", required=True, choices=["m3d", "mmlb", "mhr"])
    p.add_argument("--mode", default="M5", choices=["M5", "P1", "P0", "S4"])
    p.add_argument("--loc", type=int, default=4)
    p.add_argument("--deep", type=int, default=4)
    p.add_argument("--no-bridge", action="store_true")
    p.add_argument("--set", default="full", choices=["full", "mid", "smoke"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--qids", default="")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_answer)
    r = sub.add_parser("retrieval")
    r.add_argument("--config", default="config.toml")
    r.add_argument("--packages", default="data/kb_packages.json")
    r.add_argument("--bench", required=True, choices=["m3d", "mhr"])
    r.add_argument("--set", default="full", choices=["full", "mid", "smoke"])
    r.add_argument("--topk", type=int, default=8)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_retrieval)
    s = sub.add_parser("score")
    s.add_argument("--config", default="config.toml")
    s.add_argument("--bench", required=True, choices=["m3d", "mmlb", "mhr"])
    s.add_argument("--answers", required=True)
    s.add_argument("extra", nargs="*")
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
