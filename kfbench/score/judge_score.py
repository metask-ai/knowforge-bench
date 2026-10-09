#!/usr/bin/env python3
"""SPEC28 M1: 官方两段式评分——qwen 抽取(官方 prompt 原文) + 官方 eval_score 规则匹配。

用法: python3 judge_score.py [--answers ../results/mmlb_smoke_answers.json]
输出: 总 acc/F1 + 模态桶 + 位置桶；judged 明细存 results/mmlb_smoke_judged.json。
"""
import argparse, contextlib, io, json, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

import os
HERE = Path(__file__).resolve().parent
# The official MMLongBench-Doc repo (cloned by data/fetch_mmlb.py into <data_dir>/vendor/MMLongBench-Doc).
# kfbench.run passes its location via KFBENCH_MMLB_REPO; the fallback is the default data_dir layout.
MMLB_REPO = Path(os.getenv("KFBENCH_MMLB_REPO") or (HERE.parent.parent / "data" / "vendor" / "MMLongBench-Doc"))
if not (MMLB_REPO / "eval" / "eval_score.py").is_file():
    raise SystemExit(f"MMLongBench-Doc eval code not found at {MMLB_REPO}/eval — run `python data/fetch_mmlb.py` first")
sys.path.insert(0, str(MMLB_REPO / "eval"))
from eval_score import eval_score, eval_acc_and_f1  # noqa: E402

PROMPT = (MMLB_REPO / "eval" / "prompt_for_answer_extraction.md").read_text()
# Judge model = the benchmark's model-judge protocol run on an OpenAI-compatible endpoint that YOU provide.
# The paper used qwen3.6-35b-a3b-fp8 as judge; judge choice moves MMLongBench-Doc scores by 2-3 points, so report it.
QWEN_URL = os.environ.get("KFBENCH_JUDGE_BASE_URL", "").rstrip("/") + "/chat/completions" if os.environ.get("KFBENCH_JUDGE_BASE_URL") else ""
QWEN_MODEL = os.environ.get("KFBENCH_JUDGE_MODEL", "")
QWEN_KEY = os.environ.get("KFBENCH_JUDGE_API_KEY", "")
if not QWEN_URL:
    raise SystemExit("set KFBENCH_JUDGE_BASE_URL (OpenAI-compatible, e.g. https://api.example.com/v1) and KFBENCH_JUDGE_MODEL")


def extract(question: str, output: str) -> str:
    body = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": f"\n\nQuestion:{question}\nAnalysis:{output}\n"},
        ],
        "temperature": 0.0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for _ in range(3):
        try:
            r = requests.post(QWEN_URL, json=body, timeout=120, headers=({"Authorization": f"Bearer {QWEN_KEY}"} if QWEN_KEY else None))
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            continue
    return "Failed"


def parse_extraction(resp: str) -> tuple[str, str]:
    ans = re.search(r"Extracted answer\s*[:：]\s*(.+)", resp)
    fmt = re.search(r"Answer format\s*[:：]\s*(.+)", resp)
    return (ans.group(1).strip() if ans else resp.strip()[:80],
            fmt.group(1).strip() if fmt else "Str")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(HERE.parent / "results" / "mmlb_smoke_answers.json"))
    ap.add_argument("--out", default="", help="judged 明细输出路径（默认 results/mmlb_smoke_judged.json）")
    args = ap.parse_args()
    records = json.load(open(args.answers))
    items = sorted(records.values(), key=lambda q: q["qid"])

    def work(q):
        pred, _ = parse_extraction(extract(q["question"], q["response"] or ""))
        with contextlib.redirect_stdout(io.StringIO()):   # eval_score 的 list 分支有 print
            try:
                score = eval_score(q["gold"], pred, q.get("answer_format") or "Str")
            except Exception:
                score = 0.0
        return {**q, "pred": pred, "score": score, "answer": q["gold"]}

    with ThreadPoolExecutor(max_workers=4) as ex:
        judged = list(ex.map(work, items))

    acc, f1 = eval_acc_and_f1(judged)
    print(f"官方口径: ACC={acc:.1%}  F1={f1:.1%}  (n={len(judged)}, judge={QWEN_MODEL})\n")

    def bucket(pred_fn, name):
        agg = defaultdict(lambda: [0.0, 0])
        for s in judged:
            for k in pred_fn(s):
                agg[k][0] += s["score"]; agg[k][1] += 1
        print(name + ":")
        for k, (sc, n) in sorted(agg.items()):
            print(f"  {k:14s} {sc:.1f}/{n}  ({sc/n:.0%})")
        print()

    bucket(lambda s: (["Unanswerable"] if "Not answerable" in str(s["gold"]) else
                      [{"Pure-text (Plain-text)": "Text", "Generalized-text (Layout)": "Layout"}.get(x, x)
                       for x in (s["evidence_sources"] or ["?"])]), "模态桶")
    bucket(lambda s: (["unanswerable"] if "Not answerable" in str(s["gold"]) else
                      ["cross-page" if len(s.get("evidence_pages") or []) > 1 else "single-page"]), "位置桶")

    out = Path(args.out) if args.out else HERE.parent / "results" / "mmlb_smoke_judged.json"
    json.dump(judged, open(out, "w"), ensure_ascii=False, indent=1)
    print("明细 ->", out)


if __name__ == "__main__":
    main()
