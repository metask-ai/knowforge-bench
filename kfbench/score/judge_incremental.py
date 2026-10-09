#!/usr/bin/env python3
"""SPEC28 全量分段评分：只判 answers 里尚未判过的题，合并进 judged 明细，打印阶段汇总。

用法: .venv/bin/python benchmark/eval/judge_incremental.py \
        --answers results/full_mmlb_task_t42_answers.json --judged results/full_mmlb_task_t42_judged.json
与 judge_score.py 同一套抽取(qwen 官方 prompt)+eval_score；evidence_sources 在 samples 里是字符串
形式的列表，这里 literal_eval 后再分桶（judge_score 直接迭代字符串会拆成单字符）。
"""
import argparse, ast, contextlib, io, json, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge_score import extract, parse_extraction, eval_score, eval_acc_and_f1, QWEN_MODEL  # noqa: E402


def as_list(v):
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except Exception:
            return [v]
    return list(v or [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--judged", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    records = json.load(open(a.answers))
    jp = Path(a.judged)
    judged = json.load(open(jp)) if jp.exists() else []
    judged = judged if isinstance(judged, list) else list(judged.values())
    done = {j["qid"] for j in judged}
    todo = [q for q in records.values() if q["qid"] not in done and (q.get("response") or q.get("error"))]

    def work(q):
        pred, _ = parse_extraction(extract(q["question"], q["response"] or ""))
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                score = eval_score(q["gold"], pred, q.get("answer_format") or "Str")
            except Exception:
                score = 0.0
        return {**q, "pred": pred, "score": score, "answer": q["gold"]}

    if todo:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            new = list(ex.map(work, sorted(todo, key=lambda q: q["qid"])))
        judged.extend(new)
        judged.sort(key=lambda j: j["qid"])
        json.dump(judged, open(jp, "w"), ensure_ascii=False, indent=1)
    else:
        new = []

    def summary(items, title):
        if not items:
            return
        acc, f1 = eval_acc_and_f1(items)
        una = [s for s in items if "Not answerable" in str(s["gold"])]
        ans = [s for s in items if s not in una]
        line = (f"{title}: n={len(items)} ACC={acc:.1%} F1={f1:.1%} | 可答 {sum(s['score'] for s in ans):.0f}/{len(ans)}"
                f" | 不可答 {sum(s['score'] for s in una):.0f}/{len(una)}")
        print(line)

    summary(judged, "累计")
    summary(new, "本段新增")
    if new:
        errs = sum(1 for s in new if s.get("error"))
        over = sum(1 for s in new if "Not answerable" in str(s["pred"]) and "Not answerable" not in str(s["gold"]))
        leak = sum(1 for s in new if "Not answerable" not in str(s["pred"]) and "Not answerable" in str(s["gold"]))
        print(f"本段错法: 过度拒答 {over} | 护栏失守 {leak} | harness错误 {errs}")

    def bucket(items, fn, name):
        agg = defaultdict(lambda: [0.0, 0])
        for s in items:
            for k in fn(s):
                agg[k][0] += s["score"]; agg[k][1] += 1
        print(name + ": " + "  ".join(f"{k} {sc:.0f}/{n}({sc/n:.0%})" for k, (sc, n) in sorted(agg.items())))

    bucket(judged, lambda s: (["Unanswerable"] if "Not answerable" in str(s["gold"]) else
                              [{"Pure-text (Plain-text)": "Text", "Generalized-text (Layout)": "Layout"}.get(x, x)
                               for x in (as_list(s.get("evidence_sources")) or ["?"])]), "累计模态桶")
    bucket(judged, lambda s: (["unanswerable"] if "Not answerable" in str(s["gold"]) else
                              ["cross-page" if len(as_list(s.get("evidence_pages"))) > 1 else "single-page"]), "累计位置桶")
    bucket(judged, lambda s: [str(s.get("answer_format"))], "累计格式桶")
    print(f"judge={QWEN_MODEL} 明细 -> {jp}")


if __name__ == "__main__":
    main()
