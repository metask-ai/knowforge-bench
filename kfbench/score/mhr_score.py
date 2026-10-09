#!/usr/bin/env python3
"""MultiHop-RAG 双口径评分（SPEC28.4 §4.1，2026-09-19）。

输入：full_mhr_ask.py / ablation_search.py --bench mhr 的 answers.json（qid → 记录，含 response / answer / question_type / search_hits）。
口径一（官方）：qa_evaluate.py 的 has_intersection——预测与 gold 小写后按空格分词有交集即对；null 题 gold="Insufficient information."。
口径二（我方）：token 级 F1（与 m3d_score 同款规范化），null 题按 EM。
按 question_type 分桶 + 检索侧（evidence 文档命中：首搜 R@k、全程并集、全书找到）。
--export-qa <path>：导出官方 qa_evaluate.py 输入（model_answer 写成 'The answer to the question is "<FINAL>"'）
--export-retrieval <path>：导出官方 retrieval_evaluate.py 输入（retrieval_list=首搜片段正文，取 Mongo；gold_list=evidence fact）
用法：.venv/bin/python benchmark/eval/mhr_score.py --answers benchmark/results/full_mhr_task_t43h_answers.json [--md]
"""
import argparse, json, re, string, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark" / "eval"))

_FINAL_RE = re.compile(r"(?im)^\s*\**\s*FINAL ANSWER\s*[:：]\s*\**\s*(.*?)\s*\**\s*$")
NULL_GOLD = "insufficient information"


def final_of(resp: str) -> str:
    m = None
    for m in _FINAL_RE.finditer(resp or ""):
        pass
    return (m.group(1).strip() if m else (resp or "").strip().splitlines()[-1:] or [""])[0] if not m else m.group(1).strip()


def normalize(s: str) -> str:
    s = (s or "").lower().strip().strip("'\"")
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def official_ok(pred: str, gold: str) -> bool:
    """qa_evaluate.py 的 has_intersection（小写、空格分词、有交集）。"""
    return bool(set(pred.lower().split()) & set(gold.lower().split()))


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc)


def is_null(gold: str) -> bool:
    return normalize(gold).startswith(NULL_GOLD)


def score_one(rec: dict) -> dict:
    pred = final_of(rec.get("response") or "")
    gold = rec.get("answer") or ""
    if is_null(gold):
        ok = normalize(pred).startswith(NULL_GOLD)
        return {"pred": pred, "official": float(ok), "f1": float(ok), "em": float(ok)}
    return {"pred": pred, "official": float(official_ok(pred, gold)), "f1": token_f1(pred, gold),
            "em": float(normalize(pred) == normalize(gold))}


def retrieval_of(rec: dict) -> dict:
    sup = set(rec.get("supporting_doc_ids") or [])
    hits = rec.get("search_hits") or []
    first = [h for h in hits if (h.get("call") or 0) == 0]
    docs_first = []
    for h in first:
        if h.get("doc") and h["doc"] not in docs_first:
            docs_first.append(h["doc"])
    found_all = {h.get("doc") for h in hits if h.get("doc")}
    out = {}
    if sup:
        for k in (1, 3, 8):
            out[f"r@{k}"] = len(sup & set(docs_first[:k])) / len(sup)
        out["r@all"] = len(sup & found_all) / len(sup)
        out["all_found"] = float(sup <= found_all)
    return out


def pct(v):
    return round(100 * sum(v) / len(v), 1) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--export-qa", default="")
    ap.add_argument("--export-retrieval", default="")
    a = ap.parse_args()
    data = json.loads(Path(a.answers).read_text())
    rows = [(k, v) for k, v in data.items() if v.get("response")]
    per = {}
    for k, v in rows:
        per[k] = {**score_one(v), **retrieval_of(v), "type": v.get("question_type", "?")}
    groups = defaultdict(list)
    for k, s in per.items():
        groups[s["type"]].append(s)
        groups["ALL"].append(s)
    print(f"答题 {len(rows)} 题（{a.answers}）")
    hdr = "| 桶 | n | 官方准确率 | F1 | EM | 首搜R@1 | 首搜R@3 | 首搜R@8 | R@all | 全证据找到 |"
    print(hdr); print("|---|---|---|---|---|---|---|---|---|---|")
    for g in ["ALL", "inference_query", "comparison_query", "temporal_query", "null_query"]:
        v = groups.get(g) or []
        if not v:
            continue
        ret = [x for x in v if "r@1" in x]
        print(f"| {g} | {len(v)} | {pct([x['official'] for x in v])} | {pct([x['f1'] for x in v])} | {pct([x['em'] for x in v])} | "
              f"{pct([x['r@1'] for x in ret])} | {pct([x['r@3'] for x in ret])} | {pct([x['r@8'] for x in ret])} | {pct([x['r@all'] for x in ret])} | {pct([x['all_found'] for x in ret])} |")
    # 拒答率：预测为 Insufficient information 的比例（非 null 题上是误拒）
    refuse = [k for k, s in per.items() if normalize(s["pred"]).startswith(NULL_GOLD)]
    wrong_refuse = [k for k in refuse if not is_null(data[k].get("answer") or "")]
    print(f"预测'Insufficient information'：{len(refuse)} 题，其中非 null 题误拒 {len(wrong_refuse)}")
    if a.export_qa:
        out = [{"query": v["question"], "model_answer": f'The answer to the question is "{per[k]["pred"]}"',
                "gold_answer": v["answer"], "question_type": v.get("question_type")} for k, v in rows]
        Path(a.export_qa).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print("官方 qa_evaluate 输入 ->", a.export_qa, "（python test/MultiHop-RAG/qa_evaluate.py --file <它> --queries test/MultiHop-RAG/MultiHopRAG.json）")
    if a.export_retrieval:
        from ablation_table import load_frag_texts, _FRAG_TEXT
        fids = [h.get("fragment_id") for _, v in rows for h in (v.get("search_hits") or []) if (h.get("call") or 0) == 0]
        load_frag_texts(fids)
        out = []
        for k, v in rows:
            first = [h for h in (v.get("search_hits") or []) if (h.get("call") or 0) == 0]
            out.append({"query": v["question"], "question_type": v.get("question_type"),
                        "retrieval_list": [{"text": _FRAG_TEXT.get(h.get("fragment_id") or "", "")} for h in first],
                        "gold_list": [{"fact": e.get("fact", "")} for e in (v.get("evidence_list") or [])]})
        Path(a.export_retrieval).write_text(json.dumps(out, ensure_ascii=False))
        print("官方 retrieval_evaluate 输入 ->", a.export_retrieval, "（注意：其 gold fact 是原文子串匹配，我方片段文本经规范化，命中口径偏严）")


if __name__ == "__main__":
    main()
