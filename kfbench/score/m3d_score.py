#!/usr/bin/env python3
"""SPEC28 M3: M3DocVQA(MMQA) 确定性评分——SQuAD 式规范化 + MMQA 式列表对齐 EM/F1,无 judge 模型。

答案取 response 的 'FINAL ANSWER:' 行;JSON 列表按元素对齐(小规模穷举最优匹配)。
用法: python3 m3d_score.py [--answers .../m3d_smoke_answers.json]
"""
import argparse, ast, json, re, string
from collections import defaultdict
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE.parent / "results" / "m3d_smoke_answers.json"

NO_ANSWER_PAT = re.compile(r"not (available|found|present|answerable)|no information|unable to", re.I)


def normalize(s):
    s = str(s).lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = defaultdict(int)
    for t in pt:
        common[t] += 1
    overlap = sum(min(common.get(t, 0), gt.count(t)) for t in set(gt))
    if overlap == 0:
        return 0.0
    p, r = overlap / len(pt), overlap / len(gt)
    return 2 * p * r / (p + r)


def list_f1(preds, golds):
    """MMQA 式:元素两两 token-F1 建分矩阵,取最优一一匹配;P=匹配和/|pred|,R=/|gold|。"""
    if not preds or not golds:
        return float(bool(preds) == bool(golds))
    short, long_, transposed = (preds, golds, False) if len(preds) <= len(golds) else (golds, preds, True)
    mat = [[token_f1(a, b) for b in long_] for a in short]
    if len(long_) <= 8:
        best = max(sum(mat[i][p[i]] for i in range(len(short))) for p in permutations(range(len(long_)), len(short)))
    else:  # 大列表退化为贪心
        best, used = 0.0, set()
        for row in mat:
            j = max((j for j in range(len(long_)) if j not in used), key=lambda j: row[j])
            used.add(j); best += row[j]
    p, r = best / len(preds), best / len(golds)
    return 2 * p * r / (p + r) if p + r else 0.0


def extract_pred(response):
    """取末个 FINAL ANSWER 行;解析 JSON/Python 列表;识别'无答案'表述。"""
    lines = [l for l in (response or "").splitlines() if "FINAL ANSWER" in l.upper()]
    if not lines:
        return []  # 协议失守,按空答计
    ans = re.sub(r".*FINAL ANSWER\s*[:：]\s*", "", lines[-1], flags=re.I).strip().strip("*").strip()
    if not ans or NO_ANSWER_PAT.search(ans):
        return []
    if ans.startswith("["):
        try:
            v = ast.literal_eval(ans)
            if isinstance(v, list):
                return [str(x) for x in v]
        except (ValueError, SyntaxError):
            pass
    return [ans]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=str(DEFAULT_IN))
    args = ap.parse_args()
    data = json.loads(Path(args.answers).read_text())

    rows, agg = [], defaultdict(list)
    for qid, r in data.items():
        preds, golds = extract_pred(r.get("response")), [str(a) for a in r["answers"]]
        em = float({normalize(p) for p in preds} == {normalize(g) for g in golds})
        f1 = list_f1(preds, golds)
        qtype = r.get("qtype") or "?"
        hop = "multi-hop" if qtype.startswith("Compose") or "Compare" in qtype or "Intersect" in qtype else "single-hop"
        row = {"qid": qid, "pred": preds, "gold": golds, "em": em, "f1": round(f1, 3),
               "qtype": qtype, "hop": hop, "modalities": r.get("modalities"),
               "doc_hit_rate": r.get("doc_hit_rate"), "elapsed_s": r.get("elapsed_s"),
               "no_final_line": "FINAL ANSWER" not in (r.get("response") or "").upper()}
        rows.append(row)
        for key in ("ALL", f"qtype:{qtype}", f"hop:{hop}", *(f"mod:{m}" for m in (r.get("modalities") or []))):
            agg[key].append(row)

    def stat(rs):
        n = len(rs)
        return {"n": n, "em": round(100 * sum(x["em"] for x in rs) / n, 1),
                "f1": round(100 * sum(x["f1"] for x in rs) / n, 1),
                "dochit": round(100 * sum(x["doc_hit_rate"] or 0 for x in rs) / n, 1)}

    report = {k: stat(v) for k, v in sorted(agg.items())}
    out = Path(args.answers).with_name("m3d_smoke_scored.json")
    out.write_text(json.dumps({"summary": report, "rows": rows}, ensure_ascii=False, indent=1))

    print(f"{'bucket':<28}{'n':>4}{'EM':>8}{'F1':>8}{'找对书':>9}")
    for k, s in report.items():
        print(f"{k:<28}{s['n']:>4}{s['em']:>8}{s['f1']:>8}{s['dochit']:>9}")
    misses = [r for r in rows if r["f1"] < 0.5]
    print(f"\n低分题({len(misses)}):")
    for r in misses:
        print(f"  {r['qid'][:8]} {r['qtype']:<28} dochit={r['doc_hit_rate']:.0%} pred={str(r['pred'])[:50]!r} gold={str(r['gold'])[:50]!r}")
    print("\n->", out)


if __name__ == "__main__":
    main()
