"""Materials: how tool results become the text the agents read, and the typed, size-bounded evidence pool.

Port of the deployed system's MaterialBuilder / materializers / `_pool_materials`, with two deliberate
differences that do not affect the pipeline's decisions:
  * image de-duplication is by path only (the deployed system also uses a perceptual hash of the local file);
  * table assets are referenced by their path and fetched through the KB `download` endpoint, which returns the
    rendered PNG (the deployed system renders the PDF crop locally).
The block formats ("【材料N】…", "【相关图表】…") are kept verbatim: the synthesizer prompt was tuned on them.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

RELEVANCE_GATE = 2.0          # hybrid score (0-10) below which a search block is prefixed as weakly relevant
MATERIAL_CAP = int(os.getenv("KFBENCH_MATERIAL_CAP") or 30000)
SUB_CAPS_DEFAULT = {"image": 3000, "table": 3000, "example": 2000}
BOOKS_CAP, BOOK_ONELINE_CLIP, TOC_CAP = 80, 60, 200


class MaterialBuilder:
    def __init__(self):
        self.citations: dict[str, Any] = {}   # "1" -> {fragment_id, source_file, source_path, label}
        self.images: dict[str, Any] = {}      # "图1" -> {path, caption}
        self.blocks: list[str] = []
        self.block_meta: dict[int, list[str]] = {}
        self.plain_fids: dict[str, list[str]] = {}
        self._seen_frag: set[str] = set()
        self._seen_path: set[str] = set()

    @staticmethod
    def _page_tag(path: str) -> str:
        m = re.search(r"page_(\d+)_", os.path.basename(path or ""))
        return f"（第{m.group(1)}页）" if m else ""

    def _emit_image(self, path: str, caption: str) -> Optional[str]:
        if not path or path in self._seen_path:
            return None
        self._seen_path.add(path)
        n = f"图{len(self.images) + 1}"
        self.images[n] = {"path": path, "caption": caption}
        return n

    def _emit_media(self, kind: str, path: str, ext: str, caption: str) -> Optional[tuple[str, str]]:
        ext = (ext or "").lower()
        if kind == "image":
            n = self._emit_image(path, caption)
            return (n, caption or "（无描述）") if n else None
        if kind == "table" and ext == "pdf":
            n = self._emit_image(path, f"{caption}（表格）" if caption else "（表格）")
            if n:
                return (n, caption or "（表格）")
        return None

    def add_search(self, data: dict[str, Any]) -> str:
        results = data.get("results") or []
        hit_media = (data.get("images") or []) + (data.get("tables") or [])
        examples = data.get("examples") or []
        lines: list[str] = []
        for r in results:
            frag = str(r.get("fragment_id") or "")
            if frag and frag in self._seen_frag:
                continue
            if frag:
                self._seen_frag.add(frag)
            cid = str(len(self.citations) + 1)
            label = (r.get("section_title") or r.get("source_file") or "").strip()
            tag = "（相关知识）" if r.get("from_graph") else ""
            self.citations[cid] = {"fragment_id": r.get("fragment_id"), "source_file": r.get("source_file"),
                                   "source_path": r.get("source_path"), "label": label}
            text = " ".join(str(r.get("text") or "").split())[:700]
            block = [f"【材料{cid}】出处标记 [{cid}]（{label or r.get('source_file') or '未知来源'}{tag}）：{text}"]
            mp = (r.get("media_path") or "").strip()
            if mp and r.get("origin_kind") in ("image", "video"):
                is_video = r.get("origin_kind") == "video"
                mcap = f"{label or r.get('source_file') or ''}（{'视频' if is_video else '图片'}原件）"
                if is_video:
                    if not any(im.get("path") == mp for im in self.images.values()):
                        n = f"图{len(self.images) + 1}"
                        self.images[n] = {"path": mp, "caption": mcap, "cite": str(cid)}
                        block.append(f"  · 本片段源自视频文件，原件可展示 [{n}]")
                else:
                    emitted = self._emit_image(mp, mcap)
                    if emitted:
                        block.append(f"  · 本片段源自图片文件，原图可展示 [{emitted}]")
            for c in (r.get("context") or []):
                if c.get("anchor"):
                    continue
                kind = c.get("kind")
                caption = " ".join(str(c.get("caption") or "").split())[:160]
                path = c.get("path") or ""
                ext = (c.get("ext") or "").lower()
                if kind == "text":
                    seg = " ".join(str(c.get("text") or "").split())
                    if seg:
                        block.append(f"  · 邻近上下文：{seg}")
                elif kind == "table" and ext == "csv":
                    if caption:
                        block.append(f"  · 配套数据表（含完整条目/价格，需具体数值时可用 query_table 精确查询）：{caption}")
                else:
                    emitted = self._emit_media(kind, path, ext, caption)
                    if emitted:
                        cfid = str(c.get("fragment_id") or "")
                        cfid_note = f"（完整描述用 fragment_context 读 {cfid}）" if cfid else ""
                        block.append(f"  · 可配图 [{emitted[0]}]{self._page_tag(path)}{cfid_note}：{emitted[1]}")
                    elif kind == "table" and caption:
                        block.append(f"  · 相关表格（暂无法配图，仅描述）：{caption}")
            lines.append("\n".join(block))
        hit_lines: list[str] = []
        for m in hit_media:
            kind = m.get("kind")
            caption = " ".join(str(m.get("caption") or "").split())[:160]
            ext = (m.get("ext") or "").lower()
            if kind == "table" and ext == "csv":
                if caption:
                    hit_lines.append(f"  · 配套数据表（含完整条目/价格，需具体数值时可用 query_table 精确查询）：{caption}")
                continue
            emitted = self._emit_media(kind, m.get("path") or "", ext, caption)
            if emitted:
                fid = str(m.get("fragment_id") or "")
                fid_note = f"（完整描述用 fragment_context 读 {fid}）" if fid else ""
                hit_lines.append(f"  · 相关配图 [{emitted[0]}]{self._page_tag(m.get('path') or '')}{fid_note}：{emitted[1]}")
        if hit_lines:
            lines.append("【相关图表】（与问题主题相关，可按需配图；说明文字是截断的，"
                         "若答案可能在图/表内容里，先用 fragment_context 读完整描述再作答）：\n" + "\n".join(hit_lines))
        ex_lines = [f"  · {' '.join(str(e.get('text') or '').split())[:300]}" for e in examples if (e.get("text") or "").strip()]
        if ex_lines:
            lines.append("【相关示例】：\n" + "\n".join(ex_lines))
        if not lines:
            return ""
        block_text = "\n\n".join(lines)
        self.blocks.append(block_text)
        return block_text

    def add_table_rows(self, item: str, results: list[dict[str, Any]]) -> str:
        tbl_lines: list[str] = []
        for tr in results:
            rows = tr.get("rows") or []
            if not rows:
                continue
            more = f"（另有 {tr.get('matched', len(rows)) - len(rows)} 行未列出）" if tr.get("truncated") else ""
            tbl_lines.append(f"  数据表「{tr.get('name')}」匹配「{item}」命中 {tr.get('matched')} 行{more}：")
            for row in rows[:20]:
                tbl_lines.append("    - " + "，".join(f"{k}：{v}" for k, v in row.items()))
        if not tbl_lines:
            return ""
        block_text = ("【数据表精确查询结果】（来自 csv 逐项直查，是真实精确值，可直接引用；表内没有的不要编）：\n" + "\n".join(tbl_lines))
        self.blocks.append(block_text)
        self.block_meta[len(self.blocks) - 1] = [f"doc:{tr.get('name')}" for tr in (results or []) if isinstance(tr, dict) and tr.get("name")]
        return block_text

    def add_plain(self, block_text: str, fragment_ids: Optional[list] = None) -> str:
        if block_text:
            self.blocks.append(block_text)
            self.block_meta[len(self.blocks) - 1] = [str(x) for x in (fragment_ids or []) if x]
        return block_text

    def materials(self) -> str:
        return "\n\n".join(self.blocks)

    def has_content(self) -> bool:
        return bool(self.blocks)

    def meta(self) -> dict[str, Any]:
        return {"citations": self.citations, "images": self.images}


# ---------------------------------------------------------------------------------------------------------------
# Materializers (tool result -> text fed back to the agent).  Verbatim formats.
# ---------------------------------------------------------------------------------------------------------------
def top_score(data: dict[str, Any]) -> float:
    all_scores = ([r.get("score", 0.0) or 0.0 for r in (data.get("results") or [])]
                  + [m.get("score", 0.0) or 0.0 for m in (data.get("images") or [])]
                  + [m.get("score", 0.0) or 0.0 for m in (data.get("tables") or [])]
                  + [e.get("score", 0.0) or 0.0 for e in (data.get("examples") or [])])
    return max(float(x) for x in all_scores) if all_scores else 0.0


def search_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    top = top_score(data)
    block = builder.add_search(data)
    if not block:
        return "（本次检索没有返回新材料——换个关键词，或需要精确数值就改用 query_table）"
    return ("（相关度弱，可能不相关）\n" if top < RELEVANCE_GATE else "") + block


def fragment_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    import json
    if data.get("error"):
        return json.dumps({"error": {"code": data["error"], "recoverable": True}}, ensure_ascii=False)
    parts = []
    for f in (data.get("context") or []):
        mark = "→ " if f.get("anchor") else "  "
        is_media = f.get("kind") in ("image", "table")
        cap = None if (f.get("anchor") and is_media) else (1500 if f.get("anchor") else 500)
        seg = " ".join(str(f.get("text") or "").split())
        if cap:
            seg = seg[:cap]
        line = f"{mark}{seg}"
        if is_media and f.get("anchor") and f.get("media_path"):
            line += f"\n{mark}（此段为图/表的多模态完整描述，原件：{f['media_path']}）"
        parts.append(line)
    block = "【片段上下文】（→ 为命中段，上下为相邻原文，引用出处仍用原 [N] 标记）：\n" + "\n".join(parts)
    return builder.add_plain(block, [f.get("fragment_id") for f in (data.get("context") or []) if f.get("fragment_id")])


def list_tables_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    tables = data.get("tables") or []
    if not tables:
        return "（本库没有可精确直查的数据表）"
    return "可直查的数据表：\n" + "\n".join(
        f"- {t.get('name')}（table_id={t.get('table_id')}，{t.get('row_count')} 行）：列[{'、'.join(t.get('columns') or [])}]" for t in tables)


def query_table_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    block = builder.add_table_rows(str(data.get("item") or ""), data.get("results") or [])
    return block or "（数据表里没有匹配该关键词的行——换更短的产品名/型号关键词试试）"


def get_table_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    import json
    if not data:
        return json.dumps({"error": {"code": "table_not_found", "recoverable": True}}, ensure_ascii=False)
    rows = data.get("rows") or []
    more = f"（另有 {data.get('row_count', len(rows)) - len(rows)} 行未列出）" if data.get("truncated") else ""
    lines = [f"整表「{data.get('name')}」共 {data.get('row_count')} 行{more}（真实精确值，可直接引用）："]
    for row in rows:
        lines.append("  - " + "，".join(f"{k}：{v}" for k, v in row.items()))
    return builder.add_plain("\n".join(lines), [f"doc:{data.get('name')}"] if data.get("name") else None)


def list_books_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    total = int(data.get("total") or 0)
    books = data.get("books") or []
    if not total:
        return "（本知识库还没有收录任何资料）"
    if not books:
        return f"（共 {total} 本资料，但书名都不含「{data.get('keyword')}」——换更短的关键词，或不带 keyword 看全量头部）"
    head = f"【书目】本知识库共收录 {total} 本资料"
    if data.get("keyword"):
        head += f"，书名含「{data['keyword']}」的 {data.get('matched')} 本"
    if data.get("truncated"):
        head += f"；数量太多只列出前 {len(books)} 本，可用 keyword 按书名过滤缩小"
    lines = [head + "："]
    for b in books:
        one = " ".join(b["one_line"].split())[:BOOK_ONELINE_CLIP]
        lines.append(f"- {b['name']}" + (f"：{one}" if one else ""))
    return builder.add_plain("\n".join(lines))


def book_toc_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    import json
    if data.get("error"):
        return json.dumps({"error": {"code": data["error"], "recoverable": True, "hint": "没有书名匹配该关键词，先用 list_books 看书目"}}, ensure_ascii=False)
    if data.get("candidates"):
        return f"匹配到 {data.get('matched')} 本书，用更完整的书名再调一次 book_toc：\n" + "\n".join(f"- {n}" for n in data["candidates"])
    secs = (data.get("toc") or {}).get("sections") or []
    lines = [f"【目录】《{data.get('book')}》共 {len(secs)} 个标题："]
    for s in secs[:TOC_CAP]:
        lines.append("  " * max(0, int(s.get("level") or 1) - 1) + "- " + str(s.get("title") or ""))
    if len(secs) > TOC_CAP:
        lines.append(f"（目录太长，另有 {len(secs) - TOC_CAP} 个标题未列出）")
    return builder.add_plain("\n".join(lines), [f"doc:{data.get('book')}"] if data.get("book") else None)


def list_media_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    err = data.get("error")
    if err in ("book_required", "ambiguous"):
        if data.get("candidates"):
            return "（本库有多本资料，请传 book 指定哪一本；候选：" + "；".join(data["candidates"]) + "）"
        return (f"（本库有 {data.get('matched')} 本资料，必须传 book：先用 search_knowledge 找到答案所在的书"
                "（材料出处括号里的书名），再把书名传进来）")
    if err:
        return f"（list_media 未找到：{err}）"
    span = data.get("span")
    if span:
        if span.get("mode") == "text_anchor":
            nxt = (f"，标题树里没有这个标题，按正文文字命中第 {', '.join(str(p) for p in span['hit_pages'])} 页，"
                   f"从第 {span['start']} 页起到书末")
        else:
            nxt = f"，止于同级标题「{span['next_title']}」" if span.get("next_title") else "，到全书末"
        head = (f"【标题下图表清单】《{data['book']}》标题「{span['title']}」（第 {span['start']}–{span['end']} 页{nxt}）"
                f"下的图/表 {len(data.get('items') or [])} 条（全书共 {data['total_images']} 图 {data['total_tables']} 表；"
                f"首末页可能跨节，标 ⚠边界 的自行判断是否属于本节）")
    else:
        head = (f"【全书图表清单】《{data['book']}》全书共 {data['total_images']} 张图、{data['total_tables']} 张表"
                + (f"（标题「{data['heading']}」标题树和正文里都没找到，已回退为整本）" if data.get("heading_missed") else "")
                + (f"（只列第 {data['page']} 页）" if data.get("page") not in (None, "") else ""))
    head += (f"；数量太多只列前 {data['shown']} 条，可按 page 分页看" if data.get("truncated") else "") + "："
    lines = [head]
    for it in data.get("items") or []:
        k = "图" if it["kind"] == "image" else "表"
        pg = f"第{it['page']}页 " if it["page"] is not None else ""
        bd = "⚠边界 " if it.get("boundary") else ""
        lines.append(f"- {pg}{bd}{k} {it['fragment_id']}：{it['desc'] or '（无描述）'}")
    lines.append("（要看某张的细节：fragment_id 传 read_media 现场看图 / fragment_context 读完整描述 / media_context 看它的图题与同页正文。"
                 "**数有几张满足条件**：能从上面描述判定的直接数；判不了的把全部 fragment_id 一次传给 read_media（≤40，并发）问是/否，再数）")
    return builder.add_plain("\n".join(lines), [it.get("fragment_id") for it in (data.get("items") or [])])


def find_figure_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    err = data.get("error")
    if err == "bad_label":
        return f"（label 格式不对：请传 'Figure 5' / 'Table 3' / '图5' 这种形式，收到 {data.get('label')!r}）"
    if err in ("book_required", "ambiguous"):
        if data.get("candidates"):
            return "（本库有多本资料，请传 book 指定哪一本；候选：" + "；".join(data["candidates"]) + "）"
        return (f"（本库有 {data.get('matched')} 本资料，必须传 book：先用 search_knowledge 找到该图所在的书"
                "（材料出处括号里的书名），再把书名传进来）")
    if err:
        return f"（find_figure 未找到：{err}）"
    lab = data["label"]
    if not data.get("pages"):
        return builder.add_plain(f"【图号定位】《{data['book']}》正文里没有出现「{lab}」字样——图题可能没进文字层。"
                                 f"改用 list_media 按页浏览全书图表，或用 search_knowledge 搜该图的主题词。")
    lines = [f"【图号定位】「{lab}」在《{data['book']}》："]
    for c in data.get("captions") or []:
        lines.append(f"- 图题（第{c['page']}页）：{c['line']}")
    if not data.get("captions"):
        for m_ in data.get("mentions") or []:
            lines.append(f"- 正文提及（第{m_['page']}页）：…{m_['line']}…")
    if data.get("media"):
        lines.append(f"该页面上的图/表（{lab} 应是其中之一，按描述对照选择，再传 read_media 看图）：")
        for it in data["media"]:
            k = "图" if it["kind"] == "image" else "表"
            lines.append(f"  - 第{it['page']}页 {k} {it['fragment_id']}：{it['desc'] or '（无描述）'}")
    else:
        lines.append(f"（第 {', '.join(str(p) for p in data['pages'])} 页没有单独切出的图/表片段——图可能与正文合并；用 fragment_context 读图题所在片段的前后文）")
    return builder.add_plain("\n".join(lines), [it.get("fragment_id") for it in (data.get("media") or [])]
                             + [c.get("fragment_id") for c in (data.get("captions") or []) if isinstance(c, dict)])


def media_context_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    err = data.get("error")
    if err:
        return f"（media_context：{err}——它只接受 search/list_media 给的图/表 fragment_id）"
    k = "图" if data["kind"] == "image" else "表"
    lines = [f"【图片上下文】{k} {data['fragment_id']}（第{data['page']}页）", f"- 该{k}的完整描述：{data['desc'] or '（无描述）'}"]
    if data.get("captions"):
        lines.append("- 同页疑似图题/表题：" + " ｜ ".join(data["captions"]))
    if data.get("page_text"):
        lines.append(f"- 同页正文：{data['page_text']}")
    if data.get("siblings"):
        lines.append("- 同页其它图/表：" + "；".join(f"{'图' if s['kind']=='image' else '表'} {s['fragment_id']}：{s['desc']}" for s in data["siblings"]))
    lines.append("（要读图中数值/颜色/数量，把 fragment_id 传给 read_media）")
    return builder.add_plain("\n".join(lines), [data.get("fragment_id")] + [s_.get("fragment_id") for s_ in (data.get("siblings") or [])])


def read_media_mat(builder: MaterialBuilder, data: dict[str, Any]) -> str:
    lines: list[str] = []
    for r in data.get("answers") or []:
        if r.get("error"):
            lines.append(f"  · {r.get('fragment_id')}：{r['error']}")
            continue
        path = r.get("path") or ""
        tag = next((n for n, im in builder.images.items() if (im.get("path") or "") == path), None)
        if tag is None:
            tag = builder._emit_image(path or f"fragment:{r.get('fragment_id')}", f"{r.get('name') or ''}（{r.get('kind')}）") or "图?"
        lines.append(f"  · [{tag}]（{r.get('kind')}）读图结果：{' '.join(str(r.get('answer') or '').split())[:1200]}")
    if data.get("skipped"):
        lines.append(f"  · 超出单次上限未读：{', '.join(data['skipped'])}")
    if not lines:
        return "（read_media 没有返回任何结果）"
    block = ("【读图结果】（多模态模型按你的问题现场看图得到的，以此为准；引用时用对应 [图N]；"
             "若答'图中没有'，换别的图或如实说明）：\n" + "\n".join(lines))
    return builder.add_plain(block, [r.get("fragment_id") for r in (data.get("answers") or [])])


# ---------------------------------------------------------------------------------------------------------------
# Typed evidence pool (30K cap; text / image / table / example ranked separately)
# ---------------------------------------------------------------------------------------------------------------
_MAT_SPLIT_RE = re.compile(r"(?=^【材料(\d+)】)|(?=^【相关图表】)|(?=^【相关示例】)", re.M)
_FID_RE = re.compile(r"fragment_context 读 (frag_[0-9a-f]+)")


def split_units(builder: MaterialBuilder) -> list[tuple[str, str]]:
    units, p, g, e = [], 0, 0, 0
    builder.plain_fids.clear()
    for bi, b in enumerate(builder.blocks):
        parts = [x for x in _MAT_SPLIT_RE.split(b) if x and not x.isdigit()]
        got = False
        for part in parts:
            m = re.match(r"【材料(\d+)】", part)
            if m:
                units.append((m.group(1), part.rstrip())); got = True
            elif part.startswith("【相关图表】"):
                lines_ = [ln for ln in part.rstrip().split("\n") if ln.strip()]
                header, items = lines_[0], lines_[1:]
                for j, ln in enumerate(items):
                    g += 1; units.append((f"G{g}", (header + "\n" + ln) if j == 0 else ln)); got = True
                if not items:
                    g += 1; units.append((f"G{g}", header)); got = True
            elif part.startswith("【相关示例】"):
                lines_ = [ln for ln in part.rstrip().split("\n") if ln.strip()]
                header, items = lines_[0], lines_[1:]
                for j, ln in enumerate(items):
                    e += 1; units.append((f"E{e}", (header + "\n" + ln) if j == 0 else ln)); got = True
                if not items:
                    e += 1; units.append((f"E{e}", header)); got = True
        if not got:
            p += 1
            units.append((f"P{p}", b.rstrip()))
            builder.plain_fids[f"P{p}"] = list(builder.block_meta.get(bi) or [])
    return units


def pool_materials(builder: MaterialBuilder, search_log: list, cap: int = MATERIAL_CAP, protect_n: int = 3,
                   sub_caps: dict | None = None) -> tuple[str, dict]:
    """Merge every search's ranked list into one pool.  Within a search the top-`protect_n` hits of each type are
    protected; across searches blocks compare by vector cosine (the only cross-query comparable score); when a
    sub-pool is full the lowest unprotected block is evicted.  Expansion blocks inherit their anchor's score;
    blocks produced by reading tools inherit their source fragment's score (unknown source = protected)."""
    units = split_units(builder)
    full = "\n\n".join(t for _, t in units)
    stats: dict[str, Any] = {"total_chars": len(full), "units": len(units), "cap": cap}
    cits = builder.citations
    if len(full) <= cap:
        stats.update(kept_chars=len(full), dropped=0,
                     kept_fids=sorted({(cits.get(u) or {}).get("fragment_id") for u, _t in units if u in cits} - {None}
                                      | {f for u, _t in units if u.startswith("P") for f in builder.plain_fids.get(u, [])}))
        return full, stats
    info: dict[str, dict] = {}
    for e in search_log or []:
        data = e.get("data") or {}
        for key, kind in (("results", "text"), ("images", "image"), ("tables", "table"), ("examples", "example")):
            ranked = sorted(data.get(key) or [], key=lambda it: -(it.get("raw_score") or 0.0))
            for r, it in enumerate(ranked, 1):
                fid = it.get("fragment_id")
                if not fid:
                    continue
                anc = it.get("anchor_fragment_id")
                base = info.get(anc) if anc else None
                vec = float(it.get("vec_score") or 0.0)
                prot = r <= protect_n
                if base:
                    vec, prot = base["vec"], base["prot"]
                k = it.get("kind") or kind
                cur = info.get(fid)
                if cur is None or vec > cur["vec"]:
                    info[fid] = {"vec": vec, "prot": prot or bool(cur and cur["prot"]), "kind": k}
    known = sorted(d["vec"] for d in info.values())
    median = known[len(known) // 2] if known else 0.0
    caps = dict(sub_caps or SUB_CAPS_DEFAULT)
    caps["text"] = max(4000, cap - sum(caps.values()))
    pools: dict[str, list] = {k: [] for k in caps}
    used: dict[str, int] = {k: 0 for k in caps}
    dropped = 0
    for idx, (uid, text) in enumerate(units):
        if uid.startswith("P"):
            srcs = [f for f in builder.plain_fids.get(uid, []) if f in info]
            vec = max((info[f]["vec"] for f in srcs), default=1.0)
            prot = any(info[f]["prot"] for f in srcs) if srcs else True
            kind = "text"
        elif uid.startswith("G"):
            m = _FID_RE.search(text)
            d = info.get(m.group(1)) if m else None
            kind = (d or {}).get("kind") or "image"
            if kind not in caps:
                kind = "image"
            vec, prot = ((d or {}).get("vec", 0.0)), ((d or {}).get("prot", False))
        elif uid.startswith("E"):
            kind, vec, prot = "example", median, False
        else:
            fid = (cits.get(uid) or {}).get("fragment_id")
            d = info.get(fid) or {"vec": 0.0, "prot": False}
            kind, vec, prot = "text", d["vec"], d["prot"]
        pools[kind].append([idx, uid, text, vec, prot]); used[kind] += len(text) + 2
        while used[kind] > caps[kind] and pools[kind]:
            cands = [b for b in pools[kind] if not b[4]] or pools[kind]
            victim = min(cands, key=lambda b: (b[3], -b[0]))
            pools[kind].remove(victim); used[kind] -= len(victim[2]) + 2; dropped += 1
    pool = sorted([b for ps in pools.values() for b in ps], key=lambda b: b[0])
    kept_fids = []
    for b in pool:
        if b[1].startswith("P"):
            kept_fids.extend(builder.plain_fids.get(b[1], []))
        else:
            f = (cits.get(b[1]) or {}).get("fragment_id")
            if f:
                kept_fids.append(f)
    stats.update(kept_chars=sum(used.values()), dropped=dropped, sub_used=dict(used), kept_fids=sorted(set(kept_fids)),
                 min_kept_vec=round(min([b[3] for b in pool if not b[4]], default=0.0), 3))
    return "\n\n".join(b[2] for b in pool), stats
