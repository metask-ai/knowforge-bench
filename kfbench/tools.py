"""Tool registry over the KB HTTP API.  Descriptions and parameter strings are verbatim from the deployed catalog:
the agents' behaviour depends on them.

Role tool sets (the physical isolation the paper is about):
  LOCATOR = search_knowledge, list_books, book_toc, list_media, find_figure, list_tables, query_table, get_table
  READER  = fragment_context, read_media, media_context, find_figure
  P0      = union of both (single agent)
`read_media` downloads the asset from the KB and asks the [vision] endpoint of *your* configuration.
"""
from __future__ import annotations

import asyncio
from typing import Any

from . import materials as M
from .agent_loop import ToolSpec
from .prompts import READ_PROMPT

LOCATOR_TOOLS = ["search_knowledge", "list_books", "book_toc", "list_media", "find_figure", "list_tables", "query_table", "get_table"]
READER_TOOLS = ["fragment_context", "read_media", "media_context", "find_figure"]
READ_MEDIA_MAX, READ_CONCURRENCY = 40, 4


def _kb(ctx):
    return ctx["kb"]


# ---- afns -----------------------------------------------------------------------------------------------------
async def search_afn(ctx: dict, args: dict) -> dict:
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query 不能为空")
    topk = int(args.get("topk") or 8)
    data = await _kb(ctx).a(_kb(ctx).search, query, topk, True)
    ctx.setdefault("queries", []).append(query)
    log = ctx.setdefault("search_log", [])
    log.append({"call": len(log), "query": query, "data": data})
    return data


async def fragment_afn(ctx, args):
    fid = str(args.get("fragment_id") or "").strip()
    if not fid:
        raise ValueError("fragment_id 不能为空")
    return await _kb(ctx).a(_kb(ctx).fragment_context, fid, int(args.get("window") or 1))


async def list_books_afn(ctx, args):
    data = await _kb(ctx).a(_kb(ctx).sources)
    books = [{"name": str(s.get("name") or ""), "root_node_id": str(s.get("root_node_id") or ""),
              "one_line": str(s.get("one_line") or "")} for s in (data.get("sources") or [])]
    total = len(books)
    keyword = str(args.get("keyword") or "").strip()
    if keyword:
        kw = keyword.lower()
        books = [b for b in books if kw in b["name"].lower()]
    return {"total": total, "matched": len(books), "keyword": keyword, "books": books[:M.BOOKS_CAP], "truncated": len(books) > M.BOOKS_CAP}


async def book_toc_afn(ctx, args):
    name = str(args.get("book") or "").strip()
    if not name:
        raise ValueError("book 不能为空（书名或其中足够独特的片段）")
    return await _kb(ctx).a(_kb(ctx).book_toc, name)


async def list_tables_afn(ctx, args):
    return await _kb(ctx).a(_kb(ctx).list_tables)


async def query_table_afn(ctx, args):
    item = str(args.get("item") or "").strip()
    if not item:
        raise ValueError("item 不能为空")
    return await _kb(ctx).a(_kb(ctx).query_table, item, str(args.get("table_id") or ""))


async def get_table_afn(ctx, args):
    tid = str(args.get("table_id") or "").strip()
    if not tid:
        raise ValueError("table_id 不能为空（先 list_tables 拿 id）")
    return await _kb(ctx).a(_kb(ctx).get_table, tid)


async def list_media_afn(ctx, args):
    return await _kb(ctx).a(_kb(ctx).list_media, str(args.get("book") or ""), str(args.get("kind") or "all"),
                            str(args.get("page") or ""), str(args.get("heading") or args.get("section") or ""))


async def find_figure_afn(ctx, args):
    return await _kb(ctx).a(_kb(ctx).find_figure, str(args.get("label") or ""), str(args.get("book") or ""))


async def media_context_afn(ctx, args):
    fid = str(args.get("fragment_id") or "").strip()
    if not fid:
        raise ValueError("fragment_id 不能为空")
    return await _kb(ctx).a(_kb(ctx).media_context, fid)


def _short_answer(answer: str) -> str:
    for line in reversed((answer or "").splitlines()):
        s = line.strip()
        if s.startswith(("答案：", "答案:")):
            return s.split("：", 1)[-1].split(":", 1)[-1].strip() if "：" in s or ":" in s else s
    return (answer or "").strip()


async def read_media_afn(ctx, args):
    """Download each image/table asset from the KB and ask the vision model, one image per call."""
    fids = args.get("fragment_ids")
    if isinstance(fids, str):
        fids = [x.strip() for x in fids.replace("，", ",").split(",") if x.strip()]
    if not fids and args.get("fragment_id"):
        fids = [args["fragment_id"]]
    ids = [str(f).strip() for f in (fids or []) if str(f).strip()]
    question = str(args.get("question") or "").strip()
    if not ids:
        raise ValueError("fragment_ids 不能为空")
    if not question:
        raise ValueError("question 不能为空")
    vision = ctx.get("vision")
    if vision is None:
        return {"question": question, "answers": [{"fragment_id": f, "error": "read_media 未启用（配置 [vision]）"} for f in ids]}
    kb = _kb(ctx)
    sem = asyncio.Semaphore(READ_CONCURRENCY)

    async def one(fid: str) -> dict:
        try:
            data, mime, name = await kb.a(kb.download, fid)
        except Exception as exc:  # noqa: BLE001
            return {"fragment_id": fid, "error": f"不是图片/表格片段，或原件不存在（{str(exc)[:120]}）"}
        if not mime.startswith("image/"):
            return {"fragment_id": fid, "error": f"原件不是图片（{mime}），无法读图"}
        try:
            async with sem:
                ans = await vision.complete(vision.image_message(READ_PROMPT.format(question=question), data, mime),
                                            temperature=0.0, max_tokens=1024, retries=1)
        except Exception as exc:  # noqa: BLE001
            return {"fragment_id": fid, "kind": "image", "name": name, "error": f"读图失败：{str(exc)[:200]}"}
        kind = "table" if "table" in (name or "").lower() else "image"
        return {"fragment_id": fid, "kind": kind, "name": name, "path": f"asset:{fid}", "answer": ans, "short": _short_answer(ans)}

    answers = list(await asyncio.gather(*(one(f) for f in ids[:READ_MEDIA_MAX])))
    ctx["read_media_calls"] = ctx.get("read_media_calls", 0) + len(answers)
    out: dict[str, Any] = {"question": question, "answers": answers}
    if ids[READ_MEDIA_MAX:]:
        out["skipped"] = ids[READ_MEDIA_MAX:]
    return out


# ---- registry -------------------------------------------------------------------------------------------------
def _mat(fn):
    return lambda ctx, args, data: fn(ctx["builder"], data)


def _mat_search(ctx, args, data):
    return M.search_mat(ctx["builder"], data)


SPECS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("search_knowledge",
             "智能搜索知识库（向量+关键字+图扩展），返回正文片段和相关图表，打底检索先用它。一次没搜到就换关键词再搜。",
             '{"query": "关键词", "topk": 8}', search_afn, _mat_search),
    ToolSpec("fragment_context",
             "看 search 命中片段的前后相邻原文，补足上下文（用 search 材料里没展开的细节时用）。"
             "也用于读图/表的完整描述：相关配图的说明是截断的，答案在图表里时传配图行给的 fragment_id。",
             '{"fragment_id": "search 返回的片段 id", "window": 1}', fragment_afn, _mat(M.fragment_mat)),
    ToolSpec("list_books",
             "列出本知识库收录的资料书目（书名+一句话简介）。用户问'库里有哪些书/资料/文献'、或要按某本书定位内容时先用它；书太多会截断，可传 keyword 按书名过滤。",
             '{"keyword": "可选，按书名过滤"}', list_books_afn, _mat(M.list_books_mat)),
    ToolSpec("book_toc",
             "看某一本书的标题目录（章节树），了解这本书的内容结构。用户问'某本书讲了什么/有哪些章节'时用它；book 传书名（list_books 里的名字，或其中足够独特的片段），要具体内容细节再用 search_knowledge。",
             '{"book": "书名或书名关键词"}', book_toc_afn, _mat(M.book_toc_mat)),
    ToolSpec("list_tables", "列出本库可精确逐项直查的数据表（名称+列头+行数）。要查精确值前先看有哪些表。", '{}',
             list_tables_afn, _mat(M.list_tables_mat)),
    ToolSpec("query_table",
             "在数据表里按条目关键词精确查行（子串匹配），返回真实精确值。要某型号/产品的精确价格、规格、参数、编码时用它——search 会漏低排名行，精确数值必须以此为准。",
             '{"item": "条目关键词（产品名/型号）", "table_id": "可选，限定某张表"}', query_table_afn, _mat(M.query_table_mat)),
    ToolSpec("get_table", "取整张数据表（列头+行）。只用于'列出全部/汇总/算总价'这类要遍历整表的问题，查单个条目用 query_table。",
             '{"table_id": "list_tables 返回的表 id"}', get_table_afn, _mat(M.get_table_mat)),
    ToolSpec("read_media",
             "把问题和已找到的图/表交给多模态模型**现场看图作答**。只在文字材料不够用时才用："
             "要数图里有几个/几根、看颜色/位置/表情、读被裁掉标题的图表数值、核对表格某一格。"
             "先 search 找到配图，把配图行给的 fragment_id（可多个，一次 ≤40，并发读）和一句明确的问题传进来；"
             "**数全书有几张满足条件的图/表**：先 list_media 拿清单，把清单里全部 fragment_id 一次传入，"
             "问题写成是非判定（如'这张图是否比较了公众与拉丁裔？答是/否'），按逐图结果自己数。"
             "不要用它代替 search 找图，也不要每题都看图。",
             '{"fragment_ids": ["配图行给的 fragment_id", "…"], "question": "要从图里读出什么"}', read_media_afn, _mat(M.read_media_mat)),
    ToolSpec("list_media",
             "列出一本书**全部**图/表片段（按页码，每条一句描述 + fragment_id）。问'这份文档/报告里有几张图/几张表/几张照片、哪些页有图表'这类**全文枚举、计数**问题必须用它——search 只见top-k，数出来一定偏小。单书库可不传 book；可传 kind=image|table、page 只看某页；传 heading（章节/标题名，book_toc 或材料出处里的标题）只列该标题下的图/表——标题树里找不到会自动回退整本并说明。",
             '{"book": "可选，书名或关键词", "heading": "可选，章节标题", "kind": "可选 image|table|all", "page": "可选 页码"}',
             list_media_afn, _mat(M.list_media_mat)),
    ToolSpec("find_figure",
             "题目点名 'Figure 5' / 'Table 3' / '图5' 这种**编号**时用它：在正文里找该图题/表题所在页，并列出该页的图/表片段（图片片段本身没有编号，编号只在正文图题里）。拿到候选 fragment_id 后再 read_media 看图或 fragment_context 读描述。",
             '{"label": "Figure 5 / Table 3 / 图5", "book": "可选"}', find_figure_afn, _mat(M.find_figure_mat)),
    ToolSpec("media_context",
             "看某张图/表**所在页的正文与图题**（fragment_context 的相邻段对图片来说是别的图，不是图题）。读图前想知道这张图是讲什么、图题怎么写、同页还有哪些图时用；传 search/list_media 给的图/表 fragment_id。",
             '{"fragment_id": "图/表片段 id"}', media_context_afn, _mat(M.media_context_mat)),
]}


def registry(names: list[str]) -> list[ToolSpec]:
    return [SPECS[n] for n in names]
