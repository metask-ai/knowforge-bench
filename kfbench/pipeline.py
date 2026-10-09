"""The six-job pipeline: Locator -> bridge judge -> typed pool -> Deep Reader -> synthesizer -> format checker.

`answer()` runs one question.  `mode`:
  M5  two agents with disjoint tools, caps (loc_rounds, deep_rounds), bridge judge on   (paper's full pipeline)
  P1  M5 without the Deep Reader
  P0  one agent with all tools and loc_rounds + deep_rounds rounds (same judge / pool / synthesizer / checker)
  S4  one search with the question, then the synthesizer (one-shot search-and-answer baseline)
Round caps are upper bounds; the agents usually stop earlier.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from . import materials as M
from . import prompts as P
from .agent_loop import run_loop
from .kb_client import KB
from .llm import LLM
from .tools import LOCATOR_TOOLS, READER_TOOLS, registry

_FINAL_LINE_RE = re.compile(r"(?im)^\s*\**\s*FINAL ANSWER\s*[:：]\s*\**\s*(.*?)\s*\**\s*$")
_NO_HIT_CLAIM_RE = re.compile(r"知识库.{0,8}(未涉及|未收录|没有|不包含|无相关)|没有.{0,6}(图片|图表|资料)")
_FINAL_GUARD_MIN_CHARS = 280
_REFUSALS = ("none", "null", "n/a", "na", "not applicable", "no answer", "unknown", "not found", "not mentioned",
             "not answerable", "not answerable.", "unanswerable", "cannot be determined", "does not exist",
             "insufficient information.", "insufficient information")


def strip_guardrail(question: str) -> str:
    i = question.find("\n\nAnswer requirements:")
    return question if i < 0 else question[:i]


def final_guard(rnd: int, tool_calls: list, fin: Any) -> Optional[str]:
    if tool_calls:
        return None
    text = fin if isinstance(fin, str) else json.dumps(fin, ensure_ascii=False)
    if len(text) < _FINAL_GUARD_MIN_CHARS and not _NO_HIT_CLAIM_RE.search(text):
        return None
    return P.FINAL_GUARD_TEXT


def profile_text(ov: dict) -> str:
    """Scope 'expert' persona from the KB overview (name, goal, module list)."""
    name = ov.get("name") or "本知识库"
    goal = " ".join(str(ov.get("goal") or "").split())[:300]
    modules = [str(m.get("title")).strip() for m in (ov.get("modules") or []) if m.get("title")]
    lines = [f"你是知识库「{name}」的知识专家，回答严格依据该知识库的内容，超出其范围的如实说明。用用户提问所用的语言回答（英文问题用英文答）。"]
    if goal:
        lines.append(f"知识库主题：{goal}")
    if modules:
        lines.append("该知识库划分为以下知识模块（社区），可据此把握它覆盖的范围：\n" + "\n".join(f"- {t}" for t in modules))
    return "\n".join(lines)


class Pipeline:
    def __init__(self, kb: KB, llm: LLM, vision: Optional[LLM], bench: str, mode: str = "M5",
                 loc_rounds: int = 4, deep_rounds: int = 4, bridge: bool = True, material_cap: int = M.MATERIAL_CAP,
                 profile: Optional[str] = None, stage_llms: Optional[dict] = None):
        """`llm` is the default text model; `stage_llms` may override it per stage
        (locator / reader / bridge / synth / checker).  `vision` is only used by read_media and may be None."""
        assert bench in ("m3d", "mmlb", "mhr")
        assert mode in ("M5", "P0", "P1", "S4")
        self.kb, self.llm, self.vision, self.bench, self.mode = kb, llm, vision, bench, mode
        self.loc_rounds, self.deep_rounds, self.bridge, self.cap = loc_rounds, deep_rounds, bridge, material_cap
        self.profile = profile
        self.stage_llms = dict(stage_llms or {})

    def L(self, stage: str) -> LLM:
        return self.stage_llms.get(stage) or self.llm

    def _loop_fn(self, stage: str):
        llm = self.L(stage)

        async def fn(messages):
            return await llm.complete(messages, temperature=0.0, max_tokens=2048, retries=2)
        return fn

    async def _profile(self) -> str:
        if self.profile is None:
            try:
                self.profile = profile_text(await self.kb.a(self.kb.overview))
            except Exception:  # noqa: BLE001
                self.profile = profile_text({"name": (await self.kb.a(self.kb.card)).get("name", "")})
        return self.profile

    # -- judge / checker / synthesizer -----------------------------------------------------------------------------
    async def bridge_judge(self, question: str, materials: str, searched: list) -> dict:
        user = (f"Question: {strip_guardrail(question)}\n\nQueries already searched: {json.dumps(searched or [], ensure_ascii=False)}\n\n"
                f"Collected materials (may be truncated):\n{(materials or '')[:8000] or '(none)'}\n\nJSON:")
        try:
            raw = await self.L("bridge").complete([{"role": "system", "content": P.BRIDGE_SYSTEM}, {"role": "user", "content": user}],
                                                  temperature=0.0, max_tokens=300, retries=1)
            j = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            qs = [str(q).strip() for q in (j.get("queries") or []) if str(q).strip()]
            seen = {str(x).strip().lower() for x in (searched or [])}
            qs = [q for q in qs if q.lower() not in seen][:2]
            return {"need_more": bool(j.get("need_more")) and bool(qs), "queries": qs,
                    "bridge_entities": [str(x) for x in (j.get("bridge_entities") or [])][:3], "reason": str(j.get("reason") or "")[:200]}
        except Exception as exc:  # noqa: BLE001
            return {"need_more": False, "queries": [], "bridge_entities": [], "error": f"{type(exc).__name__}: {exc}"[:120]}

    async def final_check(self, question: str, response: str) -> tuple[str, dict]:
        m = None
        for m in _FINAL_LINE_RE.finditer(response or ""):
            pass
        before = m.group(1).strip() if m else ""
        try:
            raw = await self.L("checker").complete([{"role": "system", "content": P.FINAL_CHECKERS[self.bench]},
                                                    {"role": "user", "content": f"Question: {strip_guardrail(question)}\n\nAnswer text:\n{response}\n\nJSON:"}],
                                                   temperature=0.0, max_tokens=200, retries=1)
            j = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            fin = j.get("final")
            fin_s = str([str(x) for x in fin]) if isinstance(fin, list) else str(fin if fin is not None else "").strip()
            if self.bench != "m3d" and fin_s.strip("'\"").lower() in _REFUSALS:
                fin_s = "Insufficient information" if self.bench == "mhr" else "Not answerable"
        except Exception as exc:  # noqa: BLE001
            return response, {"before": before, "after": before, "error": f"{type(exc).__name__}: {exc}"[:120]}
        if not fin_s or fin_s == before:
            return response, {"before": before, "after": before}
        line = f"FINAL ANSWER: {fin_s}"
        new_resp = (response[:m.start()] + line + response[m.end():]) if m else ((response or "").rstrip() + "\n" + line)
        return new_resp, {"before": before, "after": fin_s}

    async def synthesize(self, question: str, materials: str, profile: str, note: str = "") -> str:
        system = f"{profile}\n\n{P.SYNTH_SYSTEM}"
        note_line = f"（系统提示：{note}，请如实向用户说明检索情况后按分级规则作答）\n\n" if note else ""
        user = (f"对话历史：\n（无历史）\n\n检索材料：\n{materials or '（没有检索到任何材料）'}\n\n"
                f"{note_line}用户问题：{question}\n\n直接以专家身份解答（不要写来源声明/开场白），按规则插入 [N] 与 [图N] 标记：")
        return await self.L("synth").complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                              temperature=0.3, max_tokens=2048, retries=2)

    # -- the pipeline ----------------------------------------------------------------------------------------------
    async def answer(self, question: str) -> dict:
        profile = await self._profile()
        builder = M.MaterialBuilder()
        ctx: dict[str, Any] = {"kb": self.kb, "builder": builder, "queries": [], "search_log": [], "vision": self.vision}
        events: list[dict] = []
        timing: dict[str, float] = {}
        t0 = time.time()

        def on_event(etype: str, payload: dict) -> None:
            if etype == "tool_start":
                timing.setdefault("first_event_s", round(time.time() - t0, 2))
                events.append({"name": payload.get("name", ""), "brief": payload.get("brief", "")})

        rule = {"m3d": P.M3D_RULE, "mhr": P.MHR_RULE, "mmlb": ""}[self.bench]
        loc_rules = P.LOC_RULES_V43.rstrip() + rule
        deep_rules = P.DEEP_RULES_V43.rstrip() + rule
        bridge: dict = {}
        r1 = r2 = None

        if self.mode == "S4":
            data = await self.kb.a(self.kb.search, question, 8, True)
            ctx["search_log"].append({"call": 0, "query": question, "data": data})
            ctx["queries"].append(question)
            events.append({"name": "search_knowledge", "brief": question[:60]})
            M.search_mat(builder, data)
        elif self.mode == "P0":
            names = LOCATOR_TOOLS + [n for n in READER_TOOLS if n not in LOCATOR_TOOLS]
            r1 = await run_loop(registry=registry(names), llm_fn=self._loop_fn("locator"),
                                system=f"{profile}\n\n{P.LOOP_RULES}{P.P0_RULES_V43.rstrip() + rule}",
                                task=f"当前问题：{question}", ctx=ctx, max_rounds=self.loc_rounds + self.deep_rounds,
                                on_event=on_event, site="p0", final_guard=final_guard, final_example=P.FINAL_EXAMPLE,
                                feedback_suffix=P.TASK_FEEDBACK_SUFFIX_V2)
        else:
            r1 = await run_loop(registry=registry(LOCATOR_TOOLS), llm_fn=self._loop_fn("locator"),
                                system=f"{profile}\n\n{P.LOOP_RULES}{loc_rules}",
                                task=f"当前问题：{question}", ctx=ctx, max_rounds=self.loc_rounds,
                                on_event=on_event, site="locator", final_guard=final_guard, final_example=P.FINAL_EXAMPLE,
                                feedback_suffix=P.LOC_SUFFIX)

        if self.mode != "S4":
            if self.bridge:
                pooled, _ = M.pool_materials(builder, ctx["search_log"], self.cap)
                bridge = await self.bridge_judge(question, pooled, ctx["queries"])
                for q2 in bridge.get("queries") or []:
                    events.append({"name": "search_knowledge", "brief": f"[bridge1] {q2[:60]}"})
                    try:
                        data = await self.kb.a(self.kb.search, q2, 8, True)
                        ctx["search_log"].append({"call": len(ctx["search_log"]), "query": q2, "data": data, "bridge": True})
                        ctx["queries"].append(q2)
                        M.search_mat(builder, data)
                    except Exception as exc:  # noqa: BLE001
                        bridge.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}"[:120])
            if self.mode == "M5":
                digest, fit_h = M.pool_materials(builder, ctx["search_log"], self.cap)
                bridge["handoff_fit"] = fit_h
                task2 = (f"当前问题：{question}\n\n定位阶段已收集的材料（出处 [N]/[图N] 标记沿用，"
                         f"配图行给的 fragment_id 可直接传给 read_media/fragment_context）：\n"
                         f"{digest or '（定位阶段没有拿到材料）'}\n\n请深读补全细节后收笔。")
                r2 = await run_loop(registry=registry(READER_TOOLS), llm_fn=self._loop_fn("reader"),
                                    system=f"{profile}\n\n{P.LOOP_RULES}{deep_rules}",
                                    task=task2, ctx=ctx, max_rounds=self.deep_rounds, on_event=on_event, site="reader",
                                    final_guard=None, final_example=P.FINAL_EXAMPLE, feedback_suffix=P.TASK_FEEDBACK_SUFFIX_V2)

        if not builder.has_content():
            try:
                data = await self.kb.a(self.kb.search, question, 8, True)
                ctx["search_log"].append({"call": len(ctx["search_log"]), "query": question, "data": data})
                M.search_mat(builder, data)
            except Exception:  # noqa: BLE001
                pass

        fit: dict = {}
        if builder.has_content():
            materials, fit = M.pool_materials(builder, ctx["search_log"], self.cap)
            note = ((r2 or r1).note if (r2 or r1) else "") or ""
            response = await self.synthesize(question, materials, profile, note)
            timing["ttft_s"] = round(time.time() - t0, 2)
        else:
            rl = r2 or r1
            fin = rl.final.strip() if (rl and isinstance(rl.final, str)) else ""
            response = fin or ((rl.note if rl else "") or "本次没有检索到可用材料")

        check: dict = {}
        if "FINAL ANSWER" in question:
            response, check = await self.final_check(question, response)

        meta = builder.meta()
        status = "s4" if self.mode == "S4" else f"{r1.status}+{r2.status if r2 else self.mode.lower()}"
        return {"response": response, "tool_events": events, "n_tool_calls": len(events), "status": status,
                "citations": meta["citations"], "images": meta["images"], "elapsed_s": round(time.time() - t0, 1),
                "timing": timing, "final_check": check, "search_log": ctx["search_log"], "bridge": bridge, "fit": fit,
                "read_media_calls": ctx.get("read_media_calls", 0),
                "deep_fids": sorted({f for fs in builder.block_meta.values() for f in fs if f and not str(f).startswith("doc:")}),
                "llm_calls": self.llm.calls + sum(l.calls for l in self.stage_llms.values() if l is not self.llm),
                "vision_enabled": self.vision is not None, "kb_calls": self.kb.calls}
