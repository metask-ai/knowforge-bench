#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SPEC22 —— 站点无关的有界 agent loop 内核（提示词 JSON 协议）。

把编排反转为模型自控循环：工具（名称+描述+参数说明）交给模型，它自己决定调什么、
调几轮、何时收笔（`final`）。工具注册即用——加工具=注册一条 `ToolSpec`，循环与提示词
零改动（system prompt 里的工具清单由注册表动态生成）。

协议（SPEC22 §2.2）：每轮模型只输出一个 JSON：
    {"thought": "...", "tool": "名", "args": {...}}
    {"thought": "...", "final": ...}
`final` 的语义由站点定（Ask=收笔转流式合成；总结=写正文信号）。

三防线（§2.3，实测必配）：
1. 容错解析：final 变体一律收；JSON 解析容围栏容噪声；解析失败先过修复梯子
   （截断补括号/补引号、trailing comma、错位闭括号——对齐 metacodes message_repair）。
2. 引导式错误：未知工具给"可用工具清单+收笔格式示例"；工具异常给结构化 error。
3. 同签名熔断：hash(tool+args) 第 3 次重复即断。
终止：`final` 唯一自然完成；轮次上限只当防呆兜底；上限/熔断触发时强制收笔并如实
说明，绝不静默失败。

SPEC23 预留（§2.5①）：模型输出里的可选 `used_cognition: [ids]` 字段被解析后原样
透传到结果（未申报=空列表，不报错），认知层计数由站点消费。

LLM 调用由站点注入（Ask=async_stream_client.acomplete 逐调用过闸；总结=PooledClient
包装），内核不碰供给池——计量/准入零改动（§2.4）。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# 轮次上限默认值（env 覆盖由站点读好传入；这里只是内核兜底默认）。
DEFAULT_MAX_ROUNDS = 8
# 同签名熔断阈值：第 N 次出现同一 (tool, args) 即断（覆盖"成功但零增益"与"同错重试"）。
CIRCUIT_BREAK_AT = 3


@dataclass
class ToolSpec:
    """一条可注册工具（SPEC22 §2.1）。

    description 写法参照 skills/knowforge/mcp_server.py 的 TOOLS：何时用/何时用别的
    工具/参数含义。materialize 把工具结果转成喂回模型的材料文本，并在 ctx 里登记
    [N]/[图N] 映射（副作用）——模型永远只见短标记，没有编造路径的可能。
    """
    name: str
    description: str     # 模型看到的工具说明（何时用）
    params: str          # 参数说明文本，如 '{"query": "关键词", "topk": 8}'
    afn: Callable[[dict[str, Any], dict[str, Any]], Awaitable[Any]]   # async (ctx, args) -> Any
    materialize: Callable[[dict[str, Any], dict[str, Any], Any], str]  # (ctx, args, result) -> str


ToolRegistry = list  # list[ToolSpec]


@dataclass
class LoopResult:
    """循环结果。status: final=模型自然收笔 / forced=上限或熔断强制收笔。

    forced 时 note 给出如实说明（"检索了 N 轮未找到充分材料"类），站点必须把它带给
    收尾路径转达用户，绝不静默失败（SPEC22 §2.3-4）。
    """
    status: str                      # "final" | "forced"
    final: Any                       # final 载荷（forced 时为 None）
    rounds: int                      # 实际消耗的 LLM 轮次
    tool_calls: list                 # [(tool_name, args), ...] 成功执行的调用
    used_cognition: list             # SPEC23 预留：模型申报的认知条目 id（去重保序）
    note: str = ""                   # forced 原因的如实说明；final 时为空
    forced_reason: str = ""          # "max_rounds" | "circuit_break" | "bad_json"
    flags: Optional[dict] = None     # 收笔 JSON 里的可选申报（如 deliverable=true），站点自解释


def _repair_json(s: str) -> Optional[dict[str, Any]]:
    """修复梯子（防线1 加强，2026-08-25 对齐 metacodes ``message_repair.zig`` 的弱模型
    健壮性层）：常规解析失败才进来，逐项救治弱模型高频畸形——

    - **截断收尾**（本内核 bad_json 主因=长 thought/args 被 max_tokens 截断）：补未
      闭合字符串的引号（metacodes 登记为缺口，Python 侧补上）、按栈逆序补 ``}``/``]``；
    - **trailing comma**：``{"a":1,}`` 与截断尾 ``{"a":1,`` 两种形态都删；
    - **错位闭括号**：与栈顶不配对的 closer 丢弃（metacodes ③b 的简化版）。

    全部只做字符串感知的单遍走扫，修完必须整体过 ``json.loads`` 才采用；修不好返回
    None，由调用方走 bad_json 引导——宁可引导重试，不喂进半猜的结构。"""
    start = s.find("{")
    if start < 0:
        return None
    out: list[str] = []
    stack: list[str] = []
    in_str = escaped = False
    for ch in s[start:]:
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                continue  # 错位闭括号：丢弃；最终 json.loads 把关
            stack.pop()
            j = len(out) - 1
            while j >= 0 and out[j] in " \t\r\n":
                j -= 1
            if j >= 0 and out[j] == ",":
                del out[j]  # closer 前的 trailing comma
        out.append(ch)
        if not stack and ch in "}]":
            break  # 完整对象到手，尾部噪声不要
    if in_str:
        if escaped:
            out.pop()  # 悬空转义符会吃掉补上的引号
        out.append('"')
    tail = "".join(out).rstrip()
    if tail.endswith(","):
        tail = tail[:-1]  # 截断尾 trailing comma（补括号前删，否则补完又非法）
    tail += "".join(reversed(stack))
    try:
        obj = json.loads(tail)
    except (json.JSONDecodeError, RecursionError):
        return None
    return obj if isinstance(obj, dict) else None


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """从模型输出里抠出 JSON 对象，容忍 ```json 围栏、前后噪声、以及**连续多个
    对象**（防线1）。qwen 实测惯犯（2026-08-18）：一次吐 {"thought":...}\\n{"final":...}
    两个对象——优先取含 final/tool 的那个。常规扫描拿不到协议对象时过修复梯子
    ``_repair_json``（截断/trailing comma 等，能把浪费的 bad_json 引导轮救成可用轮）。"""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s).rstrip("`").strip()
    objs: list[dict[str, Any]] = []
    dec = json.JSONDecoder()
    idx = 0
    while True:
        brace = s.find("{", idx)
        if brace < 0:
            break
        try:
            obj, end = dec.raw_decode(s, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(obj, dict):
            objs.append(obj)
        idx = end
    for obj in objs:
        if "final" in obj or "tool" in obj or "action" in obj:
            return obj
    # 扫描没拿到协议对象（整体坏了/只捞到内层碎片如 args）→ 修复梯子重建外层对象。
    repaired = _repair_json(s)
    if repaired is not None and any(k in repaired for k in ("final", "tool", "action", "thought")):
        return repaired
    return objs[0] if objs else repaired


def _norm_final(obj: dict[str, Any]) -> Optional[Any]:
    """容错收笔（防线1）：{"final":...} / {"tool":"final","args":{...}} /
    {"action":"final",...} 变体一律收。返回 final 载荷；非收笔返回 None。"""
    if "final" in obj:
        return obj["final"]
    if obj.get("tool") == "final" or obj.get("action") == "final":
        a = obj.get("args") if isinstance(obj.get("args"), dict) else obj
        for k in ("final", "answer", "content", "text"):
            if a.get(k) is not None:
                return a[k]
        # 收笔意图明确但没找到载荷字段：整个 args 当载荷（宁收勿转）
        return {k: v for k, v in a.items() if k not in ("thought", "tool", "action")} or True
    return None


def _used_cognition(obj: dict[str, Any]) -> list[str]:
    """SPEC23 预留①：解析可选 used_cognition 申报，未申报/坏形状=空列表不报错。"""
    raw = obj.get("used_cognition")
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


# final 示例默认值。多数站点 final 只是收笔信号（正文由后续步骤生成），示例故意
# 用占位 "ok"——旧示例写"给用户的最终回答"，qwen 实测会照着把整篇答案塞进收笔轮
# （或干脆写成散文），再被站点丢弃重生成，一问白烧 8-10s（2026-08-24 计时实测）。
# 站点若需要 final 携带真实载荷（如 Ask 元对话短回应），通过 run_loop(final_example=…)
# 自定义示例并在自己的纪律段里说明。
DEFAULT_FINAL_EXAMPLE = '{"thought": "...", "final": "ok"}'


def render_tool_prompt(registry: list[ToolSpec],
                       final_example: str = DEFAULT_FINAL_EXAMPLE) -> str:
    """由注册表动态生成 system prompt 的工具清单 + 循环协议段（SPEC22 §2.1/2.2）。"""
    tools = "\n".join(f"- {t.name}：{t.description} 参数 {t.params}" for t in registry)
    return f"""可用工具：
{tools}

每轮只输出一个 JSON 对象（不要任何其它文字），二选一：
{{"thought": "简短思考", "tool": "工具名", "args": {{...}}}}
{final_example}

规则：
- 一次没查到就换关键词、换工具再试，不要立刻放弃。
- 材料够了就尽快收笔（final），不要为凑轮数继续调工具。
- 收笔轮**不要写答案正文**——final 按示例给个简短值即可，正文不归这一步。"""


async def run_loop(
    *,
    registry: list[ToolSpec],
    llm_fn: Callable[[list[dict[str, str]]], Awaitable[str]],
    system: str,
    task: str,
    ctx: Optional[dict[str, Any]] = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    on_event: Optional[Callable[[str, dict[str, Any]], Any]] = None,
    site: str = "agent_loop",
    final_guard: Optional[Callable[[int, list, Any], Optional[str]]] = None,
    final_example: str = DEFAULT_FINAL_EXAMPLE,
    feedback_suffix: str = "",
    stop_check: Optional[Callable[[dict, list], Optional[str]]] = None,
) -> LoopResult:
    """跑一轮有界 agent loop，返回 LoopResult。

    - ``system``：站点组好的人设+纪律段；内核在其后追加工具清单+循环协议
      （站点保持"分段追加"结构，SPEC23 认知段由站点拼在 system 里，§2.5③）。
    - ``task``：首轮 user 内容（问题/任务说明，含历史由站点自拼）。
    - ``ctx``：站点上下文，透传给 afn/materialize（materializer 在里面累积
      citations/images 映射，副作用）。
    - ``llm_fn``：async (messages) -> str。Ask 注入 acomplete(gate_user=...)，
      总结注入 PooledClient 包装——供给/计量在站点侧，内核不管。
    - ``on_event``：语义事件回调（tool_start/tool_done/final），同步函数即可；
      异常吞掉不拦循环。
    - ``final_example``：收笔 JSON 示例（进 system 协议段与各引导文本）。默认占位
      "ok"；站点需要 final 携带真实载荷时自定义。
    - ``feedback_suffix``：追加在**每轮**工具回喂末尾的就地提醒（qwen 对就地提醒的
      依从性远好于 system 开头的规则；Ask 用它顶住"收笔轮写正文"的惯性）。
    - ``stop_check``：每次工具成功执行后调 ``(ctx, tool_calls) -> Optional[str]``，
      返回非空原因则站点主动收笔（省掉一整轮"模型确认够了"的 LLM 调用——总结站点
      实测 7/10 篇的收笔轮是纯开销）。异常吞掉不拦循环。
    - 历史压缩：P0 全量保留（Ask 实测 3-5 轮、单轮材料有界，SPEC22 §2.4）。
    """
    ctx = ctx if ctx is not None else {}
    by_name = {t.name: t for t in registry}
    full_system = f"{system}\n\n{render_tool_prompt(registry, final_example)}"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": task},
    ]
    tool_calls: list = []
    used_cog: list[str] = []
    repeat: dict[str, int] = {}
    bad_json_streak = 0

    def emit(etype: str, payload: dict[str, Any]) -> None:
        if on_event is None:
            return
        try:
            on_event(etype, payload)
        except Exception as exc:  # noqa: BLE001 - 事件回调绝不拦循环
            logger.warning("agent_loop:%s on_event failed err=%s", site, exc)

    def note_used(obj: dict[str, Any]) -> None:
        for cid in _used_cognition(obj):
            if cid not in used_cog:
                used_cog.append(cid)

    prose_guided = False
    for rnd in range(1, max_rounds + 1):
        raw = await llm_fn(messages)
        obj = _extract_json(raw)
        is_prose = obj is None and "{" not in (raw or "") and len((raw or "").strip()) > 50
        if is_prose and not tool_calls and not prose_guided:
            # 一个工具都没调就直接写答案正文（qwen 惯犯）——不能照单全收：实测它会
            # 跳过检索谎报"知识库未涉及"（2026-08-18 天涯库"介绍下五行"）。引导一次，
            # 把"先检索"纪律顶回去；引导后仍写散文才收（见下）。
            prose_guided = True
            logger.info("agent_loop:%s round=%s prose_guided chars=%s", site, rnd, len(raw))
            messages.append({"role": "assistant", "content": (raw or "")[:2000]})
            messages.append({"role": "user", "content":
                             "你直接写了答案正文，但还没有检索。带实质信息诉求的问题必须先用工具"
                             "检索知识库再回答（用户找图片就用 search_knowledge 搜该主题词），"
                             '不要凭自己的知识断言"知识库未涉及"。材料够了再输出 '
                             f"{final_example} 收笔；只有纯元对话（寒暄/道谢）才可直接 final。"})
            continue
        if is_prose:
            # 已经检索过（或引导过一次仍写散文）→ 按 metacodes"无工具调用即完成"
            # 当 final 收下：有材料时 final 文本本会被合成替换（零风险），继续引导
            # 只会空转烧轮次（负载高时一轮几十秒，实测卡"思考中"主因）。
            logger.info("agent_loop:%s round=%s prose_as_final chars=%s", site, rnd, len(raw))
            emit("final", {"status": "final", "rounds": rnd})
            return LoopResult("final", raw.strip(), rnd, tool_calls, used_cog)
        if obj is None:
            # JSON 写了但坏了（多为长 thought 被 max_tokens 截断）：连续两次强制
            # 收笔；单次给引导重试（防线2 精神）。
            bad_json_streak += 1
            logger.warning("agent_loop:%s round=%s bad_json raw=%r", site, rnd, (raw or "")[:200])
            if bad_json_streak >= 2:
                note = f"检索 {rnd} 轮后模型输出异常，基于已有材料作答"
                emit("final", {"status": "forced", "reason": "bad_json", "rounds": rnd})
                return LoopResult("forced", None, rnd, tool_calls, used_cog, note, "bad_json")
            messages.append({"role": "assistant", "content": (raw or "")[:2000]})
            messages.append({"role": "user", "content":
                             "上一条不是合法 JSON。每轮只输出一个 JSON 对象："
                             f'{{"thought": "...", "tool": "工具名", "args": {{...}}}} 或 {final_example}'})
            continue
        bad_json_streak = 0
        note_used(obj)

        fin = _norm_final(obj)
        if fin is not None:
            # final 守卫（站点策略，可选）：返回引导文本则驳回一次让模型改道
            # （典型：一个工具没调就长篇实质回答/断言"知识库未涉及"——qwen 实测
            # 会用合法 JSON final 绕过散文防线直接跳检索）。只驳一次，防纠缠。
            if final_guard is not None and not prose_guided:
                guide = None
                try:
                    guide = final_guard(rnd, tool_calls, fin)
                except Exception as exc:  # noqa: BLE001 - 守卫绝不拦循环
                    logger.warning("agent_loop:%s final_guard failed err=%s", site, exc)
                if guide:
                    prose_guided = True
                    logger.info("agent_loop:%s round=%s final_rejected_by_guard", site, rnd)
                    messages.append({"role": "assistant", "content": raw[:2000]})
                    messages.append({"role": "user", "content": guide})
                    continue
            emit("final", {"status": "final", "rounds": rnd})
            return LoopResult("final", fin, rnd, tool_calls, used_cog,
                              flags={k: obj[k] for k in ("deliverable",) if k in obj} or None)

        tool = obj.get("tool")
        args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
        sig = f"{tool}\x00{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        repeat[sig] = repeat.get(sig, 0) + 1
        if repeat[sig] >= CIRCUIT_BREAK_AT:
            # 防线3：同签名第 3 次 —— 空转（成功零增益/同错重试），强制收笔。
            note = f"检索 {rnd} 轮后重复同一操作未获新材料，基于已有材料作答"
            logger.info("agent_loop:%s circuit_break tool=%s rounds=%s", site, tool, rnd)
            emit("final", {"status": "forced", "reason": "circuit_break", "rounds": rnd})
            return LoopResult("forced", None, rnd, tool_calls, used_cog, note, "circuit_break")

        spec = by_name.get(str(tool))
        if spec is None and not tool and "thought" in obj:
            # 防线2：thought-only 停格（qwen 实测：写完 {"thought": "先用 X 检索…"}
            # 就 EOS，tool/args 键没出来）——对症引导，比"工具 None 不存在"有效。
            result_text = ('你只输出了 thought，没有带 "tool"/"args"。要调用工具必须在'
                           '同一个 JSON 对象里一次写全：{"thought": "…", "tool": "工具名", '
                           '"args": {…}}；要收笔用 ' + final_example)
        elif spec is None:
            # 防线2：引导式未知工具错误（实测模型能据此自纠）。
            result_text = (f"工具 '{tool}' 不存在。可用工具：{sorted(by_name)}。"
                           f"要收笔请直接输出 {final_example}")
        else:
            emit("tool_start", {"name": spec.name, "brief": _brief(spec.name, args)})
            tool_ok = False
            try:
                result = await spec.afn(ctx, args)
                result_text = spec.materialize(ctx, args, result)
                tool_calls.append((spec.name, args))
                tool_ok = True
            except Exception as exc:  # noqa: BLE001 - 防线2：结构化工具错误，不裸抛
                logger.warning("agent_loop:%s tool=%s failed err=%s", site, spec.name, exc)
                result_text = json.dumps(
                    {"error": {"code": "tool_failed", "detail": str(exc)[:300],
                               "recoverable": True}}, ensure_ascii=False)
            emit("tool_done", {"name": spec.name})
            if stop_check is not None and tool_ok:
                # 站点主动收笔：材料已判定够用，省掉"模型再确认一轮"的整次 LLM 调用。
                reason = None
                try:
                    reason = stop_check(ctx, tool_calls)
                except Exception as exc:  # noqa: BLE001 - 短路判断绝不拦循环
                    logger.warning("agent_loop:%s stop_check failed err=%s", site, exc)
                if reason:
                    logger.info("agent_loop:%s round=%s stop_check_finish reason=%s",
                                site, rnd, reason)
                    emit("final", {"status": "final", "rounds": rnd})
                    return LoopResult("final", None, rnd, tool_calls, used_cog)

        messages.append({"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)})
        left = max_rounds - rnd
        suffix = f"\n\n【你还剩 {left} 轮，材料够了就尽快收笔 final】" if left <= 2 else ""
        messages.append({"role": "user",
                         "content": f"工具返回：\n{result_text}{feedback_suffix}{suffix}"})

    # 轮次上限（防呆兜底）：强制收笔并如实说明，绝不静默失败。
    note = f"检索了 {max_rounds} 轮未找到充分材料，基于已有材料作答"
    logger.info("agent_loop:%s max_rounds=%s forced finish", site, max_rounds)
    emit("final", {"status": "forced", "reason": "max_rounds", "rounds": max_rounds})
    return LoopResult("forced", None, max_rounds, tool_calls, used_cog, note, "max_rounds")


def _brief(name: str, args: dict[str, Any]) -> str:
    """tool_start 事件给前端的一句话（"正在检索：xxx"的 xxx）。"""
    for k in ("query", "item", "fragment_id", "table_id", "label", "book"):
        v = str(args.get(k) or "").strip()
        if v:
            return v[:60]
    fids = args.get("fragment_ids")
    if isinstance(fids, (list, tuple)) and fids:   # read_media：留痕传了几张（2026-09-05 计数题排查用）
        return f"{name} ×{len(fids)}"
    return name
