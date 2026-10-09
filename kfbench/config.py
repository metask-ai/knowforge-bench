"""Runner configuration: one TOML file, four sections, environment overrides.

[model]   OpenAI-compatible chat endpoint for every text step (Locator, bridge judge, Deep Reader,
          synthesizer, format checker).  REQUIRED.  This is *your* model — the authors' endpoint is not public.
[vision]  OpenAI-compatible endpoint WITH image input, used only by read_media.  It is NOT assumed that [model]
          can see images: if [vision] is absent or enabled=false, read_media is disabled (the Deep Reader keeps
          fragment_context / media_context; read_media returns an explanatory error and the run records it).
[stages]  Optional per-stage overrides of [model]: [stages.locator], [stages.reader], [stages.bridge],
          [stages.synth], [stages.checker] — each a full endpoint table.  Unset stages use [model].
[judge]   Optional endpoint for the MMLongBench-Doc model-judge protocol (scoring only).
[kb]      Retrieval API base URL (the public KnowForge entry) and the knowledge-base package UUIDs.

Environment variables override file values: KFBENCH_MODEL_BASE_URL, KFBENCH_MODEL_NAME, KFBENCH_MODEL_API_KEY,
KFBENCH_VISION_BASE_URL, KFBENCH_VISION_NAME, KFBENCH_VISION_API_KEY, KFBENCH_KB_BASE_URL.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


@dataclass
class Endpoint:
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    extra_body: dict = field(default_factory=dict)
    timeout_s: float = 120.0
    enabled: bool = True

    def ok(self) -> bool:
        return self.enabled and bool(self.base_url) and bool(self.model)


@dataclass
class KBConfig:
    base_url: str = ""
    api_key: str = ""                               # account API key from https://mind.metask-ai.com ('MCP resource packages')
    packages: dict = field(default_factory=dict)   # {"m3docvqa": {name: uuid}, "multihop_rag": {...}, "mmlongbench_doc": {doc: {...}}}
    rate_sleep_s: float = 0.0                       # optional pause between kb calls (be nice to a public server)


STAGES = ("locator", "reader", "bridge", "synth", "checker")


@dataclass
class Config:
    model: Endpoint
    vision: Endpoint
    judge: Endpoint
    kb: KBConfig
    data_dir: Path
    path: Path
    stages: dict = field(default_factory=dict)   # stage name -> Endpoint (only the overridden ones)

    def stage(self, name: str) -> Endpoint:
        return self.stages.get(name) or self.model


def _endpoint(d: dict, env_prefix: str) -> Endpoint:
    e = Endpoint(
        base_url=os.getenv(f"{env_prefix}_BASE_URL", d.get("base_url", "")).rstrip("/"),
        model=os.getenv(f"{env_prefix}_NAME", d.get("model", "")),
        api_key=os.getenv(f"{env_prefix}_API_KEY", d.get("api_key", "")),
        extra_body=dict(d.get("extra_body") or {}),
        timeout_s=float(d.get("timeout_s", 120)),
        enabled=bool(d.get("enabled", True)),
    )
    return e


def load(path: str | Path, need_model: bool = True) -> Config:
    p = Path(path)
    raw: dict[str, Any] = tomllib.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    model = _endpoint(raw.get("model") or {}, "KFBENCH_MODEL")
    vision = _endpoint(raw.get("vision") or {}, "KFBENCH_VISION")   # no fallback to [model]: image input is opt-in
    judge = _endpoint(raw.get("judge") or {}, "KFBENCH_JUDGE")
    stages: dict[str, Endpoint] = {}
    for name, d in (raw.get("stages") or {}).items():
        if name not in STAGES:
            raise SystemExit(f"config: unknown stage [stages.{name}]; valid: {', '.join(STAGES)}")
        ep = _endpoint(d or {}, f"KFBENCH_STAGE_{name.upper()}")
        ep.base_url = ep.base_url or model.base_url
        ep.model = ep.model or model.model
        ep.api_key = ep.api_key or model.api_key
        stages[name] = ep
    kbr = raw.get("kb") or {}
    packages = dict(kbr.get("packages") or {})
    pk_file = kbr.get("packages_file")
    if pk_file:
        pf = (p.parent / pk_file) if not Path(pk_file).is_absolute() else Path(pk_file)
        packages = {**json.loads(pf.read_text(encoding="utf-8")), **packages}
    kb = KBConfig(base_url=os.getenv("KFBENCH_KB_BASE_URL", kbr.get("base_url", "")).rstrip("/"),
                  api_key=os.getenv("KFBENCH_KB_API_KEY", kbr.get("api_key", "")).strip(),
                  packages=packages, rate_sleep_s=float(kbr.get("rate_sleep_s", 0) or 0))
    # data_dir may be given at top level (before any table) or inside [kb]; relative paths resolve against the config file
    dd = raw.get("data_dir") or kbr.get("data_dir") or "data"
    data_dir = Path(dd) if Path(dd).is_absolute() else (p.parent / dd)
    if need_model and not model.ok():
        raise SystemExit("config: [model] base_url and model are required (an OpenAI-compatible endpoint of your own)")
    if not kb.base_url:
        raise SystemExit("config: [kb] base_url is required")
    if not kb.api_key:
        raise SystemExit("config: [kb] api_key is required — register at https://mind.metask-ai.com and copy the key "
                         "shown under 'MCP resource packages' (or set KFBENCH_KB_API_KEY)")
    return Config(model=model, vision=vision, judge=judge, kb=kb, data_dir=data_dir, path=p, stages=stages)
