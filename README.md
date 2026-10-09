**English** | [简体中文](README.zh-CN.md)

# knowforge-bench

Reproduction kit for the paper

> **Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA** — Wenlin Lin, Fanzhe Wei (Metask Lab). The preprint link will be added here once it is public.

What you get: the three benchmark corpora (M3DocVQA, MultiHop-RAG, MMLongBench-Doc) hosted as read-only knowledge bases (free account, see §2), and a small Python runner that reproduces every row of the paper's tables with **your own** model endpoint. Nothing here needs a GPU on your side, and no benchmark documents or run outputs are redistributed: the question sets are fetched from their official sources, the documents are served by the knowledge bases, and the numbers are re-run, not downloaded.

## 1. Reproduce the paper

### Install

```
git clone https://github.com/metask-ai/knowforge-bench && cd knowforge-bench
pip install -r requirements.txt            # Python >= 3.10; only `requests` (+ `tomli` on 3.10)
python data/fetch_all.py                   # official question sets -> data/ (a few MB; also clones the MMLongBench-Doc eval code)
cp config.example.toml config.toml
# put your knowledge-base API key into [kb] api_key (see "Get an API key" below), then, once:
python -m kfbench.run subscribe            # adds the 137 benchmark knowledge bases to your account
```

### Get an API key

The knowledge-base API needs a (free) account. Register at [mind.metask-ai.com](https://mind.metask-ai.com), open **MCP 资源包管理** (MCP resource packages): every account has a default API key there (`kf_…`; "regenerate" revokes the old one). Put it into `[kb] api_key` in `config.toml`, or export `KFBENCH_KB_API_KEY`. `python -m kfbench.run subscribe` then adds the package UUIDs from `data/kb_packages.json` to your account in one call; it is idempotent, and `--bench m3d|mhr|mmlb` limits it to one benchmark. Without a key the API answers HTTP 426, with an unknown key 401, and for a package that is not on your account 403.

Requirements: outbound HTTPS to `mind.metask-ai.com` (knowledge bases), to your model endpoints, and once to GitHub / Hugging Face for the question sets. Tested on Linux; the runner is pure Python.

### Configure your models (`config.toml`)

| Section | Role | Required | What the paper used |
|---|---|---|---|
| `[model]` | text model (OpenAI-compatible chat) for Locator, bridge judge, Deep Reader, synthesizer, format checker | yes | Qwen3.6-35B-A3B-FP8 on sglang (4×RTX 4090, `--tp 4`), thinking off: `extra_body = { chat_template_kwargs = { enable_thinking = false } }`, `timeout_s = 120`. Transfer rows: DeepSeek Flash / Pro via their APIs |
| `[vision]` | model with image input, used only by `read_media` (the Deep Reader looks at a downloaded figure/table) | opt-in | the same Qwen3.6-35B-A3B endpoint. Leave it out and `read_media` is disabled (recorded in the output); text-only MultiHop-RAG is unaffected |
| `[stages.<name>]` | per-stage override for `locator` / `reader` / `bridge` / `synth` / `checker`; unset keys inherit `[model]` | optional | not used in the main tables |
| `[judge]` | MMLongBench-Doc scoring only (the benchmark's model-judge protocol) | for `score --bench mmlb` | Qwen3.6-35B-A3B; report the judge you used |
| `[kb]` | knowledge-base base URL, your API key, package map | `api_key` yes; keep the other defaults | `https://mind.metask-ai.com`, `data/kb_packages.json` |

The authors' endpoints are not public; everything model-side runs on endpoints you provide.

### Run

```
# layer one — retrieval only, no model needed (still needs [kb] api_key)
python -m kfbench.run retrieval --bench m3d --set full --out results/retrieval_m3d.json

# the paper's pipeline variants (--bench m3d | mmlb | mhr; --set smoke | mid | full)
python -m kfbench.run answer --bench m3d --mode M5 --set mid --out results/m3d_mid_M5.json   # full pipeline (Locator 4 + Deep Reader 4)
python -m kfbench.run answer --bench m3d --mode P1 --set mid --out results/m3d_mid_P1.json   # no Deep Reader
python -m kfbench.run answer --bench m3d --mode P0 --set mid --out results/m3d_mid_P0.json   # one agent, all tools, 8 rounds
python -m kfbench.run answer --bench m3d --mode S4 --set mid --out results/m3d_mid_S4.json   # one-shot search-and-answer

# score
python -m kfbench.run score --bench m3d  --answers results/m3d_mid_M5.json     # EM / F1 / document found
python -m kfbench.run score --bench mhr  --answers results/mhr_mid_M5.json     # official MultiHop-RAG accuracy
python -m kfbench.run score --bench mmlb --answers results/mmlb_mid_M5.json    # judge accuracy / F1 (needs [judge])
```

Flags: `--loc / --deep` round caps (paper: 4 / 4), `--workers` concurrent questions (2 is safe on a shared endpoint; the paper's latency numbers used 1), `--limit`, `--qids`.

Sizes and rough wall-clock with a 35B model on 4 GPUs at 2 workers: `smoke` = 30 / 31 / 25 questions (minutes); `mid` = 300 / 299 / 98 (1–3 h); `full` = 2,441 / 2,556 / 1,073 (M3DocVQA ≈ 14 h, MultiHop-RAG ≈ 9 h, MMLongBench-Doc ≈ 9 h). Retrieval-only full runs take minutes. Outputs go to `results/` (git-ignored).

The runner uses the paper's prompts and the same read-only knowledge-base endpoints the product's MCP server wraps (`/api/v2/kb/{uuid}/...`), but not the MCP protocol itself. Known differences from the in-process system are listed at the top of `kfbench/materials.py`. Reproduction check (2026-09-21): on the MultiHop-RAG 299-question subset with the same Qwen3.6-35B-A3B endpoint this runner scores 73.9 official accuracy against 71.9 for the authors' in-process harness (paired wins 16 : losses 10), inside the run-to-run band.

## 2. The knowledge bases

The corpora are hosted on our verification node as read-only KnowForge knowledge bases. Access = your account's API key + the package UUID (added to your account by `kfbench.run subscribe`); all UUIDs are in `data/kb_packages.json`.

| Benchmark | Packages | Package UUID | Contents |
|---|---|---|---|
| M3DocVQA | 1 | `resource_package:d64aa2dd7f1b424487122e0732e37502` | the full 3,368-document pool in one knowledge base (open-domain) |
| MultiHop-RAG | 1 | `resource_package:fd6ca737b71447a9b724f8e80e9258d2` | the 609-article news corpus |
| MMLongBench-Doc | 135 | one per document, keyed by the official `doc_id` | single-document setting |

Base URL `https://mind.metask-ai.com`; every call carries `Authorization: Bearer <your API key>`; endpoints are `{base}/api/v2/kb/{uuid}/{card | overview | sources | search | fragment-context | book-toc | list-media | find-figure | media-context | tables | query-table | table | download}`. Quick check with no model:

```
curl -s -X POST https://mind.metask-ai.com/api/v2/kb/resource_package:fd6ca737b71447a9b724f8e80e9258d2/search \
     -H "Authorization: Bearer $KFBENCH_KB_API_KEY" -H 'content-type: application/json' -d '{"query": "Which company acquired the studio in 2023?", "topk": 8, "score_fields": true}'
```

You can also use the same knowledge bases outside the runner: add the UUIDs on the **MCP 资源包管理** page and browse them there, or point your own agent at the endpoints above with your key (there is no server-side image reading; download the asset and use your own vision model). Calls are rate-limited per UUID; set `rate_sleep_s` in `config.toml` if you see HTTP 429.

The public knowledge bases were rebuilt on the verification node from the same converted documents and extracted knowledge points as the paper's runs (build pipeline as of 2026-09-21, which also builds per-document section tables the paper's local index lacked for most PDFs); multimodal descriptions and typed relations were regenerated by the same model family. Document, image, table and example counts match the paper's index exactly; the paper's open-verification section reports the measured answer gap (the MultiHop-RAG check above).

## 3. Licences

Code: Apache-2.0. Paper text: CC BY 4.0. Benchmark data is **not** redistributed here; fetch scripts point at the official sources (M3DocVQA; MMLongBench-Doc CC BY-NC 4.0, research use; MultiHop-RAG ODC-BY).
