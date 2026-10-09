[English](README.md) | **简体中文**

# knowforge-bench

论文复现套件：

> **Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA**（定位、判定、深读：基于知识点索引的职能分工工具智能体，用于开放域多模态文档问答）— Wenlin Lin, Fanzhe Wei（Metask Lab）。预印本公开后在此补充链接。

你能得到的：三套基准语料（M3DocVQA、MultiHop-RAG、MMLongBench-Doc）以只读知识库托管（需注册免费账号，见第 2 节），以及一个小型 Python 程序，用**你自己的**模型端点复现论文各表的每一行。你这边不需要 GPU；这里不分发任何基准文档和运行结果——题集从官方来源下载，文档由知识库提供，数字靠重跑得到。

## 1. 复现论文

### 安装

```
git clone https://github.com/metask-ai/knowforge-bench && cd knowforge-bench
pip install -r requirements.txt            # Python >= 3.10；仅依赖 requests（3.10 另需 tomli）
python data/fetch_all.py                   # 官方题集 -> data/（几 MB；同时拉取 MMLongBench-Doc 的官方评分代码）
cp config.example.toml config.toml
# 把知识库 API Key 填进 [kb] api_key（见下方「获取 API Key」），然后执行一次：
python -m kfbench.run subscribe            # 把 137 个基准知识库加到你的账号下
```

### 获取 API Key

知识库接口需要一个（免费）账号。到 [mind.metask-ai.com](https://mind.metask-ai.com) 注册，打开 **MCP 资源包管理**：每个账号都自带一把默认 API Key（`kf_…`；点「重新生成」后旧 Key 立即失效）。把它填进 `config.toml` 的 `[kb] api_key`，或设置环境变量 `KFBENCH_KB_API_KEY`。然后 `python -m kfbench.run subscribe` 一次性把 `data/kb_packages.json` 里的资源包 UUID 加到你的账号下；可重复执行，`--bench m3d|mhr|mmlb` 只加某一个基准。没带 Key 时接口返回 HTTP 426，Key 无效返回 401，资源包不在你账号下返回 403。

要求：能出网 HTTPS 访问 `mind.metask-ai.com`（知识库）和你的模型端点，下载题集时访问一次 GitHub / Hugging Face。在 Linux 上测试过，程序为纯 Python。

### 配置模型（`config.toml`）

| 配置段 | 作用 | 是否必需 | 论文使用 |
|---|---|---|---|
| `[model]` | 文本模型（OpenAI 兼容 chat）：Locator、桥接判定、Deep Reader、合成器、格式检查 | 必需 | Qwen3.6-35B-A3B-FP8，sglang 部署（4×RTX 4090，`--tp 4`），关闭思考：`extra_body = { chat_template_kwargs = { enable_thinking = false } }`，`timeout_s = 120`。迁移行：DeepSeek Flash / Pro 公开 API |
| `[vision]` | 带图片输入的模型，只供 `read_media` 使用（Deep Reader 查看下载下来的图/表） | 可选 | 同一个 Qwen3.6-35B-A3B 端点。不配则 `read_media` 关闭（记录在输出里）；纯文本的 MultiHop-RAG 不受影响 |
| `[stages.<name>]` | 按阶段覆盖 `locator` / `reader` / `bridge` / `synth` / `checker`，未设置的键继承 `[model]` | 可选 | 主表未使用 |
| `[judge]` | 仅用于 MMLongBench-Doc 打分（该基准的模型评判协议） | `score --bench mmlb` 需要 | Qwen3.6-35B-A3B；请报告你用的评判模型 |
| `[kb]` | 知识库地址、你的 API Key、资源包映射 | `api_key` 必填，其余保持默认 | `https://mind.metask-ai.com`、`data/kb_packages.json` |

作者的模型端点不公开，模型侧全部跑在你提供的端点上。

### 运行

```
# 第一层——只测检索，不需要模型
python -m kfbench.run retrieval --bench m3d --set full --out results/retrieval_m3d.json

# 论文的流水线变体（--bench m3d | mmlb | mhr；--set smoke | mid | full）
python -m kfbench.run answer --bench m3d --mode M5 --set mid --out results/m3d_mid_M5.json   # 完整流水线（Locator 4 轮 + Deep Reader 4 轮）
python -m kfbench.run answer --bench m3d --mode P1 --set mid --out results/m3d_mid_P1.json   # 无 Deep Reader
python -m kfbench.run answer --bench m3d --mode P0 --set mid --out results/m3d_mid_P0.json   # 单 agent、全工具、8 轮
python -m kfbench.run answer --bench m3d --mode S4 --set mid --out results/m3d_mid_S4.json   # 一搜一答

# 打分
python -m kfbench.run score --bench m3d  --answers results/m3d_mid_M5.json     # EM / F1 / 找对文档
python -m kfbench.run score --bench mhr  --answers results/mhr_mid_M5.json     # MultiHop-RAG 官方准确率
python -m kfbench.run score --bench mmlb --answers results/mmlb_mid_M5.json    # 评判准确率 / F1（需要 [judge]）
```

参数：`--loc / --deep` 轮次上限（论文 4 / 4），`--workers` 并发题数（共享端点下 2 路安全；论文时延数字用 1 路），`--limit`，`--qids`。

规模与大致耗时（4 卡 35B 模型、2 路）：`smoke` = 30 / 31 / 25 题（几分钟）；`mid` = 300 / 299 / 98 题（1–3 小时）；`full` = 2,441 / 2,556 / 1,073 题（M3DocVQA ≈ 14 h，MultiHop-RAG ≈ 9 h，MMLongBench-Doc ≈ 9 h）。只测检索的全量几分钟。输出写到 `results/`（已 git-ignore）。

程序使用论文提示词，通过产品 MCP 服务器封装的同一批只读知识库端点（`/api/v2/kb/{uuid}/...`）访问知识库，但不走 MCP 协议。与进程内系统的已知差异列在 `kfbench/materials.py` 文件头。复现校验（2026-09-21）：MultiHop-RAG 299 题子集、同一 Qwen3.6-35B-A3B 端点，本程序官方准确率 73.9，作者进程内 harness 71.9（配对胜 16 : 负 10），在轮间波动带内。

## 2. 知识库

语料以只读 KnowForge 知识库托管在我们的验证节点上，访问凭证 = 你账号的 API Key + 资源包 UUID（由 `kfbench.run subscribe` 加到你的账号下）；全部 UUID 列在 `data/kb_packages.json`。

| 基准 | 资源包数 | 资源包 UUID | 内容 |
|---|---|---|---|
| M3DocVQA | 1 | `resource_package:d64aa2dd7f1b424487122e0732e37502` | 3,368 篇文档全池放在一个库里（开放域） |
| MultiHop-RAG | 1 | `resource_package:fd6ca737b71447a9b724f8e80e9258d2` | 609 篇新闻语料 |
| MMLongBench-Doc | 135 | 每篇文档一个库，按官方 `doc_id` 索引 | 单文档设置 |

基础地址 `https://mind.metask-ai.com`；每次调用带 `Authorization: Bearer <你的 API Key>`；端点为 `{base}/api/v2/kb/{uuid}/{card | overview | sources | search | fragment-context | book-toc | list-media | find-figure | media-context | tables | query-table | table | download}`。不需要模型的快速检查：

```
curl -s -X POST https://mind.metask-ai.com/api/v2/kb/resource_package:fd6ca737b71447a9b724f8e80e9258d2/search \
     -H "Authorization: Bearer $KFBENCH_KB_API_KEY" -H 'content-type: application/json' -d '{"query": "Which company acquired the studio in 2023?", "topk": 8, "score_fields": true}'
```

这些知识库也可以在程序之外使用：在 **MCP 资源包管理** 页面加上这些 UUID 直接浏览，或用你的 Key 把自己的 agent 指向上面的端点（服务端不提供读图，下载原件后用你自己的多模态模型看）。调用按 UUID 限流，遇到 HTTP 429 请在 `config.toml` 里设置 `rate_sleep_s`。

公开知识库是在验证节点上用与论文实验相同的转换文档和抽取知识点重建的（建库流程为 2026-09-21 版本，该版本还会建立每篇文档的章节表，论文所用本地索引对多数 PDF 缺这一项）；图表描述与带类型关系由同一模型族重新生成。文档、图、表、例题数量与论文索引完全一致，论文的开放验证一节报告了实测的答题差距（即上文 MultiHop-RAG 对齐检查）。

## 3. 许可

代码：Apache-2.0。论文文本：CC BY 4.0。基准数据**不**在此分发；下载脚本指向官方来源（M3DocVQA；MMLongBench-Doc CC BY-NC 4.0 仅限研究；MultiHop-RAG ODC-BY）。
