# 智能文档问答 Agent

这是一个面向“附件 PDF 闭环问答”的 Agent 原型，也是一个生产风格最小 Demo。它不是简单把 PDF 文本抽出来后做问答，而是明确覆盖了 PDF 类型判断、正文/条款/表格抽取、问题路由、混合检索、检索重试、重排序、证据校验、来源引用、自检拒答、测试评估和可复现运行。

项目优先满足以下目标：

- 能围绕单个 PDF 跑通从 ingest 到 ask 的闭环。
- 能识别文本型、扫描型、混合型 PDF，并选择不同解析策略。
- 能根据问题类型区分条款问答、表格问答和高风险无答案问题。
- 首轮证据不足时，能够自动改写 query 并重试检索。
- 生成答案后，能够校验答案是否被 citation 支撑。
- 能返回页码、证据片段和自检结果，而不是只给自然语言答案。
- 在没有下载正式模型时仍可运行；下载 `bge-m3` 和 `bge-reranker-large` 后无需改代码即可切换到正式检索能力。
- 保持最小依赖面和清晰结构，方便你继续替换为真实生产组件。

## 1. 系统设计

### 1.1 模块划分

- `src/docqa_agent/parsers`
  - `classifier.py`：判断 PDF 是文本型、扫描型还是混合型。
  - `text_pdf.py`：用 PyMuPDF + pdfplumber 抽取正文、条款编号、表格。
  - `scanned_pdf.py`：把页面渲染成图片，通过 OCR API 提取文本和表格。
- `src/docqa_agent/retrieval`
  - `embeddings.py`：优先加载本地 `bge-m3`；模型缺失时退化为哈希向量。
  - `store.py`：构建本地知识库、保存 artifacts、混合检索、重排。
  - `reranker.py`：优先加载本地 `bge-reranker-large`；模型缺失时退化为词面重排。
- `src/docqa_agent/services`
  - `ocr_client.py`：对接 PaddleOCR API。
  - `query_router.py`：问题路由、query rewrite、检索重试判定。
  - `generator.py`：抽取式答案生成，或接 OpenAI 兼容大模型接口。
  - `answer_verifier.py`：校验答案是否被 citation 支撑。
  - `self_check.py`：综合检索分数、答案重合度、证据校验结果做 groundedness 校验。
- `src/docqa_agent/agent.py`
  - 统一编排 ingest、load、ask。
- `src/docqa_agent/api.py`
  - 提供 HTTP 服务。
- `main.py`
  - 根目录 CLI 入口，支持直接 `python main.py ...`。
- `serve.py`
  - 根目录 API 入口，支持直接 `python serve.py` 或 `uvicorn serve:app`。
- `scripts`
  - 固定问题评估脚本。
- `tests`
  - 单测和端到端 smoke test。

### 1.2 Agent 流程

1. 读取 PDF。
2. 根据每页文本字符数判断 PDF 类型。
3. 文本页走原生解析，扫描页走 OCR API。
4. 抽取正文段落、条款编号、表格 markdown。
5. 对抽取结果分块，构建本地知识库。
6. 问答时先做问题路由，判断当前问题更偏条款、表格还是高风险无答案类型。
7. 先做向量召回 + BM25 召回，再做 rerank；必要时自动改写 query 并重试检索。
8. 生成答案后再做证据校验，判断答案是否被 citation 支撑。
9. 最后做自检，判断是否 grounded、是否高幻觉风险、是否需要拒答。

### 1.3 为什么这是“生产风格最小 Demo”

- 支持配置化运行，不把路径、模型、OCR 接口硬编码在代码里。
- 产出 artifacts，便于复核解析结果、做回归测试和定位问题。
- API 明确区分 400/404/422，不把所有异常都吞成 200。
- 知识库对象缓存，避免每次问答重新加载大模型。
- 正式模型缺失时可以 fallback，保证仓库开箱即跑。

## 2. 技术路线

你当前计划的路线已纳入：

- OCR：PaddleOCR，采用 API 方式接入。
- Embedding：`bge-m3`，本地模型目录加载。
- Reranker：`bge-reranker-large`，本地模型目录加载。
- Generator：默认抽取式；如需接入推理模型，可填 OpenAI 兼容接口配置。

### 2.1 OCR API 约定

服务端只要求一个简单 JSON 接口：

请求体：

```json
{
  "page": 1,
  "image_base64": "..."
}
```

兼容以下字段之一作为文本输出：

- `data.lines`
- `data.texts`
- `data.rec_texts`

表格支持以下格式之一：

- `data.tables[].markdown`
- `data.tables[].rows`
- `data.tables[].cells`

## 3. 快速启动

### 3.1 安装依赖

```powershell
cd d:\workspace\agent
D:/Anaconda3/envs/agent/python.exe -m pip install -r requirements.txt
```

如果你更偏好 editable 安装，也可以执行：

```powershell
D:/Anaconda3/envs/agent/python.exe -m pip install -e .[dev]
```

### 3.2 准备配置

```powershell
Copy-Item .env.example .env
```

如果你已经下载好了模型，把下面两个路径改到真实目录：

- `DOCQA_EMBEDDING_MODEL_PATH=models/bge-m3`
- `DOCQA_RERANKER_MODEL_PATH=models/bge-reranker-large`

如果要测试扫描 PDF，把 OCR API 配好：

- `DOCQA_OCR_API_URL=http://your-paddleocr-service/ocr`
- `DOCQA_OCR_API_KEY=...`

### 3.3 放入你的附件 PDF

```powershell
New-Item -ItemType Directory -Force data/input
```

然后把你的附件 PDF 放到以下任一位置：

- `data/input/任何文件名.pdf`
- 你自己的任意绝对路径

如果你不想改 `.env`，默认行为是：

```powershell
扫描 data/input 目录，并读取其中全部 PDF 文件
```

如果 `data/input` 下有多个 PDF，它们会被统一解析并合并到同一个检索库中；回答时 citation 会带上来源文件名和页码。

### 3.4 构建知识库

```powershell
D:/Anaconda3/envs/agent/python.exe main.py ingest --pdf-path data/input --artifact-dir data/artifacts/attachment
```

如果你的 `.env` 保持默认值，也可以省略 `--pdf-path`：

```powershell
D:/Anaconda3/envs/agent/python.exe main.py ingest --artifact-dir data/artifacts/attachment
```

### 3.5 提问

```powershell
D:/Anaconda3/envs/agent/python.exe main.py ask "预付款金额是多少？" --artifact-dir data/artifacts/attachment
```

### 3.6 启动 HTTP 服务

```powershell
D:/Anaconda3/envs/agent/python.exe serve.py
```

接口示例：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9060/ingest -Method Post -ContentType "application/json" -Body '{"pdf_path":"data/input","artifact_dir":"data/artifacts/attachment"}'
Invoke-RestMethod -Uri http://127.0.0.1:9060/ask -Method Post -ContentType "application/json" -Body '{"question":"预付款金额是多少？","artifact_dir":"data/artifacts/attachment"}'
```

### 3.7 Docker 部署

如果你希望直接用容器部署 HTTP 服务，仓库现在已经包含：

- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

先确认本机已经安装 Docker 和 Docker Compose，然后在仓库根目录执行：

```powershell
docker compose up --build -d
```

启动后服务默认监听：

```text
http://127.0.0.1:9060
```

容器编排默认会挂载以下目录：

- `./data -> /app/data`
- `./models -> /app/models`

因此你的 PDF、artifacts 和本地模型建议放在：

- `data/input`
- `data/artifacts/attachment`
- `models/bge-m3`
- `models/bge-reranker-large`

如果 `.env` 里配置了 OCR 相关密钥，`docker-compose.yml` 会自动读取；同时会覆盖容器内的路径类配置为 Linux 路径，避免直接使用 Windows 绝对路径。

常用命令：

```powershell
docker compose up --build -d
docker compose logs -f docqa-agent
docker compose down
```

容器启动后，你可以继续用 Apifox 或 curl 调用：

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9060/health -Method Get
Invoke-RestMethod -Uri http://127.0.0.1:9060/ingest -Method Post -ContentType "application/json" -Body '{"pdf_path":"/app/data/input","artifact_dir":"/app/data/artifacts/attachment"}'
Invoke-RestMethod -Uri http://127.0.0.1:9060/ask -Method Post -ContentType "application/json" -Body '{"question":"预付款金额是多少？","artifact_dir":"/app/data/artifacts/attachment"}'
```

## 4. 解析与索引产物

构建后会在 `data/artifacts/attachment` 或你指定的 artifact 目录下生成：

- `parsed_document.json`：完整解析结果，包括正文、条款、表格、页码。
- `chunks.json` / `chunks.jsonl`：可检索分块。
- `embeddings.npy`：向量矩阵。
- `metadata.json`：索引元数据、后端类型、总页数、总元素数。

这些文件就是演示材料里“PDF 解析结果”的直接证据。

## 5. 自检与拒答策略

当前实现的是可解释的规则化自检，适合作为生产第一版的兜底：

- 若最高 rerank 分低于 `DOCQA_REFUSE_THRESHOLD`，直接拒答。
- 若有证据但分数偏低，或答案与证据重合度偏低，标记为中风险。
- 若答案与高分证据一致，标记为 grounded、低风险。

这不是最终形态，但它具备两个生产优势：

- 可审计，便于回溯为什么拒答。
- 可逐步替换成 LLM-as-a-judge 或多路校验，而不改主流程。

## 6. 测试与评估

### 6.1 pytest

```powershell
D:/Anaconda3/envs/agent/python.exe -m pytest
```

覆盖内容：

- chunking 是否保留条款号和表格 markdown。
- self-check 是否在低证据分时拒答。
- 端到端流程是否能回答表格问题并拒答无答案问题。

### 6.2 固定用例评估脚本

```powershell
D:/Anaconda3/envs/agent/python.exe scripts/evaluate_demo.py --artifact-dir data/artifacts/attachment
```

这个脚本可以直接从仓库根目录运行，不要求先做 editable 安装。

会输出 5 个问题的结果，满足题目要求：

- 至少 1 个表格问题。
- 至少 1 个无答案问题。
- 每个问题都带自检结果和 citations。

## 7. 建议演示流程

建议录 5-10 分钟视频时按这个顺序：

1. 展示 `.env` 配置。
2. 展示你放入 `data/input` 的真实附件 PDF。
3. 运行 ingest，展示 `parsed_document.json` 和 `chunks.json`。
4. 连续提 5 个问题，包含表格问题和无答案问题。
5. 运行 `scripts/evaluate_demo.py`。
6. 展示 `/health` 和 `/ask` HTTP 接口结果。

## 8. 边界情况与取舍

### 已完成

- 文本 PDF、扫描 PDF、混合 PDF 的分流。
- 正文、条款、表格抽取。
- 混合检索、重排、引用、自检、拒答。
- CLI、HTTP、测试、评估脚本。

### 当前取舍

- 条款编号识别采用规则法，适合作业场景，但复杂法律编号体系还需增强。
- OCR 表格结果依赖 PaddleOCR API 输出质量；若业务里表格是核心，应增加版面分析和表格恢复专项评估。
- 默认答案生成是抽取式而非生成式，优先保证 groundedness；接 LLM 只作为增强选项。
- 当前知识库是单文档单索引，若扩展到多文档场景，需要增加 doc_id、租户隔离和增量更新机制。

## 9. 迁移到不同业务场景的保障方案

### 金融

- 对金额、日期、阈值、比例类问题做结构化抽取校验。
- 对关键字段做双通道验证：表格解析结果和正文描述必须一致。
- 所有回答默认带证据与页码，不允许裸答。

### 合规/法务

- 对条款号建立一级索引，优先返回条款原文。
- 对拒答阈值设置更保守，宁可少答也不猜答。
- 增加 OCR 错字黑名单和人工复核入口。

### 客户交付

- 用真实交付 PDF 和固定评估集做回归测试。
- artifacts 持久化存档，方便客户验收与问题追踪。
- API 增加鉴权、限流、请求日志和链路追踪后可直接进入第一版交付。

## 10. 后续最值得补的三项

1. 给 PaddleOCR API 增加异步批处理和重试退避。
2. 给扫描 PDF 增加版面分析与单元格恢复质量评估。
3. 给多文档场景增加 metadata filter、租户隔离和增量索引。
