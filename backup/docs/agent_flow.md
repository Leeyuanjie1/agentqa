# 智能文档问答 Agent 流程与架构说明

本文档用于单独说明当前项目的执行链路、模块职责、关键取舍和运行状态流，方便你在提交作业、录制演示视频或面试讲解时直接使用。

文档重点回答四个问题：

- 系统是如何从多个 PDF 进入统一知识库的。
- 文本型 PDF、扫描型 PDF、混合型 PDF 分别怎么处理。
- 问答时是如何完成问题路由、召回、重排、检索重试、回答、证据校验、自检和拒答的。
- 为什么这套实现可以被定义为“生产风格最小 Demo”。

## 1. 总体说明

当前系统支持两种输入方式：

- 传入单个 PDF 文件路径。
- 传入一个目录路径，自动读取目录下全部 PDF 文件，并统一构建一个检索库。

系统主流程分成两段：

1. Ingest 阶段：解析 PDF，抽取正文、条款、表格，切分 chunk，生成向量并保存 artifacts。
2. Ask 阶段：加载 artifacts，做问题路由，执行检索和必要的检索重试，生成答案，完成证据校验、自检与拒答，输出 citation 和状态元数据。

## 2. 系统总览图

```mermaid
flowchart TD
    A[用户输入\nCLI main.py / HTTP serve.py] --> B{请求类型}
    B -->|/ingest 或 main.py ingest| C[读取 pdf_path\n支持单文件或目录]
    B -->|/ask 或 main.py ask| D[读取 artifact_dir\n加载已构建知识库]

    C --> E[扫描输入目录\n收集全部 PDF 文件]
    E --> F{逐文件处理}
    F --> G[PDF 类型判断\ntext / scanned / mixed]
    G --> H[文本页解析\nPyMuPDF + pdfplumber]
    G --> I[扫描页解析\n渲染图片 + OCR API]
    H --> J[抽取正文 / 条款 / 表格]
    I --> J
    J --> K[统一构造成 DocumentElement]
    K --> L[切分为 DocumentChunk]
    L --> M[为每个 chunk 注入\nsource_pdf / page / chunk_id]
    M --> N[合并所有 PDF 的 chunks]
    N --> O[Embedding 编码\nbge-m3 或 fallback]
    O --> P[构建本地知识库\nBM25 + embeddings]
    P --> Q[保存 artifacts\nparsed_document.json / chunks.json / embeddings.npy / metadata.json]

    D --> R[加载 KnowledgeBase]
    R --> S[问题规范化\n空字符串直接报错]
    S --> T[问题路由\ngeneral / clause / table / high_no_answer_risk]
    T --> U[首轮 Hybrid Retrieval\n向量召回 + BM25]
    U --> V[Rerank\nbge-reranker-large 或 fallback]
    V --> W{证据是否偏弱或命中特殊路由}
    W -->|是| X[改写 query 并重试检索]
    W -->|否| Y[进入候选筛选]
    X --> Z[比较首轮与重试结果\n择优保留]
    Z --> Y
    Y --> AA[按路由选择优先证据\ntable/clause/general]
    AA --> AB[Answer Generator\n抽取式或兼容 LLM]
    AB --> AC[Answer Verifier\n检查答案是否被 citation 支撑]
    AC --> AD[Self Check\n综合 rerank 分、重合度和校验结果]
    AD --> AE{是否拒答}
    AE -->|是| AF[返回拒答结果\n无法根据当前文档证据可靠回答]
    AE -->|否| AG[返回答案 + citations + self_check + metadata]
```

## 3. Ingest 详细流程图

这一部分对应代码中的核心链路：

- `DocumentQaAgent.build_index`
- `resolve_pdf_inputs`
- `parse_documents`
- `parse_document`
- `KnowledgeBase.build`

```mermaid
flowchart TD
    A[开始 ingest] --> B[读取 pdf_path]
    B --> C{pdf_path 是文件还是目录}
    C -->|文件| D[生成单文件列表]
    C -->|目录| E[递归扫描目录下全部 PDF]
    D --> F[进入逐文件解析循环]
    E --> F

    F --> G[取当前 PDF]
    G --> H[按页统计文本字符数]
    H --> I{PDF 类型判断}

    I -->|text| J[全部页面走文本解析]
    I -->|scanned| K[全部页面走 OCR 解析]
    I -->|mixed| L[高文本页走文本解析\n低文本页走 OCR 解析]

    J --> M[PyMuPDF 提取 blocks]
    J --> N[pdfplumber 提取 tables]
    K --> O[页面渲染为图片]
    O --> P[调用 PaddleOCR API]
    P --> Q[解析 OCR 返回\nlines / tables]
    L --> M
    L --> N
    L --> O

    M --> R[正文清洗\n空白归一化]
    N --> S[表格转 markdown]
    Q --> T[OCR 文本拼接与表格恢复]

    R --> U[条款编号识别\n第X条 / 1.1 / 2.3.4]
    S --> V[构造 table element]
    T --> W[构造 OCR paragraph/table element]
    U --> X[构造 clause / paragraph element]

    X --> Y[生成 DocumentElement 列表]
    V --> Y
    W --> Y

    Y --> Z[Chunking\n按 chunk_size / overlap 切分]
    Z --> AA[为每个 chunk 注入\nsource_pdf / source_path / doc_index]
    AA --> AB[给 chunk_id 增加文档前缀\n避免多 PDF 冲突]
    AB --> AC{是否还有下一个 PDF}
    AC -->|是| G
    AC -->|否| AD[聚合全部 ParsedDocument]
    AD --> AE[合并总页数 / 总元素数 / source_documents]
    AE --> AF[Embedding 编码]
    AF --> AG[构建 BM25 语料]
    AG --> AH[落盘 artifacts]
    AH --> AI[返回 ingest summary]
```

## 4. 问答阶段详细流程图

这一部分对应代码中的核心链路：

- `DocumentQaAgent.ask`
- `route_question`
- `KnowledgeBase.search`
- `AnswerGenerator.generate`
- `verify_answer_support`
- `run_self_check`

```mermaid
flowchart TD
    A[开始 ask] --> B[读取问题 question]
    B --> C{问题是否为空}
    C -->|是| D[抛出 422 / ValueError]
    C -->|否| E[加载 artifact_dir 对应 KnowledgeBase]
    E --> F{索引是否存在}
    F -->|否| G[抛出 404 / IndexNotFoundError]
    F -->|是| H[问题标准化\nstrip 空白]

    H --> I[问题路由器\n判断 general / clause / table / high_no_answer_risk]
    I --> J[首轮检索]
    J --> K[Embedding 编码 query]
    J --> L[BM25 tokenize query]
    K --> M[向量相似度打分]
    L --> N[BM25 分数计算]
    M --> O[取向量 top candidates]
    N --> P[取 BM25 top candidates]
    O --> Q[合并候选集合]
    P --> Q

    Q --> R[blended_score = vector_weight * vector_score + bm25_weight * bm25_score]
    R --> S[Rerank 候选\nbge-reranker-large 或 lexical overlap]
    S --> T[按 rerank_score 排序]
    T --> U{是否需要检索重试}
    U -->|否| V[保留首轮候选]
    U -->|是| W[根据路由生成 rewritten query]
    W --> X[第二轮检索 + rerank]
    X --> Y[比较两轮结果\n按 top rerank 与路由匹配度择优]
    Y --> Z[保留最终候选]
    V --> Z

    Z --> AA[按路由挑选优先证据]
    AA --> AB{路由是否为 table}
    AB -->|是| AC[优先选择 table chunk]
    AB -->|否| AD{路由是否为 clause}
    AD -->|是| AE[优先选择 clause chunk]
    AD -->|否| AF[保留通用高分 chunk]
    AC --> AG[生成答案]
    AE --> AG
    AF --> AG

    AG --> AH[证据校验器\n检查答案与 citation 的重合度]
    AH --> AI[Self Check]
    AI --> AJ{最高 rerank 分是否低于 refuse_threshold}
    AJ -->|是| AK[高风险\n直接拒答]
    AJ -->|否| AL{citation 校验是否失败且支撑分过低}
    AL -->|是| AM[高风险\n拒答]
    AL -->|否| AN{答案与证据重合度是否偏低}
    AN -->|是| AO[中风险\n允许返回但标记需复核]
    AN -->|否| AP[低风险\ngrounded]

    AK --> AQ[构造拒答 answer]
    AM --> AQ
    AO --> AR[构造 answer + medium risk self_check]
    AP --> AS[构造 answer + low risk self_check]

    AQ --> AT[输出 citations\nsource_pdf + page + chunk_id + snippet]
    AR --> AT
    AS --> AT
    AT --> AU[返回最终响应\n包含 query_route / retrieval_retry / answer_verification]
```

## 5. 引用与可追溯性流程图

多 PDF 输入时，最容易出问题的地方不是“能不能检索”，而是“引用是否还能说清楚来自哪个文件、哪一页、哪一个 chunk”。当前系统通过 `source_pdf` 和 `chunk_id` 解决这个问题。

```mermaid
flowchart LR
    A[原始 PDF 文件] --> B[ParsedDocument]
    B --> C[DocumentElement]
    C --> D[DocumentChunk]

    D --> E[source_pdf]
    D --> F[page]
    D --> G[chunk_id]
    D --> H[source_path]

    E --> I[检索结果 RetrievalCandidate]
    F --> I
    G --> I
    H --> I

    I --> J[AnswerCitation]
    J --> K[返回给用户的证据引用]

    K --> L[文件名]
    K --> M[页码]
    K --> N[chunk 标识]
    K --> O[证据片段 snippet]
```

## 6. 关键模块职责映射

### 6.1 入口层

- `main.py`
  - CLI 启动入口。
- `serve.py`
  - HTTP 启动入口。
- `src/docqa_agent/api.py`
  - FastAPI 路由定义和错误映射。

### 6.2 编排层

- `src/docqa_agent/agent.py`
  - 统一编排 ingest 和 ask。
  - 管理 KnowledgeBase 缓存。
  - 执行问题路由、检索重试、证据校验，并汇总 citations 和 metadata。

### 6.3 解析层

- `src/docqa_agent/parsers/classifier.py`
  - 判断 PDF 类型。
- `src/docqa_agent/parsers/text_pdf.py`
  - 文本 PDF 解析。
- `src/docqa_agent/parsers/scanned_pdf.py`
  - 扫描 PDF OCR 解析。
- `src/docqa_agent/services/ocr_client.py`
  - 对接 OCR API。

### 6.4 检索层

- `src/docqa_agent/chunking.py`
  - 负责分块与 chunk 标识生成。
- `src/docqa_agent/retrieval/embeddings.py`
  - 负责 embedding 编码。
- `src/docqa_agent/retrieval/store.py`
  - 管理 BM25、embedding 矩阵和检索逻辑。
- `src/docqa_agent/retrieval/reranker.py`
  - 管理 rerank。

### 6.5 生成与校验层

- `src/docqa_agent/services/query_router.py`
  - 问题路由、query rewrite、检索重试判定。
- `src/docqa_agent/services/generator.py`
  - 答案生成。
- `src/docqa_agent/services/answer_verifier.py`
  - 答案与 citation 支撑关系校验。
- `src/docqa_agent/services/self_check.py`
  - 综合 rerank 分、答案重合度、证据校验结果做自检、风险判断和拒答。

## 7. 关键设计取舍

### 7.1 为什么支持多 PDF 合并 ingest

因为你已经明确希望 `data/input` 能放多个文件，并在一次 ingest 中统一处理。这更接近真实业务场景：一个客户项目往往不是单个 PDF，而是一组合同、制度、说明书、报价单。

当前实现采取的是“同一个 artifact_dir 对应一个聚合知识库”的方式，优点是：

- 结构简单，适合作业演示。
- 多个 PDF 可直接交叉召回。
- 不需要引入额外数据库即可完成多文档检索。

代价也很明确：

- 当前还没有 doc-level filter。
- 还没有按文档单独增量更新。
- 多文档规模继续扩大后，应升级为独立向量存储和 metadata filter。

### 7.2 为什么 OCR 只通过 API 接入

你的目标路线已经明确是 PaddleOCR API，这样的好处是：

- OCR 计算和主应用解耦。
- 更容易替换为云端或单独服务进程。
- 主项目安装依赖更轻。

### 7.3 为什么先用规则化自检

作业场景下，规则化自检比“黑盒 judge”更容易解释和验证：

- 面试时更容易讲清楚为什么拒答。
- 更利于回归测试。
- 可以逐步升级，不会推翻现有主流程。

### 7.4 为什么增加问题路由、检索重试和证据校验

这三步的目标不是让系统“更复杂”，而是让它更像一个真正的文档 QA Agent：

- 问题路由：让系统先判断当前问题更像条款问答、表格问答还是高风险无答案问题，而不是所有问题都走完全相同的路径。
- 检索重试：首轮证据不足时，不直接把失败交给生成器，而是先尝试改写 query 再检索一次。
- 证据校验：答案生成后不立即返回，而是再检查答案是否真的被 citation 支撑。

这三步共同提升了两个方面：

- 可解释性：你可以说明系统为什么选这类证据，为什么重试，为什么拒答。
- 稳健性：对表格问题、条款问题和无答案问题的处理更保守。

## 8. 适合在演示时重点强调的三个点

1. 目录下多个 PDF 会被统一 ingest，并在回答时返回来源文件名和页码。
2. 系统不是直接问 LLM，而是先做问题路由、检索、必要的检索重试、回答和证据校验。
3. 对无答案问题不会硬答，而是会在高风险路由和证据校验不足时进入拒答路径。

## 9. 后续可以继续扩展的流程节点

如果你后续还要继续完善，这几个节点最值得扩展：

- 在 ingest 后追加文档级元数据索引，例如 `doc_type`、`customer_id`、`biz_line`。
- 在 ask 阶段把当前规则化路由升级成模型化 query classification。
- 在证据校验阶段追加 LLM judge 或结构化字段一致性校验。
- 在 retrieval 后增加 table-only 路由，对数值问答优先检索表格。
