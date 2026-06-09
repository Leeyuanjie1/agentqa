# 演示检查清单

提交前建议逐项截图或录屏：

1. `.env` 中模型路径、OCR API 地址已配置。
2. `data/input` 中有待测 PDF，或你准备好了绝对路径。
3. 成功运行 ingest，控制台返回 `pdf_type`、`total_pages`、`total_chunks`。
4. 展示 `parsed_document.json` 中的正文、条款号、表格 markdown。
5. 连续展示 5 个问题的结果。
6. 至少一个问题来自付款表格。
7. 至少一个问题为无答案问题，并出现拒答。
8. 展示 answer 中的 `citations` 和 `self_check`。
9. 运行 `pytest` 或 `scripts/evaluate_demo.py`。
10. 展示根目录 `main.py`、`serve.py` 的启动方式，以及 `/health`、`/ingest`、`/ask` 接口。
