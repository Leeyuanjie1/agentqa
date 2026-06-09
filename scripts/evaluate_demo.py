from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docqa_agent.agent import DocumentQaAgent
from docqa_agent.config import get_settings


DEFAULT_CASES = [
    {
        "question": "服务范围包括哪些内容？",
        "must_contain": ["PDF 解析", "知识库构建"],
        "expect_refusal": False,
    },
    {
        "question": "项目实施期到什么时候结束？",
        "must_contain": ["2026 年 12 月 31 日"],
        "expect_refusal": False,
    },
    {
        "question": "预付款金额是多少？",
        "must_contain": ["48000"],
        "expect_refusal": False,
    },
    {
        "question": "保密义务覆盖哪些信息？",
        "must_contain": ["客户数据", "模型配置"],
        "expect_refusal": False,
    },
    {
        "question": "合同中约定的发票税率是多少？",
        "must_contain": [],
        "expect_refusal": True,
    },
]


def evaluate(artifact_dir: Path) -> dict[str, object]:
    settings = get_settings()
    agent = DocumentQaAgent(settings)
    results = []
    passed = 0
    for case in DEFAULT_CASES:
        answer = agent.ask(case["question"], artifact_dir=artifact_dir)
        refusal = answer.self_check.should_refuse or "无法根据当前文档证据可靠回答" in answer.answer
        contains_all = all(keyword in answer.answer for keyword in case["must_contain"])
        ok = (refusal if case["expect_refusal"] else contains_all and not refusal)
        if ok:
            passed += 1
        results.append(
            {
                "question": case["question"],
                "answer": answer.answer,
                "passed": ok,
                "self_check": answer.self_check.model_dump(mode="json"),
                "citations": [citation.model_dump(mode="json") for citation in answer.citations],
            }
        )
    return {
        "passed": passed,
        "total": len(DEFAULT_CASES),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the demo knowledge base with five fixed QA cases.")
    parser.add_argument("--artifact-dir", default="data/artifacts/attachment")
    args = parser.parse_args()
    report = evaluate(Path(args.artifact_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
