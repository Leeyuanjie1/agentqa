from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docqa_agent.agent import DocumentQaAgent
from docqa_agent.cli import app, serve
from docqa_agent.config import get_settings
from docqa_agent.exceptions import IndexNotFoundError, ParsingError
from docqa_agent.logging_utils import configure_logging


def _prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    if value:
        return value
    return default or ""


def _run_interactive_menu() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    agent = DocumentQaAgent(settings)

    while True:
        print("\n智能文档问答 Agent")
        print("1. 构建知识库 ingest")
        print("2. 提问 ask")
        print("3. 启动 HTTP 服务 serve")
        print("4. 查看命令示例")
        print("5. 退出")

        choice = input("请选择功能 [1-5]: ").strip().lower()

        try:
            if choice in {"1", "ingest"}:
                pdf_path = _prompt("请输入 PDF 文件或目录路径，留空使用默认配置")
                artifact_dir = _prompt("请输入 artifact 输出目录，留空使用默认配置")
                summary = agent.build_index_summary(
                    pdf_path=Path(pdf_path) if pdf_path else None,
                    artifact_dir=Path(artifact_dir) if artifact_dir else None,
                )
                print(summary.model_dump(mode="json"))
                continue

            if choice in {"2", "ask"}:
                question = _prompt("请输入问题")
                if not question:
                    print("问题不能为空。")
                    continue
                artifact_dir = _prompt("请输入 artifact 目录，留空使用默认配置")
                answer = agent.ask(question=question, artifact_dir=Path(artifact_dir) if artifact_dir else None)
                print(answer.model_dump(mode="json"))
                continue

            if choice in {"3", "serve"}:
                host = _prompt("请输入服务 Host", "127.0.0.1")
                port_raw = _prompt("请输入服务 Port", "9060")
                try:
                    port = int(port_raw)
                except ValueError:
                    print("Port 必须是整数。")
                    continue
                serve(host=host, port=port)
                return

            if choice in {"4", "help", "h"}:
                print("\n命令示例:")
                print("  python main.py ingest --pdf-path data/input --artifact-dir data/artifacts/attachment")
                print("  python main.py ask \"预付款金额是多少？\" --artifact-dir data/artifacts/attachment")
                print("  python main.py serve --host 127.0.0.1 --port 9060")
                continue

            if choice in {"5", "q", "quit", "exit"}:
                print("已退出。")
                return

            print("无效选项，请输入 1 到 5。")
        except KeyboardInterrupt:
            print("\n操作已取消。")
        except IndexNotFoundError as exc:
            print(f"执行失败: {exc}")
            print("提示: 当前知识库不存在或未构建完成，请先执行 ingest。")
        except ParsingError as exc:
            print(f"执行失败: {exc}")
            print("提示: 如果是 OCR 相关错误，请检查 DOCQA_OCR_API_URL、DOCQA_OCR_API_KEY，以及目标 OCR 服务的请求格式。")
        except Exception as exc:
            print(f"执行失败: {exc}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        app()
    else:
        _run_interactive_menu()
