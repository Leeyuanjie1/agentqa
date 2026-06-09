from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from docqa_agent.api import app


if __name__ == "__main__":
    uvicorn.run("serve:app", host="127.0.0.1", port=9060, reload=False)
