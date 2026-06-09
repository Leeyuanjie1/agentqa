FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data/input /app/data/artifacts/attachment /app/models \
    && chown -R appuser:appuser /app

COPY src ./src
COPY main.py serve.py pyproject.toml README.md .env.example ./

USER appuser

EXPOSE 9060

CMD ["python", "serve.py"]