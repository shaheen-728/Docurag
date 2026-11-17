FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

COPY alembic.ini ./
COPY migrations ./migrations

RUN uv sync --locked --no-dev

EXPOSE 8000

CMD ["uvicorn", "docurag.main:app", "--host", "0.0.0.0", "--port", "8000"]