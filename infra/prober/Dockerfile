FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY bench ./bench
COPY kb ./kb

# Override args, not the entrypoint:
#   fly machine run . --region bom -- --full --label bom
ENTRYPOINT ["python", "-m", "bench.probe.regions"]
CMD ["--rtt"]
