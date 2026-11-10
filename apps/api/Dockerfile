# Sarang gateway image.
#
# Build from the repository root so the knowledge base and the web client are in
# the context:
#   docker build -f apps/api/Dockerfile -t sarang-gateway .
#
# The final image contains no provider keys and no Node toolchain: every
# credential is injected at runtime as a Fly secret (or compose env), and the
# client is already compiled to static assets.

# ---------------------------------------------------------------- web build
FROM node:22-slim AS web

WORKDIR /web
ENV npm_config_audit=false \
    npm_config_fund=false

COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci

COPY apps/web ./
RUN npm run build

# ----------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/apps/api \
    PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra api --extra pg

COPY apps/api/sarang ./apps/api/sarang
COPY kb ./kb
COPY --from=web /web/dist ./apps/web/dist

# Writable cache for pre-rendered filler audio. Mount a volume to keep it warm
# across machine restarts; without one the bank is simply re-rendered at boot.
RUN mkdir -p /app/cache/fillers
ENV SARANG_CACHE_DIR=/app/cache

# Non-root, so a compromised provider response cannot rewrite the image.
RUN useradd --create-home --uid 10001 sarang \
    && chown -R sarang:sarang /app/cache
USER sarang

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

CMD ["python", "-m", "sarang", "--host", "0.0.0.0", "--port", "8000"]
