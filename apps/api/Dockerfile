# Sarang gateway image.
#
# Build from the repository root so the knowledge base and the web client are in
# the context:
#   docker build -f apps/api/Dockerfile -t sarang-gateway .
#
# The final image contains no provider keys and no Node toolchain: every
# credential is injected at runtime as a Fly secret / Voroa environment
# variable / compose env, and the client is already compiled to static assets.
#
# The listen port comes from ``PORT`` when the platform sets one (Voroa passes
# 3000, Heroku-style hosts pass their own), and falls back to 8000 so a local
# ``docker run -p 8000:8000`` behaves exactly as it did before.

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

# Both the probe and the server read the same ``PORT`` so a platform-injected
# port cannot leave the container reporting healthy on a port nobody serves.
HEALTHCHECK --interval=20s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,sys,urllib.request; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/healthz', timeout=2).status == 200 else 1)"

CMD ["sh", "-c", "exec python -m sarang --host 0.0.0.0 --port ${PORT:-8000}"]
