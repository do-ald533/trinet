# TriNet inference API image (PLANO §9).
#
# Multi-stage build following the official uv Docker guide
# (https://docs.astral.sh/uv/guides/integration/docker/):
#   1. `build`   - the `slim` stage below (shell + package tools), root.
#                  Installs the locked production dependencies of `trinet-api` into /app/.venv.
#   2. `runtime` - the same `slim` stage with pip removed, running as non-root 65532.
#                  Receives ONLY the virtualenv and the model artifacts (artifacts = last layer).
#
# Base image (ADR 0012): every flow of the project is free and credential-free, so the default is
# the official `python:3.13-slim-trixie` image pinned by digest (stage `slim`, below). Nothing needs
# a login, a token or an account, in CI/CD or locally:
#
#   docker build -f deploy/Dockerfile .
#   # or: make build
#
# Optional, never used by CI/CD: any other base can be passed by reference, as long as BUILD_IMAGE
# and BASE_IMAGE are the same OS/Python build. A Docker Hardened Images example (free Docker
# account, `docker login dhi.io`; `make dhi-digests` prints digest-pinned references):
#
#   --build-arg BASE_IMAGE=dhi.io/python:3.13@sha256:<digest>
#   --build-arg BUILD_IMAGE=dhi.io/python:3.13-dev@sha256:<digest>
#
# Both stages must come from the same OS/Python build so the interpreter path recorded in the
# virtualenv (/app/.venv/bin/python -> system python) is identical in the runtime image.

ARG BASE_IMAGE=slim
ARG BUILD_IMAGE=slim

# uv binary, pinned by digest (multi-arch index of ghcr.io/astral-sh/uv:0.13.7).
FROM ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 AS uv

# Default base for both stages (Debian 13 "trixie", Python 3.13), pinned by digest. It is a named
# stage so `BASE_IMAGE=slim` resolves to it without any registry round trip and so Dependabot
# (docker ecosystem) keeps the digest fresh.
FROM python:3.13-slim-trixie@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS slim
# - Drop pip: not needed at run time, and its vendored urllib3/msgpack copies only add scanner
#   findings and attack surface. uv installs the project into its own venv and never needs pip.
# - Apply Debian security updates published after the pinned digest (the pipeline must not block on
#   an already-fixed distro CVE while the upstream image catches up). Not byte-reproducible by
#   design; the pinned digest and the Trivy gate bound the drift.
# hadolint ignore=DL3005
RUN set -eux; \
    python -m pip --disable-pip-version-check uninstall --yes pip setuptools wheel; \
    apt-get update; \
    apt-get upgrade --yes --no-install-recommends; \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- build stage
FROM ${BUILD_IMAGE} AS build

COPY --from=uv /uv /uvx /bin/

# Use the system interpreter (never download one); compile bytecode at build time so the
# read-only runtime never has to write .pyc files; copy instead of hardlinking (cache mount
# and target live on different filesystems); production dependencies only.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PYTHON_PREFERENCE=only-system

# Extra arguments for `uv sync`, e.g. `--build-arg UV_SYNC_ARGS="--extra litert"` to include the
# optional LiteRT runtime (CNN fallback, PLANO §5.4).
ARG UV_SYNC_ARGS=""

WORKDIR /app

# Layer 1 - third-party dependencies only. The workspace members are excluded
# (`--no-install-workspace`, as the uv guide recommends for workspaces), so this layer is only
# invalidated by a change to the lock file or to a pyproject.toml, not by source edits.
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=contract/pyproject.toml,target=contract/pyproject.toml \
    --mount=type=bind,source=api/pyproject.toml,target=api/pyproject.toml \
    uv sync --package trinet-api --locked --no-install-workspace --no-editable ${UV_SYNC_ARGS}

# Layer 2 - the workspace packages (trinet-contract + trinet-api), non-editable, so the final
# image does not depend on /app/api/src. `.dockerignore` keeps tests, notebooks and data out.
COPY pyproject.toml uv.lock ./
COPY contract/ contract/
COPY api/ api/
# hadolint ignore=SC2086
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --package trinet-api --locked --no-editable ${UV_SYNC_ARGS}

# --------------------------------------------------------------------------- runtime stage
FROM ${BASE_IMAGE} AS runtime

ARG IMAGE_SOURCE=""
ARG IMAGE_REVISION=""
ARG IMAGE_VERSION="0.0.0-dev"
ARG IMAGE_CREATED=""

# The API key is read from the file TRINET_API_KEY_FILE (default /run/secrets/api_key, mounted by
# compose `secrets`). It is intentionally not set here: it is a path, not a secret, but naming it
# in ENV trips secret scanners (Hadolint DL3064 / Trivy DS-0031) for no benefit.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRINET_HOST=0.0.0.0 \
    TRINET_PORT=8000 \
    TRINET_METRICS_PORT=9100 \
    TRINET_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

# Only the virtualenv crosses the stage boundary: no uv, no pip cache, no build tools, no sources.
COPY --from=build /app/.venv /app/.venv

# Model artifacts go last: they change with every notebook export, the venv does not.
COPY artifacts/ /app/artifacts/

# Numeric uid:gid 65532 (the `nonroot` convention of distroless/hardened images). Numeric because the
# slim base has no such account. The container filesystem is read-only; /tmp is a tmpfs (compose).
USER 65532:65532

# 8000 = public API. The Prometheus port (9100) is deliberately NOT exposed or published.
EXPOSE 8000

# stdlib-only probe of /health/live (no curl/wget needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "trinet_api.healthcheck"]

LABEL org.opencontainers.image.title="trinet-api" \
      org.opencontainers.image.description="TriNet inference API (FastAPI + ONNX Runtime), driven by artifact manifests" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="${IMAGE_SOURCE}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}"

CMD ["python", "-m", "trinet_api"]
