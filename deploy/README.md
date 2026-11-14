# Deploying the TriNet API

One image, `deploy/Dockerfile`, serves every model found under `artifacts/` (FastAPI + ONNX Runtime,
driven by the manifests). `deploy/compose.yaml` runs it with production-grade hardening on a single host.

## Quick start (local)

```bash
make up                        # build + secrets/api_key + start on http://127.0.0.1:8000
curl -s localhost:8000/health/ready
curl -s -H "X-API-Key: $(cat secrets/api_key)" localhost:8000/v1/models
make observability             # also starts Prometheus on http://127.0.0.1:9090 (scrapes api:9100)
make down
```

No account, token or login is needed anywhere: the base image is the public, digest-pinned `python:3.13-slim-trixie` ([ADR 0012](../docs/adr/0012-fluxos-gratuitos-e-sem-credenciais.md)).

`make help` lists every target. `.env.example` documents the knobs (`TRINET_API_PORT`, `TRINET_RATE_LIMIT`, ...);
copy it to `.env` to override. The API settings themselves are `TRINET_*` variables (see `api/src/trinet_api/settings.py`).

## What the container enforces

| Control | Where |
|---|---|
| Non-root, numeric uid:gid 65532 | Dockerfile `USER` (numeric, because the slim base has no such account) |
| Read-only root filesystem, tmpfs `/tmp` (64 MiB) | compose `read_only`, `tmpfs` |
| All capabilities dropped, `no-new-privileges` | compose `cap_drop`, `security_opt` |
| 2 CPUs, 1 GiB, 256 pids | compose `cpus`, `mem_limit`, `pids_limit` |
| Port published on loopback only | `127.0.0.1:8000:8000` (put a TLS reverse proxy in front for remote access) |
| API key from a file, never an env var | Docker secret mounted at `/run/secrets/api_key` |
| Healthcheck without curl/shell | `python -m trinet_api.healthcheck` |
| No pip in the runtime | the Dockerfile `slim` stage removes pip (the base still ships a shell and apt) |

## Secrets

`make secrets` creates `secrets/api_key` (random, 43 characters, gitignored) if it is missing.
Compose (non-Swarm) bind-mounts the host file as is and **ignores `uid/gid/mode` on secrets**, so the file
must be readable by uid 65532: the helper creates it `0644` inside a `0700` directory (other host users
cannot traverse it). To rotate: write a new value into the file and `docker compose -f deploy/compose.yaml restart api`.
Never put the key in `.env`, in an image build argument or in the compose file.

## Base images

* Default and only base used by CI/CD: the `slim` stage of `deploy/Dockerfile`, the official
  `python:3.13-slim-trixie` pinned by digest (Debian 13, Python 3.13). `BASE_IMAGE=slim` / `BUILD_IMAGE=slim` resolve to
  that stage without a registry round trip, and Dependabot keeps the digest fresh because it is a literal `FROM`.
  It runs as 65532, with pip removed and Debian security updates applied at build time. It still ships a shell and apt,
  so it is not distroless.
* Optional, local only, never used by a workflow: Docker Hardened Images (`dhi.io/python:3.13` and
  `dhi.io/python:3.13-dev`; no shell in the runtime). They need `docker login dhi.io` with a free Docker account.
  Build with `make build BASE=dhi`, or print digest-pinned references for `.env` with `make dhi-digests`.
* Optional LiteRT runtime: `make build UV_SYNC_ARGS="--extra litert"`.

## Build context caveat (Docker Desktop on Linux)

Docker Desktop shares only `$HOME` with its VM. A repository outside `$HOME` (for example on another disk) builds
fine, but bind mounts of repository files fail with "mounts denied". The smoke test therefore keeps its throwaway key
under `~/.cache/trinet` (`SMOKE_TMPDIR`). For `make up` from such a path, add the directory under
Docker Desktop > Settings > Resources > File sharing, or use Docker Engine.

## Verified deploys

`cd.yml` publishes `ghcr.io/<owner>/trinet-api` by digest and attaches a SLSA build provenance and a CycloneDX SBOM
attestation. To deploy exactly what CI built and verify it first:

```bash
make deploy DEPLOY_IMAGE=ghcr.io/<owner>/trinet-api@sha256:<digest> GH_REPO=<owner>/trinet
```

That runs `gh attestation verify` twice (provenance and SBOM; signer workflow pinned to `cd.yml`, self-hosted runners
denied) and only then starts the container. `latest` moves only when the release gate passes (no dummy artifacts,
version >= 1.0.0), so until real models are committed use the `sha-<commit>` tags or the digest.

## Observability

`--profile observability` adds Prometheus (digest-pinned, read-only, loopback-only) scraping the API's dedicated
metrics port (`api:9100/metrics`, not published to the host). No Grafana dashboard is shipped: it would depend on metric
names the contract does not freeze yet. Existing series: `trinet_http_request_duration_seconds_*`,
`trinet_http_inflight_requests` and the standard process metrics.
