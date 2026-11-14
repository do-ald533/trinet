#!/usr/bin/env bash
# Boot smoke test for a built API image (no Python deps needed on the host).
#
# Starts the image with the same hardening as deploy/compose.yaml (read-only rootfs, tmpfs /tmp,
# all capabilities dropped, no-new-privileges, 2 CPUs / 1 GiB), then checks:
#   - the image runs as a non-root user and declares a HEALTHCHECK;
#   - /health/ready answers 200 (models loaded, SHA-256 verified, warm);
#   - /v1/models without an API key is rejected (401) and with the key answers 200.
#
# The API key is a throwaway random value in a temp file; it is never printed.
#
# Usage: .github/scripts/smoke-image.sh <image> [-- <command> [args...]]
#   Without a command: exits 0 when all checks pass.
#   With a command: after the checks pass it runs the command while the container is still up,
#   with TRINET_BASE_URL and TRINET_API_KEY exported (used by the weekly Schemathesis run), and
#   returns its exit status.
# Environment: SMOKE_PORT (default 18000); SMOKE_TMPDIR = where the throwaway key file lives (must be
# bind-mountable: Docker Desktop only shares some host paths, `make smoke` uses ~/.cache/trinet);
# TRINET_RATE_LIMIT is forwarded to the container if set (a long fuzzing run needs a higher limit
# than the production default).
set -euo pipefail

image="${1:?usage: $0 <image> [-- <command> [args...]]}"
shift
if [ "${1:-}" = "--" ]; then
  shift
fi
post_cmd=("$@")

port="${SMOKE_PORT:-18000}"
name="trinet-smoke-$$"
mkdir -p "${SMOKE_TMPDIR:-${TMPDIR:-/tmp}}"
tmp="$(mktemp -d "${SMOKE_TMPDIR:-${TMPDIR:-/tmp}}/trinet-smoke.XXXXXX")"
ok=false

cleanup() {
  if [ "$ok" != true ]; then
    echo "--- container logs (last 60 lines) ---" >&2
    docker logs --tail 60 "$name" >&2 || true
  fi
  docker rm -f "$name" >/dev/null 2>&1 || true
  rm -rf "$tmp"
}
trap cleanup EXIT

user="$(docker inspect --format '{{.Config.User}}' "$image")"
case "$user" in
  "" | 0 | 0:* | root | root:*) echo "image runs as root (User='$user')" >&2; exit 1 ;;
esac
health="$(docker inspect --format '{{if .Config.Healthcheck}}yes{{else}}no{{end}}' "$image")"
[ "$health" = yes ] || { echo "image has no HEALTHCHECK" >&2; exit 1; }
echo "image user=$user, healthcheck=yes"

key="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
printf '%s' "$key" > "$tmp/api_key"
chmod 0444 "$tmp/api_key"   # throwaway key; readable by the container's non-root uid

extra_env=()
if [ -n "${TRINET_RATE_LIMIT:-}" ]; then
  extra_env+=(--env "TRINET_RATE_LIMIT=${TRINET_RATE_LIMIT}")
fi

docker run --detach --name "$name" \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m,mode=1777 \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --cpus 2 --memory 1g --pids-limit 256 \
  --volume "$tmp/api_key:/run/secrets/api_key:ro" \
  --publish "127.0.0.1:${port}:8000" \
  "${extra_env[@]}" \
  "$image" >/dev/null

base="http://127.0.0.1:${port}"
code=000
for _ in $(seq 1 90); do
  code="$(curl --silent --output /dev/null --write-out '%{http_code}' "$base/health/ready" || true)"
  [ "$code" = 200 ] && break
  if [ "$(docker inspect --format '{{.State.Running}}' "$name")" != true ]; then
    echo "container exited before becoming ready" >&2
    exit 1
  fi
  sleep 1
done
[ "$code" = 200 ] || { echo "/health/ready never returned 200 (last: $code)" >&2; exit 1; }
echo "/health/ready -> 200"

code="$(curl --silent --output /dev/null --write-out '%{http_code}' "$base/v1/models")"
[ "$code" = 401 ] || { echo "/v1/models without key returned $code, expected 401" >&2; exit 1; }
echo "/v1/models (no key) -> 401"

code="$(curl --silent --output /dev/null --write-out '%{http_code}' -H "X-API-Key: $key" "$base/v1/models")"
[ "$code" = 200 ] || { echo "/v1/models with key returned $code, expected 200" >&2; exit 1; }
echo "/v1/models (key) -> 200"

echo "smoke checks passed for $image"

if [ "${#post_cmd[@]}" -gt 0 ]; then
  export TRINET_BASE_URL="$base" TRINET_API_KEY="$key"
  if "${post_cmd[@]}"; then
    ok=true
  else
    status=$?
    echo "command failed with status $status" >&2
    exit "$status"
  fi
else
  ok=true
fi
