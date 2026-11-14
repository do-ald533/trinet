#!/usr/bin/env bash
# Long black-box Schemathesis run against a running API (weekly workflow, PLANO §8.3 / §6).
# Expects TRINET_BASE_URL and TRINET_API_KEY (exported by smoke-image.sh) and `uvx` on PATH.
# Tunables: SCHEMATHESIS_EXAMPLES (default 300), SCHEMATHESIS_MAX_TIME seconds (default 1200).
set -euo pipefail

: "${TRINET_BASE_URL:?TRINET_BASE_URL is required}"
: "${TRINET_API_KEY:?TRINET_API_KEY is required}"

exec uvx --from schemathesis==4.33.2 schemathesis run "$TRINET_BASE_URL/openapi.json" \
  --url "$TRINET_BASE_URL" \
  --header "X-API-Key: $TRINET_API_KEY" \
  --phases examples,coverage,fuzzing \
  --max-examples "${SCHEMATHESIS_EXAMPLES:-300}" \
  --max-time "${SCHEMATHESIS_MAX_TIME:-1200}" \
  --workers 2 \
  --checks not_a_server_error,status_code_conformance,content_type_conformance,response_headers_conformance,response_schema_conformance,ignored_auth \
  --no-color
