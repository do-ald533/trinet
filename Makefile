# TriNet - developer and operator entry points. `make help` lists the targets.
# CI calls the same targets (single source of truth for the commands).

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# --- configuration ---------------------------------------------------------------------------
# All workspace packages + extras so mypy/pytest/ruff see contract, api and the optional LiteRT.
UVRUN       ?= uv run --all-packages --all-extras
# Lightweight runner for tools that only need trinet-contract (pydantic).
UVRUN_LIGHT ?= uv run --package trinet-contract

COMPOSE_FILE := deploy/compose.yaml
COMPOSE      := docker compose -f $(COMPOSE_FILE) $(if $(wildcard .env),--env-file .env,)

# Image built/used locally. `make deploy` takes DEPLOY_IMAGE instead.
IMAGE ?= trinet-api:local
# BASE=slim -> pinned python:3.13-slim-trixie (default; free, no credentials; used by CI/CD)
# BASE=dhi  -> optional Docker Hardened Images (needs `docker login dhi.io`; never used by CI/CD)
BASE  ?= slim
DHI_BASE_IMAGE  ?= dhi.io/python:3.13
DHI_BUILD_IMAGE ?= dhi.io/python:3.13-dev
# Extra `uv sync` arguments inside the image, e.g. UV_SYNC_ARGS="--extra litert"
UV_SYNC_ARGS ?=

ifeq ($(BASE),slim)
  BASE_ARGS := --build-arg BASE_IMAGE=slim --build-arg BUILD_IMAGE=slim
else ifeq ($(BASE),dhi)
  BASE_ARGS := --build-arg BASE_IMAGE=$(DHI_BASE_IMAGE) --build-arg BUILD_IMAGE=$(DHI_BUILD_IMAGE)
else
  $(error BASE must be 'dhi' or 'slim' (got '$(BASE)'))
endif

# Docker Desktop (Linux) bind-mounts only paths it shares ($HOME by default), so the smoke test keeps
# its throwaway key file under $HOME. Plain Docker Engine and CI accept any path.
SMOKE_TMPDIR ?= $(HOME)/.cache/trinet

# make deploy DEPLOY_IMAGE=ghcr.io/<owner>/trinet-api@sha256:<digest> [GH_REPO=<owner>/trinet]
DEPLOY_IMAGE ?=
DEPLOY_OWNER  = $(word 2,$(subst /, ,$(DEPLOY_IMAGE)))
GH_REPO      ?= $(DEPLOY_OWNER)/trinet

.PHONY: help install lint fmt typecheck imports deptry test cov integration schema schema-check dummy \
        check-artifacts check-artifacts-release free-flows model-cards model-cards-check build smoke secrets up down observability logs \
        deploy dhi-digests tools scan notebook clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# --- python workspace ------------------------------------------------------------------------
install: ## Sync the workspace (contract + api + dev tools) from uv.lock
	uv sync --locked --all-packages --all-extras

lint: ## ruff check + ruff format --check
	$(UVRUN) ruff check $(RUFF_ARGS) .
	$(UVRUN) ruff format --check .

fmt: ## Auto-fix: ruff format + ruff check --fix
	$(UVRUN) ruff format .
	$(UVRUN) ruff check --fix .

typecheck: ## mypy --strict (config in pyproject.toml)
	$(UVRUN) mypy contract api tools

imports: ## import-linter architecture contracts
	$(UVRUN) lint-imports

deptry: ## unused/missing/transitive dependency check of the API
	cd api && $(UVRUN) deptry src

free-flows: ## Fail if a CI/CD flow needs a credential or a paid service (ADR 0012)
	$(UVRUN_LIGHT) python tools/check_free_flows.py

test: ## pytest (fast, parallel)
	$(UVRUN) pytest -n auto

cov: ## pytest with coverage gate (>= 85%) and coverage.xml
	$(UVRUN) pytest -n auto --cov --cov-report=term-missing --cov-report=xml --cov-fail-under=85

# basetemp lives under SMOKE_TMPDIR because Docker Desktop (Linux) only bind-mounts shared paths ($HOME). On Docker
# Desktop also export TESTCONTAINERS_DOCKER_SOCKET_OVERRIDE=/var/run/docker.sock so the Ryuk reaper can mount the socket.
integration: ## Testcontainers smoke against IMAGE (needs Docker; IMAGE can be a digest reference)
	mkdir -p $(SMOKE_TMPDIR)
	TRINET_IMAGE=$(IMAGE) $(UVRUN) pytest -m integration api/tests/integration --basetemp=$(SMOKE_TMPDIR)/pytest-integration

schema: ## Regenerate contract/schema/*.json
	$(UVRUN) python -m trinet_contract.schema

schema-check: schema ## Fail if the committed contract schemas drifted
	git diff --exit-code -- contract/schema
	@test -z "$$(git status --porcelain -- contract/schema)" || { echo "untracked/changed schema files"; exit 1; }

dummy: ## Generate dummy artifacts (status "dummy") for pipeline development
	$(UVRUN) python tools/make_dummy_artifacts.py

check-artifacts: ## Validate manifests, golden files, SHA-256 and sizes under artifacts/
	$(UVRUN_LIGHT) python tools/check_artifacts.py

check-artifacts-release: ## Same + release gate (3 models, no dummy, latest >= 1.0.0)
	$(UVRUN_LIGHT) python tools/check_artifacts.py --release

model-cards: ## Regenerate docs/model-cards/*.md from the latest artifacts
	$(UVRUN) python tools/gen_model_card.py

model-cards-check: ## Fail if the committed model cards are out of date
	$(UVRUN) python tools/gen_model_card.py --check

# --- container -------------------------------------------------------------------------------
build: ## Build the API image (BASE=dhi|slim, IMAGE=name:tag)
	docker build --provenance=false --sbom=false -f deploy/Dockerfile $(BASE_ARGS) \
		--build-arg UV_SYNC_ARGS="$(UV_SYNC_ARGS)" \
		--build-arg IMAGE_REVISION="$$(git rev-parse --verify --quiet HEAD || echo unknown)" \
		--build-arg IMAGE_CREATED="$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
		-t $(IMAGE) .

smoke: ## Boot IMAGE with the production hardening flags and check readiness + auth
	SMOKE_TMPDIR=$(SMOKE_TMPDIR) .github/scripts/smoke-image.sh $(IMAGE)

secrets: ## Create secrets/api_key (random, gitignored) if it does not exist
	@if [ -f secrets/api_key ]; then echo "secrets/api_key already exists"; else \
		install -d -m 0700 secrets; umask 022; \
		python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > secrets/api_key; \
		echo "created secrets/api_key (0644 in a 0700 directory: the container user 65532 must read it)"; fi

up: build secrets ## Build and start the API on http://127.0.0.1:8000
	TRINET_IMAGE=$(IMAGE) $(COMPOSE) up -d --no-build --wait

down: ## Stop everything (including the observability profile)
	$(COMPOSE) --profile observability down

observability: secrets ## Start API + Prometheus (http://127.0.0.1:9090); build first with `make build`
	TRINET_IMAGE=$(IMAGE) $(COMPOSE) --profile observability up -d --no-build --wait

logs: ## Follow the API logs
	$(COMPOSE) logs -f api

deploy: ## Verify the attestations of DEPLOY_IMAGE (digest-pinned), then start it
	@case "$(DEPLOY_IMAGE)" in *@sha256:*) ;; *) \
		echo "usage: make deploy DEPLOY_IMAGE=ghcr.io/<owner>/trinet-api@sha256:<digest> [GH_REPO=<owner>/trinet]"; \
		exit 2;; esac
	gh attestation verify oci://$(DEPLOY_IMAGE) -R $(GH_REPO) \
		--signer-workflow $(GH_REPO)/.github/workflows/cd.yml --deny-self-hosted-runners
	gh attestation verify oci://$(DEPLOY_IMAGE) -R $(GH_REPO) \
		--signer-workflow $(GH_REPO)/.github/workflows/cd.yml --deny-self-hosted-runners \
		--predicate-type https://cyclonedx.org/bom
	$(MAKE) secrets
	TRINET_IMAGE=$(DEPLOY_IMAGE) $(COMPOSE) up -d --no-build --pull always --wait

dhi-digests: ## Optional (BASE=dhi): after `docker login dhi.io`, print digest-pinned DHI references for .env
	@for ref in $(DHI_BASE_IMAGE) $(DHI_BUILD_IMAGE); do \
		digest="sha256:$$(docker buildx imagetools inspect --raw $$ref | sha256sum | cut -d' ' -f1)"; \
		echo "$${ref}@$${digest}"; done

# --- security tooling (local mirror of the CI gates; each step skipped if the tool is missing) --
tools: ## Install the pinned, checksum-verified CLIs (hadolint trivy osv-scanner gitleaks zizmor actionlint)
	.github/scripts/install-cli.sh hadolint trivy osv-scanner gitleaks zizmor actionlint

scan: ## Run hadolint, trivy (config + image), osv-scanner, gitleaks, zizmor when installed
	@run() { name="$$1"; shift; if command -v "$$name" >/dev/null 2>&1; then echo "==> $$name"; "$$name" "$$@"; \
		else echo "==> $$name not installed (make tools) - skipped"; fi; }; \
	run hadolint --config .hadolint.yaml deploy/Dockerfile; \
	run trivy config --skip-check-update --severity HIGH,CRITICAL --exit-code 1 deploy; \
	run osv-scanner scan source --lockfile uv.lock; \
	run gitleaks dir . --no-banner --redact --exit-code 1; \
	run zizmor --offline .github; \
	if docker image inspect $(IMAGE) >/dev/null 2>&1; then \
		run trivy image --scanners vuln --severity HIGH,CRITICAL --ignore-unfixed --exit-code 1 \
			--db-repository public.ecr.aws/aquasecurity/trivy-db:2 --ignorefile .trivyignore.yaml $(IMAGE); \
	else echo "==> image $(IMAGE) not built - skipped (make build)"; fi

# --- notebook (independent uv project) ---------------------------------------------------------
notebook: ## Open JupyterLab in the notebook project
	cd notebook && uv run jupyter lab

clean: ## Remove caches and coverage output
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov coverage.xml .coverage .coverage.*
