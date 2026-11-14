# CI/CD and security gates

Single source of truth for the commands is the `Makefile`; workflows call the same targets.
All third-party actions are pinned to a full commit SHA with a `# vX.Y.Z` comment, every workflow starts from
`permissions: {}` and grants per job, and checkouts use `persist-credentials: false`.

## Workflows

| File | Trigger | What it does |
|---|---|---|
| `workflows/ci.yml` | pull request, push to main | static (ruff, mypy, import-linter, deptry), tests + coverage >= 85%, contract-schema drift, artifact validation (`tools/check_artifacts.py`), actionlint, zizmor, OSV-Scanner, gitleaks, hadolint + `trivy config`, image build + boot smoke + Trivy image scan, free-flows check, notebook lint. `ci-gate` is the single required check. |
| `workflows/cd.yml` | push to main, manual | build once, push to GHCR by digest, Trivy on the digest, CycloneDX SBOM, SLSA provenance + SBOM attestations, verify them, Testcontainers smoke on the digest, release gate, promote `latest` (and the version tag on a manual release). |
| `workflows/security-weekly.yml` | Monday 05:17 UTC, manual | full OSV scan of both lockfiles, re-scan of published images with unfixed findings, Schemathesis fuzzing against a booted container, and a single rolling `security-weekly` issue (opened, commented, or closed). |
| `dependabot.yml` | Dependabot | uv (root weekly, notebook monthly), docker, docker-compose, github-actions, pre-commit; 7-day cooldown on every ecosystem; security updates ignore the cooldown. |

`SECURITY_GATE` (repository variable, default `block`): set to `warn` to turn the blocking scanners (OSV, gitleaks,
zizmor, hadolint/Trivy) into annotations while a finding is triaged. `ci-gate` still fails on any other failure.

## One-time GitHub setup (not automatable from the repo)

1. **Ruleset on `main`** (Settings > Rules): require pull requests, require status check **`ci-gate`**, require
   review from Code Owners, block force pushes and deletions, require linear history.
2. **Code security** (Settings > Advanced Security): enable Dependency graph, Dependabot alerts and security updates,
   Secret scanning with **push protection**, and **CodeQL default setup** for `python` and `actions` (query suite:
   default). Then add a ruleset rule "Require code scanning results" with tool CodeQL, security alerts "High or higher".
   There is deliberately no `codeql.yml`: default setup is GitHub-maintained, needs no pinned workflow, and a ruleset
   makes it blocking.
3. **Actions** (Settings > Actions > General): allow only GitHub-owned and verified actions, set the workflow token
   default to read-only, require approval for outside-collaborator runs. The workflows work with SHA-pinned actions only.
4. **Variables** (Settings > Secrets and variables > Actions > Variables): optionally `SECURITY_GATE` (`block` or
   `warn`; unset means `block`). It is the only variable the workflows read.
5. **Secrets**: none. No workflow reads a secret other than the built-in `GITHUB_TOKEN` (see "Free-only policy").
6. **GHCR package**: after the first publish set the `trinet-api` package visibility (public for a public repo) and link
   it to the repository so `GITHUB_TOKEN` keeps push access.
7. **Placeholder to replace**: the owner handle in `.github/CODEOWNERS`.
8. **Local**: `pip install pre-commit` (or `uvx prek install`), then `make tools` for the pinned CLIs. No login is needed.

## Free-only policy

Every flow must run for free on a public repository, with no account, token or paid plan beyond the built-in
`GITHUB_TOKEN` ([ADR 0012](../docs/adr/0012-fluxos-gratuitos-e-sem-credenciais.md)). `make free-flows` fails the build when
a workflow reads a secret other than `GITHUB_TOKEN`, a variable other than `SECURITY_GATE`, logs in to a registry other
than `ghcr.io`, runs on anything but a standard GitHub-hosted Ubuntu runner, names a service that needs an account or a
plan (SonarQube/SonarCloud, Codecov, Snyk, dhi.io, ...), or when `deploy/compose.yaml` defaults to a base image that
needs a login. To allow something new on purpose, change the ADR and the allow-lists in `tools/check_free_flows.py`
in the same pull request. The one third-party service left is `step-security/harden-runner` in `cd.yml` (audit
mode, community tier, no account); it only observes, so it can be removed without touching the pipeline.

## Gate policy

| Check | Blocks merge | Notes |
|---|---|---|
| lint, typecheck, imports, deptry, tests + coverage, schema drift, artifacts | yes | |
| actionlint, zizmor (regular persona) | yes | online audits need `GH_TOKEN`; local runs use `--offline` |
| OSV-Scanner (lockfile) | yes | no `uv.lock` support in GitHub dependency review, so OSV is the SCA gate |
| gitleaks, hadolint, `trivy config` | yes | |
| Trivy image scan (HIGH/CRITICAL, fixed only) | yes | exceptions go in `.trivyignore.yaml` with an expiry and a reason |
| `free-flows` (`tools/check_free_flows.py`) | yes | runs in the `static` matrix; see the next section |
| notebook lint | no | informational (TensorFlow stack is out of scope) |

## Release flow

`push` to main publishes `sha-<12 chars>`. `latest` and the `vX.Y.Z` tag move only through a manual **CD > Run workflow**
with `release=true` and `version=1.0.0`, and only when `tools/check_artifacts.py --release` passes. Consumers pin the digest
and verify it with `make deploy` (see `deploy/README.md`).
