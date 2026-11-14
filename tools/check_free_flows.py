"""Fail when a CI/CD flow depends on something that is not free or needs a credential (ADR 0012).

Every flow of the project must run for free on a public GitHub repository, with no account, token
or paid plan beyond the built-in ``GITHUB_TOKEN``. This checker keeps that true:

* ``secrets.*``: only ``GITHUB_TOKEN`` is allowed (``github.token`` is a context, not a secret);
* ``vars.*``: only ``SECURITY_GATE`` (a non-sensitive switch with a safe default);
* container registries: only ``ghcr.io`` (free for public packages, built-in token login);
* runners: only the standard GitHub-hosted Ubuntu labels (larger and self-hosted runners cost money
  or need infrastructure);
* no third-party SaaS that needs a sign-up or a plan: SonarQube/SonarCloud, Codecov, Snyk, Docker
  Hardened Images (``dhi.io``), and the like;
* the default container base image of ``deploy/compose.yaml`` must not need a login.

Usage: ``python tools/check_free_flows.py`` from anywhere; exit status 1 lists every violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_SECRETS = frozenset({"GITHUB_TOKEN"})
ALLOWED_VARS = frozenset({"SECURITY_GATE"})
ALLOWED_REGISTRIES = frozenset({"ghcr.io"})
RUNNER_PATTERN = re.compile(r"^ubuntu-(?:\d{2}\.\d{2}|latest)(?:-arm)?$")
# Services that need an account, a token or a paid plan. Matched case-insensitively as substrings.
FORBIDDEN_SERVICES = (
    "dhi.io",
    "sonarcloud",
    "sonarqube",
    "sonarsource",
    "codecov",
    "snyk",
    "coveralls",
    "circleci",
    "datadog",
    "jfrog",
    "cloudsmith",
    "dockerhub",
)

SECRET_RE = re.compile(r"\bsecrets\.([A-Za-z_][A-Za-z0-9_]*)")
SECRETS_INDEX_RE = re.compile(r"\bsecrets\[")
VAR_RE = re.compile(r"\bvars\.([A-Za-z_][A-Za-z0-9_]*)")
REGISTRY_RE = re.compile(r"^\s*registry:\s*['\"]?([^\s'\"#]+)")
RUNS_ON_RE = re.compile(r"^\s*runs-on:\s*['\"]?([^\s'\"#]+)")
INHERIT_RE = re.compile(r"^\s*secrets:\s*inherit\b")
COMMENT_TAIL_RE = re.compile(r"\s#.*$")


def _code_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(line number, line)`` pairs with full-line and trailing YAML comments removed."""
    lines: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if raw.lstrip().startswith("#"):
            continue
        lines.append((number, COMMENT_TAIL_RE.sub("", raw)))
    return lines


def scan_workflow(name: str, text: str) -> list[str]:
    """Return one message per violation found in the text of one workflow file."""
    violations: list[str] = []
    for number, line in _code_lines(text):
        where = f"{name}:{number}"
        violations.extend(
            f"{where}: secret '{found}' needs a credential (only GITHUB_TOKEN is allowed)"
            for found in SECRET_RE.findall(line)
            if found not in ALLOWED_SECRETS
        )
        if SECRETS_INDEX_RE.search(line) or INHERIT_RE.match(line):
            violations.append(f"{where}: dynamic or inherited secrets are not allowed")
        violations.extend(
            f"{where}: variable '{found}' is not on the allow-list {sorted(ALLOWED_VARS)}"
            for found in VAR_RE.findall(line)
            if found not in ALLOWED_VARS
        )
        registry = REGISTRY_RE.match(line)
        if registry and registry.group(1) not in ALLOWED_REGISTRIES:
            violations.append(f"{where}: registry '{registry.group(1)}' is not free of credentials")
        runner = RUNS_ON_RE.match(line)
        if runner and not RUNNER_PATTERN.match(runner.group(1)):
            violations.append(
                f"{where}: runner '{runner.group(1)}' is not a standard free Ubuntu runner"
            )
        lowered = line.lower()
        violations.extend(
            f"{where}: '{service}' is a service that needs an account or a paid plan"
            for service in FORBIDDEN_SERVICES
            if service in lowered
        )
    return violations


def scan_compose(name: str, text: str) -> list[str]:
    """The default base image of the compose build must not need a login."""
    return [
        f"{name}:{number}: the default base image needs a login ({line.strip()})"
        for number, line in _code_lines(text)
        if "dhi.io" in line
    ]


def scan_repo(root: Path = REPO_ROOT) -> list[str]:
    """Scan every workflow, composite action and the compose file of the repository at ``root``."""
    violations: list[str] = []
    github = root / ".github"
    files = sorted([*github.glob("workflows/*.y*ml"), *github.glob("actions/**/*.y*ml")])
    for path in files:
        violations.extend(
            scan_workflow(str(path.relative_to(root)), path.read_text(encoding="utf-8"))
        )
    compose = root / "deploy" / "compose.yaml"
    if compose.is_file():
        violations.extend(
            scan_compose(str(compose.relative_to(root)), compose.read_text(encoding="utf-8"))
        )
    return violations


def main() -> int:
    violations = scan_repo()
    for violation in violations:
        print(f"error: {violation}", file=sys.stderr)
    if violations:
        print(
            f"{len(violations)} violation(s): flows must be free and credential-free (ADR 0012)",
            file=sys.stderr,
        )
        return 1
    print("free-flows: ok (only GITHUB_TOKEN, ghcr.io, standard runners, no paid service)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
