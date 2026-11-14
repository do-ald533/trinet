"""The CI/CD flows must stay free and credential-free (ADR 0012)."""

from pathlib import Path

import pytest
from check_free_flows import REPO_ROOT, scan_compose, scan_repo, scan_workflow

CLEAN = """\
name: x
jobs:
  build:
    runs-on: ubuntu-24.04
    continue-on-error: ${{ vars.SECURITY_GATE == 'warn' }}
    steps:
      - uses: docker/login-action@abc # v4
        with:
          registry: ghcr.io
          password: ${{ secrets.GITHUB_TOKEN }}
      - run: echo "${{ github.token }}"
"""


def test_repository_flows_are_free() -> None:
    assert scan_repo(REPO_ROOT) == []


def test_clean_workflow_has_no_violation() -> None:
    assert scan_workflow("clean.yml", CLEAN) == []


@pytest.mark.parametrize(
    ("line", "needle"),
    [
        ("      TOKEN: ${{ secrets.SONAR_TOKEN }}", "SONAR_TOKEN"),
        ("      TOKEN: ${{ secrets['DHI_TOKEN'] }}", "dynamic or inherited"),
        ("    secrets: inherit", "dynamic or inherited"),
        ("      BASE: ${{ vars.DHI_BASE_IMAGE }}", "DHI_BASE_IMAGE"),
        ("          registry: dhi.io", "registry 'dhi.io'"),
        ("          registry: docker.io", "registry 'docker.io'"),
        ("    runs-on: ubuntu-latest-8-cores", "ubuntu-latest-8-cores"),
        ("    runs-on: self-hosted", "self-hosted"),
        ("      - uses: SonarSource/sonarqube-scan-action@abc", "needs an account"),
        ("      - uses: codecov/codecov-action@abc", "codecov"),
    ],
)
def test_violations_are_reported(line: str, needle: str) -> None:
    messages = scan_workflow("bad.yml", f"jobs:\n  a:\n{line}\n")
    assert messages, line
    assert any(needle in message for message in messages), messages


def test_comments_are_not_scanned() -> None:
    text = (
        "# secrets.SONAR_TOKEN and dhi.io are only mentioned here\n"
        "jobs:\n  a:\n    runs-on: ubuntu-24.04 # not self-hosted\n"
    )
    assert scan_workflow("c.yml", text) == []


def test_compose_default_base_must_not_need_a_login() -> None:
    assert scan_compose("compose.yaml", "BASE_IMAGE: ${BASE_IMAGE:-slim}\n") == []
    assert scan_compose("compose.yaml", "BASE_IMAGE: ${BASE_IMAGE:-dhi.io/python:3.13}\n")


def test_scan_repo_reads_the_files_under_the_root(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text("jobs:\n  a:\n    runs-on: self-hosted\n", encoding="utf-8")
    deploy = tmp_path / "deploy"
    deploy.mkdir()
    (deploy / "compose.yaml").write_text("x: dhi.io/python:3.13\n", encoding="utf-8")
    found = scan_repo(tmp_path)
    assert any(".github/workflows/ci.yml:3" in message for message in found)
    assert any("deploy/compose.yaml:1" in message for message in found)
