#!/usr/bin/env bash
# Install pinned security/lint CLIs (linux x86_64) from their official GitHub releases, verifying a
# SHA-256 that is pinned HERE, in the repository (not read from the same release page that ships
# the binary). This is the lesson of CVE-2026-33634 (trivy-action tags rewritten): neither a tag nor
# a release page is trusted; the reviewed checksum in git is.
#
# Usage:   .github/scripts/install-cli.sh <tool> [<tool>...]
# Tools:   hadolint trivy osv-scanner gitleaks zizmor actionlint
# Install dir: $TRINET_BIN_DIR, else $RUNNER_TEMP/bin (CI), else ~/.local/bin. In GitHub Actions
# the directory is appended to $GITHUB_PATH.
#
# Bumping a tool: change `version`, then refresh `sha256` from the vendor checksum file
# (hadolint: checksums.sha256, trivy: trivy_<v>_checksums.txt, osv-scanner: osv-scanner_SHA256SUMS,
# gitleaks: gitleaks_<v>_checksums.txt, actionlint: actionlint_<v>_checksums.txt) and cross-check
# with `sha256sum` on the downloaded file. zizmor publishes no checksum file: its hash was computed
# from a TLS download of the release asset (trust on first use) - review it when bumping.
set -euo pipefail

if [ "$(uname -s)-$(uname -m)" != "Linux-x86_64" ]; then
  echo "install-cli.sh supports linux x86_64 only (got $(uname -s)-$(uname -m))" >&2
  exit 1
fi

bin_dir="${TRINET_BIN_DIR:-}"
if [ -z "$bin_dir" ] && [ -n "${RUNNER_TEMP:-}" ]; then
  bin_dir="${RUNNER_TEMP}/bin"
fi
bin_dir="${bin_dir:-$HOME/.local/bin}"
mkdir -p "$bin_dir"

# Sets: version url sha256 member (member = path of the binary inside the archive; empty = the
# download itself is the binary).
spec() {
  case "$1" in
    hadolint)
      version=2.15.1
      url="https://github.com/hadolint/hadolint/releases/download/v${version}/hadolint-linux-x86_64"
      sha256=c7187db94eeeeca956519a6af171adc31453941a1e777961f6e680f697c8c507
      member=""
      ;;
    trivy)
      version=0.76.2
      url="https://github.com/aquasecurity/trivy/releases/download/v${version}/trivy_${version}_Linux-64bit.tar.gz"
      sha256=bc0fc726e017b8927a62b2d3531d7a84ee01769daf5c99ed866dced54089fdbf
      member=trivy
      ;;
    osv-scanner)
      version=2.7.1
      url="https://github.com/google/osv-scanner/releases/download/v${version}/osv-scanner_linux_amd64"
      sha256=fb7dab19a8bd9bbe5c3189f09d4b7ee1820c9553a23bac202648a56efd2f6ee8
      member=""
      ;;
    gitleaks)
      version=8.30.1
      url="https://github.com/gitleaks/gitleaks/releases/download/v${version}/gitleaks_${version}_linux_x64.tar.gz"
      sha256=551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb
      member=gitleaks
      ;;
    zizmor)
      version=1.32.0
      url="https://github.com/zizmorcore/zizmor/releases/download/v${version}/zizmor-x86_64-unknown-linux-gnu.tar.gz"
      sha256=bb110cc765eca9cd7efb2e284a956be5974c60027b0970f16397d86208a9957b
      member=zizmor
      ;;
    actionlint)
      version=1.7.12
      url="https://github.com/rhysd/actionlint/releases/download/v${version}/actionlint_${version}_linux_amd64.tar.gz"
      sha256=8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8
      member=actionlint
      ;;
    *)
      echo "unknown tool: $1 (known: hadolint trivy osv-scanner gitleaks zizmor actionlint)" >&2
      return 1
      ;;
  esac
}

install_one() {
  local tool="$1" version url sha256 member tmp
  spec "$tool"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  curl --fail --silent --show-error --location --retry 3 --retry-delay 2 \
    --proto '=https' --tlsv1.2 --output "$tmp/download" "$url"
  echo "${sha256}  $tmp/download" | sha256sum --check --status \
    || { echo "SHA-256 mismatch for ${tool} ${version} (${url})" >&2; return 1; }

  if [ -n "$member" ]; then
    tar --extract --gzip --file "$tmp/download" --directory "$tmp" "$member"
    install -m 0755 "$tmp/$member" "$bin_dir/$tool"
  else
    install -m 0755 "$tmp/download" "$bin_dir/$tool"
  fi
  echo "installed ${tool} ${version} -> ${bin_dir}/${tool}"
}

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <tool>..." >&2
  exit 2
fi
for tool in "$@"; do
  install_one "$tool"
done

if [ -n "${GITHUB_PATH:-}" ]; then
  echo "$bin_dir" >> "$GITHUB_PATH"
fi
