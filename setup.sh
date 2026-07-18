#!/usr/bin/env bash
# Install/check the Apple Silicon build prerequisites, populate Cargo's cache,
# verify the committed verifier, and build the candidate. Safe to rerun.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
toolchain="${RUSTUP_TOOLCHAIN:-1.97.0}"
trusted="${root}/benchmark-tools/trusted/flock_benchmark_verifier"

die() { echo "setup.sh: $*" >&2; exit 1; }

retry() {
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    [[ "${attempt}" == 3 ]] && return 1
    echo "setup.sh: command failed; retrying (${attempt}/3): $*" >&2
    sleep "$((attempt * 5))"
  done
}

toolchain_is_installed() {
  local installed line
  while IFS= read -r line; do
    installed="${line%% *}"
    if [[ "${installed}" == "${toolchain}" || "${installed}" == "${toolchain}-"* ]]; then
      return 0
    fi
  done < <(rustup toolchain list 2>/dev/null || true)
  return 1
}

# The committed verifier is an arm64 Mach-O and the ranked sandbox is macOS
# Seatbelt, so unsupported hosts should fail before downloading anything.
[[ "$(uname -s)" == Darwin ]] || die "macOS is required"
[[ "$(uname -m)" == arm64 ]] || die "Apple Silicon (arm64) is required"

for dependency in git shasum codesign curl; do
  command -v "${dependency}" >/dev/null 2>&1 || die "${dependency} is required"
done
if ! command -v sandbox-exec >/dev/null 2>&1; then
  [[ "${FLOCK_REQUIRE_SANDBOX:-0}" == 1 ]] && die "sandbox-exec is required"
  echo "setup.sh: warning: sandbox-exec is missing; ranked runs will fail" >&2
fi

# Cargo and the cc crate need the macOS SDK plus a working Clang linker.
compiler=""
if [[ -n "${CC:-}" ]] && command -v "${CC}" >/dev/null 2>&1; then
  compiler="$(command -v "${CC}")"
elif command -v xcrun >/dev/null 2>&1; then
  compiler="$(xcrun --find clang 2>/dev/null || true)"
fi
if [[ -z "${compiler}" || ! -x "${compiler}" ]] || ! "${compiler}" --version >/dev/null 2>&1; then
  if [[ -t 0 ]] && command -v xcode-select >/dev/null 2>&1; then
    echo "setup.sh: requesting the Xcode Command Line Tools installer" >&2
    xcode-select --install 2>/dev/null || true
  fi
  die "install Xcode Command Line Tools, wait for completion, then rerun ./setup.sh"
fi
export CC="${compiler}"

# Install Rustup when the host has no managed Rust installation. This mirrors
# rustup's official TLS-only installer and then reloads its environment.
cargo_env="${CARGO_HOME:-${HOME}/.cargo}/env"
# shellcheck disable=SC1090
[[ ! -f "${cargo_env}" ]] || . "${cargo_env}"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 --retry 5 --retry-delay 2 --retry-connrefused \
    -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none \
    || die "failed to install Rustup; check outbound HTTPS/DNS"
  # shellcheck disable=SC1090
  . "${cargo_env}"
fi
command -v rustup >/dev/null 2>&1 || die "rustup is unavailable after installation"

# Avoid a network sync on every run: install the exact toolchain only when the
# requested channel/target tuple is absent.
if ! toolchain_is_installed; then
  retry rustup toolchain install "${toolchain}" --profile minimal \
    || die "failed to install Rust ${toolchain}"
fi
export RUSTUP_TOOLCHAIN="${toolchain}"
command -v cargo >/dev/null 2>&1 || die "cargo is unavailable for Rust ${toolchain}"

# Fail before compilation if the protected verifier bytes or signature differ.
(
  cd "${root}/benchmark-tools/trusted"
  shasum -a 256 -c SHA256SUMS
)
[[ -x "${trusted}" ]] || die "trusted verifier is not executable"
codesign --verify --strict "${trusted}" || die "trusted verifier signature is invalid"

export CARGO_INCREMENTAL=0
export CARGO_NET_RETRY="${CARGO_NET_RETRY:-10}"
export CARGO_HTTP_TIMEOUT="${CARGO_HTTP_TIMEOUT:-120}"
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"

# Fetch once with retries, then force the actual candidate build offline. This
# makes dependency failures occur in setup and prevents network use by Cargo
# during compilation.
retry cargo fetch --locked --manifest-path "${root}/Cargo.toml" \
  || die "failed to fetch the locked Cargo dependencies"
CARGO_NET_OFFLINE=true cargo build --locked --offline --profile challenge \
  --manifest-path "${root}/Cargo.toml" \
  --target-dir "${root}/target/challenge-candidate" \
  -p flock-benchmark-worker

echo "candidate worker:  current checkout"
echo "trusted verifier: committed binary (checksum verified)"
echo "Rust toolchain:    $(rustc --version)"
echo "C linker:          ${compiler}"
