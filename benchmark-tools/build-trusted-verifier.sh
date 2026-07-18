#!/usr/bin/env bash
# Author-only reproducibility tool. Ranked setup never runs this script.
#
# It builds the reviewed harness against the original Flock source, then
# replaces the committed verifier binary and records its new checksum.
set -euo pipefail

readonly FLOCK_COMMIT=e2b1741f7f7d3d3fac3626688e0fd5bd05830bb0
readonly TOOLCHAIN=1.97.0

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkout="${root}/.trusted-benchmark"
target="${root}/target/trusted-author-build"
output="${root}/benchmark-tools/trusted"

# Materialize the reviewed source commit without modifying it.
if [[ ! -d "${checkout}/.git" && ! -f "${checkout}/.git" ]]; then
  git -C "${root}" worktree add --detach "${checkout}" "${FLOCK_COMMIT}"
else
  [[ -z "$(git -C "${checkout}" status --porcelain --untracked-files=all)" ]] || {
    echo "trusted checkout is not clean" >&2
    exit 1
  }
  git -C "${checkout}" checkout --detach "${FLOCK_COMMIT}"
fi
[[ "$(git -C "${checkout}" rev-parse HEAD)" == "${FLOCK_COMMIT}" ]] || {
  echo "trusted checkout is not ${FLOCK_COMMIT}" >&2
  exit 1
}
[[ -z "$(git -C "${checkout}" status --porcelain --untracked-files=all)" ]] || {
  echo "trusted checkout is not clean" >&2
  exit 1
}

# Build from inside the reviewed locked workspace. The subshell returns us to
# the caller's directory automatically.
rustup toolchain install "${TOOLCHAIN}" --profile minimal
(
  cd "${checkout}"
  CARGO_INCREMENTAL=0 RUSTFLAGS="-C target-cpu=apple-m1" \
    cargo "+${TOOLCHAIN}" build --locked --release \
    --target-dir "${target}" -p flock-benchmark-harness --bin flock_benchmark_harness
)

# Publish exact bytes consumed by setup.sh and benchmark.sh.
mkdir -p "${output}"
cp "${target}/release/flock_benchmark_harness" "${output}/flock_benchmark_verifier"
chmod 755 "${output}/flock_benchmark_verifier"
(
  cd "${output}"
  shasum -a 256 flock_benchmark_verifier > SHA256SUMS
)

echo "wrote ${output}/flock_benchmark_verifier"
cat "${output}/SHA256SUMS"
