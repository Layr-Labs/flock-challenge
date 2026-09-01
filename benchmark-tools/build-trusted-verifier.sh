#!/usr/bin/env bash
# Author-only reproducibility tool. Ranked setup never runs this script.
#
# It builds the reviewed harness against the original Flock source, then
# replaces the committed verifier binary and records its new checksum.
set -euo pipefail

readonly REVIEWED_COMMIT=7b3c050dcc07ab9945899947b4c3fcf974fd21b8
readonly TOOLCHAIN=1.97.0

# `x86-64-v3` sets the portability floor (AVX2). It does NOT imply the
# carry-less-multiply extensions, which are a separate psABI-optional bundle, so
# `+pclmulqdq,+aes` must be requested explicitly: without them every GF(2^128)
# multiply in flock-core falls back to `field/gf2_128/portable.rs`, a bit-by-bit
# 64-iteration clmul64 loop. That fallback costs the verifier ~43 s per trial
# instead of ~2.2 s (the ARM build got the same kernels for free, because
# `-C target-cpu=apple-m1` includes the ARMv8 `aes`/PMULL extension).
#
# This does not narrow the supported CPU set: PCLMULQDQ and AES-NI shipped in
# Westmere (2010) and Bulldozer (2011), so every CPU that can execute the AVX2
# baseline this binary already requires also has them. Deliberately NOT enabling
# SHA-NI (no measurable gain here: the Merkle hash is BLAKE3 and the `sha2`
# crate does its own runtime dispatch) or AVX-512/VPCLMULQDQ (worth ~6%, but
# only on Zen 4 / Ice Lake-SP and newer).
readonly TARGET_FLAGS="-C target-cpu=x86-64-v3 -C target-feature=+pclmulqdq,+aes"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkout="${root}/.trusted-benchmark"
target="${root}/target/trusted-author-build"
output="${root}/benchmark-tools/trusted"

# Materialize the reviewed source commit without modifying it.
if [[ ! -d "${checkout}/.git" && ! -f "${checkout}/.git" ]]; then
  git -C "${root}" worktree add --detach "${checkout}" "${REVIEWED_COMMIT}"
else
  [[ -z "$(git -C "${checkout}" status --porcelain --untracked-files=all)" ]] || {
    echo "trusted checkout is not clean" >&2
    exit 1
  }
  git -C "${checkout}" checkout --detach "${REVIEWED_COMMIT}"
fi
[[ "$(git -C "${checkout}" rev-parse HEAD)" == "${REVIEWED_COMMIT}" ]] || {
  echo "trusted checkout is not ${REVIEWED_COMMIT}" >&2
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
  CARGO_INCREMENTAL=0 RUSTFLAGS="${TARGET_FLAGS}" \
    cargo "+${TOOLCHAIN}" build --locked --release \
    --target-dir "${target}" -p flock-benchmark-harness --bin flock_benchmark_harness
)

# Publish exact bytes consumed by setup.sh and benchmark.sh.
mkdir -p "${output}"
cp "${target}/release/flock_benchmark_harness" "${output}/flock_benchmark_verifier"
chmod 755 "${output}/flock_benchmark_verifier"
(
  cd "${output}"
  sha256sum flock_benchmark_verifier > SHA256SUMS
)

echo "wrote ${output}/flock_benchmark_verifier"
cat "${output}/SHA256SUMS"
