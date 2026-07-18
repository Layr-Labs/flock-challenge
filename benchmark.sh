#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${root}"

# Ranked defaults: 2^16 = 65,536 compressions, best of three trials.
# Environment overrides exist only for local smoke tests and diagnostics.
log2_size="${BLAKE3_LOG2:-16}"
threads="${BLAKE3_THREADS:-auto}"
runs="${BLAKE3_RUNS:-3}"
output_dir="${BENCHMARK_OUTPUT_DIR:-benchmark-results}"

# Prefer Apple performance cores. This configures the normal Rayon pool; it is
# not a hard CPU quota on solver-modified code.
if [[ "${threads}" == auto ]]; then
  threads="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"
  [[ -n "${threads}" ]] || threads="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi

# The worker links solver-editable source. The verifier is the committed,
# solver-protected binary; verify its exact bytes again immediately before use.
worker="${root}/target/challenge-candidate/challenge/flock-benchmark-worker"
verifier="${root}/benchmark-tools/trusted/flock_benchmark_verifier"
[[ -x "${worker}" && -x "${verifier}" ]] || {
  echo "benchmark binaries are missing; run ./setup.sh" >&2
  exit 1
}
(
  cd "${root}/benchmark-tools/trusted"
  shasum -a 256 -c SHA256SUMS
)

# Canonicalize the output path before embedding it in the Seatbelt profile.
# Remove stale scores first so a failed run can never upload an earlier result.
mkdir -p "${output_dir}/scratch"
output_dir="$(cd "${output_dir}" && pwd -P)"
scratch="${output_dir}/scratch"
rm -f score.json "${output_dir}/score.json" "${output_dir}/summary.md"

sandbox_profile=""
cleanup() { [[ -z "${sandbox_profile}" ]] || rm -f "${sandbox_profile}"; }
trap cleanup EXIT

# Only the candidate worker enters this profile. It may read the system, but it
# cannot use the network, create descendants, or write outside private scratch.
if [[ "$(uname -s)" == Darwin ]] && command -v sandbox-exec >/dev/null; then
  [[ "${scratch}" != *'"'* && "${scratch}" != *$'\n'* ]] || {
    echo "scratch path cannot contain quotes or newlines" >&2
    exit 1
  }
  sandbox_profile="$(mktemp -t flock-benchmark.XXXXXX.sb)"
  printf '%s\n' \
    '(version 1)' \
    '(allow default)' \
    '(deny network*)' \
    '(deny process-fork)' \
    '(deny file-write*)' \
    "(allow file-write* (subpath \"${scratch}\"))" \
    > "${sandbox_profile}"
elif [[ "${FLOCK_REQUIRE_SANDBOX:-0}" == 1 ]]; then
  echo "sandbox-exec is required for the ranked benchmark" >&2
  exit 1
else
  echo "WARNING: worker is not sandboxed (local development only)" >&2
fi

# The trusted verifier owns the private seed, external timer, proof checking,
# and score writing. It launches one fresh sandboxed worker per trial.
args=("${worker}" "${scratch}" "${root}/score.json" "${output_dir}/summary.md"
  "${log2_size}" "${threads}" "${runs}")
[[ -z "${sandbox_profile}" ]] || args+=("${sandbox_profile}")
"${verifier}" "${args[@]}"

# Reaching here means every timed proof passed pristine verification.
# Keep the root score for Yukon and copy it into the diagnostic artifact.
cp score.json "${output_dir}/score.json"
printf -- '- Candidate commit: `%s`\n' "$(git rev-parse HEAD)" >> "${output_dir}/summary.md"
cat "${output_dir}/summary.md"
