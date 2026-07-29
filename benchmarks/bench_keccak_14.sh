#!/usr/bin/env bash
# bench_keccak_14.sh — regenerate the Keccak 2^14 benchmark for EVERY system
# (flock, flock-slim, flock-secure, binius64, plonky3, hashcaster), multi-threaded,
# refreshing bench-keccak-cache, with a 20s cooldown between every benchmark.
# flock-secure is the 3-wide Flock prover at the audited 120-bit-security configs
# (~120-bit provable, vs flock/flock-slim's ~100).
#
# Rather than reimplement each system's run/parse logic, this delegates each
# (system, thread-mode) measurement to bench_keccak.sh — one system+mode per
# invocation — with NO_CACHE=1 so every run is fresh and OVERWRITES its cache
# row. All the *_EXTRA size knobs are nulled and ST_LOG2S is emptied so each
# invocation touches exactly 2^14 and nothing else.
#
# Six benchmarks (one per system, multi-threaded), so five 20s cooldowns. The
# cooldowns live here (between invocations); bench_keccak.sh's own --cooldown is
# not used.
#
# Knobs:
#   COOLDOWN=20                              seconds between benchmarks (0 = off)
#   MODES="mt"                               add "st" to also do single-threaded
#   PROVERS="flock flock-slim flock-secure binius64 plonky3 hashcaster"
#   RAYON_NUM_THREADS=N                      override the multi-threaded core count
#
# Usage: ./bench_keccak_14.sh
#
# After it finishes, see the combined table with:
#   HASH_LOG2S=14 ST_LOG2S= FLOCK_MT_EXTRA= P3_MT_EXTRA= USE_CACHE=1 ./bench_keccak.sh

set -euo pipefail
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

H=14
COOLDOWN="${COOLDOWN:-20}"
PROVERS="${PROVERS:-flock flock-slim flock-secure binius64 plonky3 hashcaster}"
MODES="${MODES:-mt}"
[[ "$COOLDOWN" =~ ^[0-9]+$ ]] || { echo "COOLDOWN must be a non-negative integer (seconds), got '$COOLDOWN'" >&2; exit 1; }
[[ -x "$BASE/bench_keccak.sh" ]] || { echo "missing $BASE/bench_keccak.sh" >&2; exit 1; }

CACHE_DIR="$BASE/bench-keccak-cache"
# P-core count for naming/reading the multi-threaded cache rows (matches how
# bench_keccak.sh detects it, so the _t<N> files line up).
if [[ -n "${RAYON_NUM_THREADS:-}" ]]; then
	PCORES="$RAYON_NUM_THREADS"
else
	if [[ "$(uname -s)" == "Darwin" ]]; then PCORES="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || true)"; fi
	: "${PCORES:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
fi

# summary_row LABEL THREADS CACHEFILE — one formatted line from a cached row.
SUMMARY_FMT='  %-12s %-4s %11s %9s %11s %12s %13s\n'
summary_row() {
	local label="$1" threads="$2" f="$3" key name target count thr prove vfy sz mem
	if [[ ! -s "$f" ]]; then printf '  %-12s %-4s %11s\n' "$label" "$threads" "(no result)"; return; fi
	IFS=$'\t' read -r key name target count thr prove vfy sz mem < "$f"
	# shellcheck disable=SC2059
	printf "$SUMMARY_FMT" "$label" "$threads" "$thr" "${prove}s" "$vfy" "$sz" "$mem"
}

# Restrict bench_keccak.sh to exactly 2^14: pin every prover's per-mode size list
# to 2^H, drop the secondary single-thread pass (we drive threads ourselves), and
# null every per-prover extra-size knob. NO_CACHE forces a fresh run that rewrites
# the cache.
#
# NOTE: bench_keccak.sh drives each prover from its own *_MT_SIZES list, NOT from
# HASH_LOG2S (sweep_provers only reads $1 to tell mt from st). So setting
# HASH_LOG2S alone would NOT confine the sweep to 2^14 — every prover would still
# run its full default list (FLOCK_MT_SIZES="12 14 16", HC_MT_SIZES="14 16 18",
# …). The *_MT_SIZES overrides below are what actually pin it to 2^H.
common_env=(
	HASH_LOG2S="$H" ST_LOG2S= NO_CACHE=1
	FLOCK_MT_SIZES="$H" FLOCK_SLIM_MT_SIZES="$H" FLOCK_SECURE_MT_SIZES="$H"
	B64_MT_SIZES="$H" P3_MT_SIZES="$H" HC_MT_SIZES="$H"
	FLOCK_MT_EXTRA= B64_MT_EXTRA= P3_MT_EXTRA= P3_ST_EXTRA=
)

run_bench() {  # mode prover
	local mode="$1" prover="$2"
	if [[ "$mode" == st ]]; then
		env "${common_env[@]}" RAYON_NUM_THREADS=1 "$BASE/bench_keccak.sh" "$prover"
	else
		env "${common_env[@]}" "$BASE/bench_keccak.sh" "$prover"
	fi
}

echo "=== regenerating keccak 2^$H for [$PROVERS], modes [$MODES], ${COOLDOWN}s cooldowns ==="
first=true
for mode in $MODES; do
	for prover in $PROVERS; do
		if [[ "$first" == true ]]; then
			first=false
		elif (( COOLDOWN > 0 )); then
			echo
			echo "  cooldown: sleeping ${COOLDOWN}s before the next benchmark to let the machine cool..." >&2
			sleep "$COOLDOWN"
		fi
		echo
		echo "######################## $prover  ($mode, 2^$H) ########################"
		run_bench "$mode" "$prover"
	done
done

echo
echo "=== results (keccak 2^$H, from $CACHE_DIR) ==="
# shellcheck disable=SC2059
printf "$SUMMARY_FMT" "system" "thr" "throughput" "prove" "verify" "proof" "peak"
for mode in $MODES; do
	if [[ "$mode" == st ]]; then threads=1; else threads="$PCORES"; fi
	for prover in $PROVERS; do
		summary_row "$prover" "$threads" "$CACHE_DIR/${prover}_2^${H}_t${threads}"
	done
done

echo
echo "=== done: refreshed keccak 2^$H cache rows in $BASE/bench-keccak-cache ==="
echo "    combined table: HASH_LOG2S=14 ST_LOG2S= FLOCK_MT_EXTRA= P3_MT_EXTRA= USE_CACHE=1 ./bench_keccak.sh"
