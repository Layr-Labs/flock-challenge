# ITER4 EVIDENCE LEDGER — angel lane (ISA-fidelity forensics + first in-place fold)

Direction: the relationship between the LOCAL BINARY and the RANKED RUNNER
BINARY. All three prior directions assumed local preflight measures the
runner's program; this iteration PROVES it does not.

## A. The AVX-512 code path is compile-gated, and this box compiles it OUT

- crates/flock-core/src/zerocheck/multilinear/kernels/x86_64.rs:19 gates the
  round-2 fold kernel on `#[cfg(all(target_arch="x86_64",
  target_feature="avx512f", target_feature="vpclmulqdq"))]` — a COMPILE-TIME
  cfg, not runtime dispatch: `rg -l is_x86_feature_detected` over
  crates/flock-core/src returns nothing.
- `rg -c avx512 crates/flock-core/src --glob '*.rs'`: 156 hits in
  zerocheck/univariate_skip_optimized/kernels/x86_64.rs, 117 in
  pcs/ligerito.rs, 113 in zerocheck/univariate_skip_optimized.rs, 106 in
  zerocheck/multilinear.rs, 98 in multilinear/kernels/x86_64.rs, 63 in
  lincheck.rs, 41 in ntt/inv_table.rs — all behind the same cfg family.
- benchmark.sh builds with `RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"`
  and .cargo/config.toml carries `rustflags = ["-C","target-cpu=native"]`.
- This box is a Ryzen 9 5950X (Zen 3): `/proc/cpuinfo` contains ZERO
  `avx512*` feature lines. `target-cpu=native` enables neither avx512f nor
  vpclmulqdq, so every `#[cfg(target_feature=...)]` kernel compiles OUT.
  The local worker is a PORTABLE-ONLY program; the ranked runner builds a
  different program with the SIMD kernels enabled.

Consequence: the 17-hang / 18-invalid-proof bugs (ITER2 #9-11) are bugs in a
program the runner never executes, AND every local perf number (incl. the
+0.28% memoization delta at 16) is a portable-path measurement whose transfer
to the runner's AVX-512 build is unproven. Local preflight is a
correctness-of-logic oracle only.

## B. The AVX-512 cfg set has a COMPILE-GATE DEPENDENCY (runner fingerprint)

Building flock-core with +avx512f,+vpclmulqdq ALONE fails: E0425 `cannot find
value tr_emit` at zerocheck/multilinear/kernels/x86_64.rs:539 — the
definition at :323 is `#[cfg(all(target_feature="avx512vbmi",
target_feature="gfni"))]`-gated. The full feature set that compiles clean on
pinned 1.97.0 is: avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,
gfni,vpclmulqdq (plus target-cpu=native). Since the ranked runner's build
compiles this code, the runner host must expose avx512vbmi+gfni → Ice Lake or
newer. Build command (works):

  CARGO_ENCODED_RUSTFLAGS via RUSTFLAGS + RUSTC=.scratch/rustc-noavx512.sh
  RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f,+avx512vl,
    +avx512bw,+avx512dq,+avx512vbmi,+avx512vbmi2,+gfni,+vpclmulqdq"
  cargo +1.97.0 build --profile challenge --target-dir target/challenge-avx512
    -p flock-benchmark-worker

Artifact: target/challenge-avx512/challenge/flock-benchmark-worker — 84,495
zmm instruction refs vs 2,018 in the portable candidate build (the portable's
zmm are blake3's own runtime-detected asm only). The rustc wrapper
(.scratch/rustc-noavx512.sh) strips the avx512 -C pairs from every unit
except flock_core/flock_prover/flock_benchmark_*: build scripts and proc
macros EXECUTE on this host and SIGILL otherwise (blake3/generic-array/
sha2-asm build scripts all crashed before the wrapper existed).

## C. Qemu-user oracle: protocol works, full AVX-512 build does not run

- qemu-user-static 10.0.11 is a transitional empty package; qemu-user 11.1.1
  from Debian pool has the real static binary (dpkg-deb -x, no root needed):
  /tmp/qemu-y/usr/bin/qemu-x86_64. `-cpu max` advertises avx512f/vl/bw/dq/
  vbmi/vbmi2/gfni/vpclmulqdq.
- PORTABLE worker runs end-to-end under qemu at log2=12: ready at ~44s,
  proof file /tmp/qrun12p/proof (329,327 bytes) produced for seed 424242.
  (verify_file result to be recorded.)
- The AVX-512 build SIGILLs under qemu ~5s in, BEFORE the ready file:
  guest #UD, exec-trace tail shows the main thread inside
  std::rt::lang_start_internal at 0x555555930e1f..0eca. Cause not yet
  isolated (suspect TCG gap in one flock kernel instruction or an
  LTO/large-binary interaction). This blocks the qemu preflight at 17/18
  for now.
- Intel SDE mirrors all return 403 from this egress (probed 836947/833713/
  831112/813591/790775/789013/815639/847072).

## D. partial_eval_lsb in-place fold — verified-clean, perf-NEUTRAL

ligerito.rs partial_eval_lsb: fold in place (write index i <= read indices
2i, 2i+1), one allocation per call instead of 1+k. Oracle test
`cargo +1.97.0 test -p flock-core --lib --
partial_eval_then_eval_equals_full_eval` PASSES. A/B at 16 (trusted
verifier, direct, 1+2): fold build 58355.7 / 58787.0 cps; reverted build
58425.9 cps — all verified=true. Delta is noise; the fold is a strict work
deletion and is KEPT in the lane (it cannot regress the runner and removes
per-call allocator churn), but it does not clear the gate by itself.

## E. Harness lessons

- ANGEL_TOOL_TIMEOUT=1800 (env prefix on the command) extends the tool
  watchdog; a trailing `| tail -N` pipe starves it (tail buffers until EOF)
  and triggers the 30s no-output kill. Streaming output (periodic echo) is
  mandatory for runs >30s.
- Never leave background jobs across tool calls — they get reaped.
- benchmark.sh remains env-broken (bwrap); use the direct trusted-verifier
  invocation (ITER3-A).

## F. Next direction (highest value first)

1. Probe the LAN fleet seats (turbo/spark/atlas/gemma) for avx512 in
   /proc/cpuinfo: a REAL AVX-512 host makes the native preflight at 17/18
   possible (scp the challenge-avx512 worker + verify_file there). This
   bypasses the qemu SIGILL entirely.
2. If no fleet host: isolate the qemu-fatal instruction (SDE blocked;
   qemu `-one-insn-per-tb` with a tighter -dfilter window, or bisect by
   disabling individual cfg features in the wrapper).
3. Candidate ground remains: in-loop work deletions inside the runner's
   AVX-512 kernels (zerocheck fold / ligerito open) — none of which can be
   validated locally without (1) or (2).
