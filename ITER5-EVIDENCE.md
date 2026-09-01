# ITER5 evidence — par-merge edit + ranked-shape A/B chain

## A. The ranked runner's actual round1 branch (read this iteration)
- zerocheck.rs:522-601: runner's round1 = `round1_shift_reduce_ab_packed_padded_with_precomputed`
  (AB half) || `round1_c_fold4_from_block_major_z` (identity-C half), joined via SMT pools
  (`zc_r1_pools`), then a POST-JOIN SERIAL merge adds `one_ab` into `ab` element-wise
  (zerocheck.rs:590-595 pre-edit).
- C half: `fold_block_major_one_shot_bind_top_ranked_one_rows` (lincheck.rs:986) does one
  block-major sweep producing BOTH fold8 banks and the one-row fold; then fold4/quad
  collapses (univariate_skip_optimized.rs:4049-4107); `one_ab` is lifted via
  `round1_lifted_from_fold8` (univariate_skip_optimized.rs:3926) + ntt_extend.
- AB half: `fill_invalid_prefix` (univariate_skip_optimized.rs:580) returns immediately when
  `invalid_prefix_bytes == 0` (ranked shape) — the "invalid-prefix recompute" seam is DEAD.
- Kill switches confirmed: FLOCK_NO_EXTRA_WARMUP (blake3.rs:4034) skips the 11-prove untimed
  warmup; FLOCK_ZC_TIMING (zerocheck.rs:508) prints the half-split line
  "round1 AB {ms} || identity-C fold {ms} -> {ms}". The FLOCK_ORACLE_NO_WARMUP env used in
  all prior SDE runs is BOGUS — not read anywhere in the source.

## B. Candidate mutation this iteration (lane .scratch/wt-angel-openfix)
- crates/flock-core/src/zerocheck.rs ~L590: post-join one_ab merge changed from serial
  `iter_mut().zip()` to `par_iter_mut().zip(par_iter())` — bit-identical element-wise adds,
  shorter critical-path tail. Diff: 12 insertions, 3 deletions.
- Portable preflight (trusted flock_benchmark_verifier, direct, no sandbox arg):
  2/2 measured trials verified=true @ log2=16 → edit is proof-canonical on the portable path.
  Score 43,932 cps is CONTENDED (SDE run co-resident) — not a perf verdict.

## C. Runner-faithful build fix (wrapper v3)
- Old .scratch/rustc-noavx512.sh is broken for the current lane: it `exec`s rustc 1.97.0 with
  cargo's argv[1] (the lane's resolved rustc path) as an input filename →
  "multiple input filenames provided".
- New .scratch/rustc-noavx512-v3.sh: shifts the rustc-path argv, gives workspace crates
  `-C target-cpu=native -C target-feature=+avx512f,+avx512vl,+avx512bw,+avx512dq,+avx512vbmi,
  +vpclmulqdq,+gfni,-sse4a`, passes everything else unchanged.
- Build: RUSTFLAGS="" RUSTC_WRAPPER=<v3> CARGO_TARGET_DIR=target/challenge-avx512
  cargo +1.97.0 build --locked --offline --profile challenge --target-dir target/challenge-avx512
  -p flock-benchmark-worker  (CARGO_INCREMENTAL=0 variant timed out at 30s — omit it).
- Baselines/binary shas:
  - .scratch/wt-worker-baseline-18 = dca86cbe157bbdc751afc0f4f3aaa5a8fd4011f88fa355645de33a5a8e5d06bd
    (pre-edit AVX-512 worker, verified clean at 12/16/17/18 in ITER4)
  - edited: target/challenge-avx512/challenge/flock-benchmark-worker =
    c23f9e03034718d016b21b365215a5daf237f81d53201a15d57466773bb7b8e2, 85,400 zmm refs, 0 insertq.
- NOTE: `--profile challenge` puts the binary in <target>/challenge/; plain --release builds
  go to <target>/release/ and are NOT the file the verifier/SDE should run.

## D. In flight (chain, launched this iteration)
nohup chain: baseline run (sde18-split.log / sde18-split-exit.txt) THEN edited run
(sde18-edit.log / sde18-edit-exit.txt), each: sde64 -icx, worker 18, seed 424242,
FLOCK_NO_EXTRA_WARMUP=1 FLOCK_ZC_TIMING=1 RAYON_NUM_THREADS=8, timeout 1700.
~2 proves (call-0 warmup + measured) per run ≈ 10-15 min each, ~25 min total, unattended.

## E. Next
1. When both exit files exist: extract the SECOND [zc-timing] block of each log (= measured
   prove) and A/B "round1 AB x || identity-C fold y -> z" + total prove time.
2. Verify /tmp/s18baseproof and /tmp/s18editproof sha1 identical (edit is canonical on the
   runner path) and equal to the ITER4 reference e66b4c7c881e1218a05f89b5afee94ceab0def0f.
3. If the merge shows in z: keep the edit, look next at round2 fused fold / rounds3+ tail.
