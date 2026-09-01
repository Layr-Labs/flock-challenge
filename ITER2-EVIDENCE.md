# ITER2 EVIDENCE LEDGER — angel lane (build-toolchain seam)

Direction: the promoted crown is validated on the board but its source does
not build with the repo-pinned toolchain as installed on this box — the
submittable-source/build seam, not a runtime kernel.

## A. Crown does not compile on pinned 1.97.0 (this box's rustup install)

`cargo +1.97.0 build --locked --offline --profile challenge -p flock-benchmark-worker`
against clean crown 7968097f3 (lane `.scratch/wt-angel-openfix`, driver build):

    error[E0658]: use of unstable library feature `isolate_most_least_significant_one`
      --> crates/flock-core/src/ntt/inv_table.rs:99:28
              let lo_bit = w.isolate_lowest_one();
      --> crates/flock-core/src/ntt/inv_table_deg4.rs:110:28
      --> crates/flock-core/src/zerocheck/multilinear.rs:684:32
      --> crates/flock-prover/src/chain.rs:399:25
    (flock-core fails first; flock-prover/chain.rs surfaced on rebuild after
     the three flock-core sites were fixed)

rustc: 1.97.0 (2d8144b78 2026-07-07), rustup 1.97.0 + 1.97.1 + stable all
resolve to the same build on this box; std sysroot metadata says compiler
built 2026-04-14. `RUSTC_BOOTSTRAP=1` is NOT propagated by cargo through to
rustc, but a rustc-wrapper that sets it compiles the unstable call (t.rs test
prints 4) — so the gate is bootstrap-only, i.e. this box's 1.97.0 std predates
the stabilization.

## B. Fix applied in lane (4 files, 4 lines, semantics-preserving)

`x.isolate_lowest_one()` -> `x & x.wrapping_neg()` (the documented std
definition of isolate_lowest_one; identical for every input incl. 0):

- crates/flock-core/src/ntt/inv_table.rs:99
- crates/flock-core/src/ntt/inv_table_deg4.rs:110
- crates/flock-core/src/zerocheck/multilinear.rs:684
- crates/flock-prover/src/chain.rs:399

Lane now builds clean on pinned 1.97.0: `Finished challenge profile in 23.07s`.

## C. Verified correctness at both geometries

- smoke (log2=12, trusted verifier, witness re-derived): both measured trials
  `verified=true`, score=34534.568 cps.
- prove 424242 18 16: 2230 ms, proof_sha=9c93b71f281ea355, 437519 bytes.

NOTE: sha differs from the 1623d8e5 "canonical" recorded in ITER1 — that value
was captured from the main-repo (shared dirty) worker, which carries the
in-flight witgen pack; attribution of the delta (dirty-tree provenance vs
edit effect) is unproven this iteration.

## D. Size-dependent failure map (lane build = crown + 4-line edit)

Trusted-verifier bench + direct worker proves, this box (Zen 3, no AVX-512,
FLOCK_NO_WITGEN_TERNLOG=1 shim), threads=16:

- log2=12  smoke: verified=true on both measured trials.
- log2=16  bench: warmup + 2/2 measured trials verified=true,
           score=59268.094 cps (score.json in benchmark-results/run-skill/).
- log2=17  WORKER HANGS: bench hangs before/after warmup line (twice), and
           direct prove seed 424242 also produces no proof in 32s+.
           Harness RUN_TIMEOUT is 900s (benchmark-tools/harness/src/main.rs:22),
           so the verifier itself never aborts it.
- log2=18  worker completes fast (prove 424242: 2230 ms, 437519 bytes,
           sha 9c93b71f281ea355) BUT two separate bench runs (fresh
           /dev/urandom seed per trial, harness/src/main.rs:354-358) both
           failed: "trusted verifier rejected proof: Zerocheck(SumcheckFinalFailed)".

Pattern: hang at exactly 2^17, invalid proofs at 2^18, clean at <= 2^16.
Seed-independent hang (424242 hangs too); proof failures seen on >= 2 random
seeds. The 4-line edit cannot cause this: x & x.wrapping_neg() is bit-identical
to isolate_lowest_one for every input, and the same edited code verifies at
12/16 — the failures are size-gated, the edit is size-independent.

## E. Consequences for the loop (hypotheses)

- On this box, the trusted-verifier bench is unusable at log2 >= 17 for
  crown-lineage builds; preflights must be log2 <= 16 verified or fixed-seed
  unverified proves at 18.
- The bug lives in crown's non-AVX-512 fallback paths (the ranked runner's
  AVX-512 build scored 1.59M over 100 runs, so its kernels are clean).
  Fixing the local fallback earns no board points by itself, but any candidate
  touching the shared code (ligerito open / NTT deep / zerocheck) must keep
  verified-clean at 12/16 and cannot be validated locally at 17/18.
- /tmp/wt-crown (clean crown worktree) reference build via RUSTC_WRAPPER
  failed: cargo does not propagate RUSTC_BOOTSTRAP and the wrapper log shows
  the gate still closed (see .scratch/wtcrown-build.log). Byte-identity of the
  4-line edit against unmodified crown remains unproven by build; it rests on
  the documented std definition of isolate_lowest_one.
