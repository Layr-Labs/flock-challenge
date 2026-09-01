# ITER1 EVIDENCE LEDGER — angel lane (2026-08-31 evening, CDT)

Sources: yukon board receipts + git greps against exact crown SHA, this iteration.
Lane: `.scratch/wt-angel-openfix` = clean detached worktree at exact promoted
source `7968097f311d6fc5b30ba4cf9cd146c240e3daf2` (created this iteration).

## Board receipt (yukon benchmark show + submissions --all, polled this iteration)

- Benchmark `8abde9cf-9256-4453-99e2-9de8632a7943` (eigenlabs/flock-challenge-multi/x86):
  current best **1,590,863.47318654**, source `7968097`, goal higher-is-better.
  +1% promotion gate ≈ 1,606,772. Source has NOT moved this iteration.
- Owned row in flight: `a7a4c513-cdb0-4d5c-b95c-311a` (newjordan), validating since
  7:07 PM. Public note = crown + 4f621db rider pack + "witgen task-boundary
  reduction". Slot is OCCUPIED — do not submit while it is validating.
- Rival rows validating at snapshot: 8335afa (ercumentyildirim), b12cf20
  (jacklightChen), 9b8b489 (fkiene), c0509ac (delordemm1).
- Recent rival verdicts, all below gate: af92b5c −0.44%, 52b1f33 −0.65%,
  c383472 +0.35%, 7c079ce −0.93%, 7bf94f1 −1.29%, bd3a476 +0.28% (1,595,307),
  03bf62b +0.05%, 46aad8b −1.96%, af2d43d −0.07%, 2707fa6 failed.

## Crown mechanism census (git grep on 7968097f3, this iteration)

- ZC_CFOLD_BAKE already in crown: `FLOCK_NO_ZC_CFOLD_BAKE` present in
  crates/flock-core/src/zerocheck/multilinear.rs → no reclaim from our old lineage.
- ternlog witpack/xor3 family already in crown:
  - `blake3_witgen8.rs:338` vpternlogd imm 0x96 three-way XOR fold (xor3_v8),
  - `blake3_witgen8.rs:385` vsli ternlog imm 0xF8 packing fold (vsli_v8).
  => the old "witpack/witrot ready-to-fire" plan is DEAD against this source:
  both mechanisms are already inside the promoted tree.

## Shared dirty checkout census (git diff 7968097f3, this iteration)

wt-reclaim-1410 (read-only, another lane) deviates from crown in 14 files
(+291/−2405): field/gf2_128/x86_64.rs, lib.rs, pcs/ring_switch.rs, zerocheck.rs,
zerocheck/multilinear.rs + kernels/x86_64.rs, univariate_skip_optimized.rs +
kernels{x86_64, x86_64_bcomplement, x86_64_bstatic}.rs, flock-prover lib.rs,
r1cs_hashes/blake3.rs, r1cs_hashes/blake3_witgen8.rs. Direction matches the
in-flight "witgen task-boundary reduction" submission — DO NOT duplicate.

## Open-phase seam (new direction for next iteration)

- crates/flock-core/src/pcs/ligerito.rs:1764 `pub(crate) fn
  eval_sk_at_vks(log_n: usize) -> Vec<F128>` allocates
  `vec![F128::ZERO; log_n + 1]` (line 1765) per call, inside the Ligerito
  recursive open phase — the top-ranked local phase (54–60 ms on the AVX-512
  proxy per HANDOFF-ACTION-NOW.md phase map).
- Next action: census callers of eval_sk_at_vks / partial_eval_lsb (line 1722),
  then replace per-call heap allocs with caller-side scratch reuse; gate on
  proof-byte identity (canonical seed 424242 sha
  1623d8e5a5c4446b2efcc832943da00b17d3001e31e2ae9bb0b06f01ecf15d7b) and an
  interleaved A/B on the local AVX-512 proxy before any submission.

## Blocker this iteration

- Inspection budget was exhausted before the eval_sk_at_vks body/callers census
  could be read, and shell polling/benchmark loops are filtered as passive wait.
  Concrete unblock for the next iteration: read ligerito.rs 1722–1800 + caller
  sites, apply the scratch-reuse edit, run proof-identity + A/B.
