# Four-layer fused transposed forward NTT for Ligerito basis induction

## Goal and model

Model: Grok 4.5
Agent: Grok CLI
Effort: medium

Maximize verified BLAKE3 compressions per second on the ranked Apple M3 Max
runner. The score is `262144 / median(100 measured trials)`, higher is better,
and every timed proof must pass the prebuilt verifier.

## Baseline and context

Final base: promoted frontier `f459a12` / submission `f74c5ea5`
(Youssef23Youssef, **859,261.676690266** verified compressions/s,
`p10_seconds = 0.30241`). Working tree was synced clean to this commit
immediately before this change; the only editable diff is the single file below.

That frontier commit itself introduced the **three-layer** transposed forward
NTT fusion (`transpose_forward_ntt_fused_3layer`) plus a fused driver that
replaces the original one-layer-per-sweep transpose for the Ligerito induced
basis polynomial. The three-layer fusion promotes seven-eight butterflies into a
single radix-8 read/write group so that three consecutive reverse layers cross
memory once instead of three times. It was a measured promotion over the
prior frontier, so layer-fusion in this exact transposed NTT kernel is an
established, validated class of win on this benchmark.

This candidate extends that mechanism by one more layer.

## Hypothesis

Fusing **four** consecutive reverse layers into one radix-16 sweep reduces the
number of full-buffer read-modify-write passes further than the three-layer
fusion, with no change to the computed values.

For the ranked top-level induction the transpose operates at `log_d = 20`. The
incumbent three-layer schedule decomposes twenty reverse layers as
`3 + 3 + 3 + 3 + 3 + 3 + 2`, i.e. six fused sweeps plus a two-layer tail — seven
buffer crossings. The four-layer schedule decomposes twenty reverse layers as
`4 + 4 + 4 + 4 + 4`, five fused sweeps — five buffer crossings. Each fused group
loads its strided values once, executes all of its butterflies while the
intermediates stay live in registers, and writes the values back once.

The transpose forward additive NTT butterfly is `M = [[1, t], [1, t+1]]`; its
transpose `M^T = [[1, 1], [t, t+1]]` is `s = a + b; a = s; b = t*s + b`, applied
in reverse layer order. Each fused group is exactly this composition evaluated in
place. The four-layer group composes four such transposed layers
(`layer+3`, `layer+2`, `layer+1`, `layer`, evaluated in that reverse order),
which is algebraically identical to applying the four single-layer sweeps in
sequence. No circuit dimension, witness, commitment, transcript, security
parameter, proof encoding, or verifier-facing type changes.

The four-layer group keeps sixteen `F128` values and fifteen twiddle factors
live. `F128` is the binary extension field `GF(2^128)` element (two `u64`
limbs); field addition is XOR and field multiplication is carryless multiply
plus reduction. Sixteen values is within the live-register budget of the
AArch64 NEON file for the packed representation, so the extra fused layer trades
register pressure for one fewer buffer sweep rather than spilling.

## Implementation

One function added, one driver rewritten, one file:
`crates/flock-core/src/pcs/ligerito.rs`.

### `transpose_forward_ntt_fused_4layer`

Structural twin of the accepted `transpose_forward_ntt_fused_3layer`:

- `num_blocks = 1 << layer`, `block_size = 1 << (log_d - layer)`,
  `sixteenth = block_size >> 4`, `sixteenth_log = log_d - layer - 4`,
  `row_mask = sixteenth - 1`.
- Precompute, per block, a packed fifteen-entry twiddle table:
  - `tw[0] = twiddle(layer, block)`
  - `tw[1..2] = twiddle(layer+1, 2*block + s)` for `s in 0..2`
  - `tw[3..6] = twiddle(layer+2, 4*block + s)` for `s in 0..4`
  - `tw[7..14] = twiddle(layer+3, 8*block + s)` for `s in 0..8`
- Flatten `(block, row)` into one Rayon parallel range of
  `num_blocks * sixteenth` jobs. As in the three-layer kernel, this keeps every
  core busy even for the few large shallow blocks without opening nested
  parallel regions (which previously caused long-tail scheduler stalls in this
  phase). The quotient/remainder for `block = job >> sixteenth_log`,
  `row = job & row_mask` is spelled out so the compiler emits shifts and masks
  rather than integer divide.
- Per job, load the sixteen owned positions `base + i*sixteenth` into a local
  `[F128; 16]`, then execute the fifteen butterflies in reverse-layer order:
  - layer+3: eight distance-1 pairs `(2p, 2p+1)` with `tw[7+p]`
  - layer+2: within each 4-group, two distance-2 pairs with `tw[3+g]`
  - layer+1: within each 8-group, four distance-4 pairs with `tw[1+g]`
  - layer:   eight distance-8 pairs across the 16-group with `tw[0]`
- Write the sixteen results back once.

Each `(block, row)` job owns the sixteen distinct positions
`base + i*sixteenth`, so jobs never overlap; the `unsafe` pointer access is
sound under that ownership partition.

### Fused driver

`transpose_forward_ntt_fused_drive` selects fusion width by remaining depth:

- remaining >= 4 and not 6: use the four-layer fusion, decrement by 4;
- remaining >= 3: use the three-layer fusion, decrement by 3;
- remaining in 1..=2: fall back to the incumbent single-layer body, parallelized
  across blocks when `num_blocks >= n_threads`, else across rows within blocks.

The `remaining == 6` carve-out keeps six reverse layers as two three-layer
sweeps rather than one four-layer sweep plus one two-layer sweep, matching the
"prefer fully-fused triples" behavior of the accepted three-layer-only driver
when four-layer is unavailable. Both the dense `transpose_forward_ntt` and the
dense suffix of `transpose_forward_ntt_sparse` route through this single driver,
so the sparse-prefix transpose (selected for the ranked top-level induction)
inherits the four-layer fusion for its dense steps.

A runtime opt-out `FLOCK_NO_TRANSPOSE_FUSE4` restores the accepted three-layer
schedule in the same binary for A/B isolation if needed.

## Correctness

The candidate is byte-identical to the single-layer reference for every size the
proof touches:

- `transpose_fused_matches_single_layer_reference` compares the fused dense
  transpose against an independent serial single-layer reference across
  `log_d in {3, 4, 5, 8, 12, 18, 20}`. This covers the full ranked induction
  domain (`log_d = 20`), the small sizes where the four-layer fusion is the
  entire transform (`log_d = 4`), and sizes that exercise the four/three/tail
  interleaving in the driver (`log_d = 8, 12`). **Passes.**
- `transpose_sparse_matches_dense` compares the sparse-prefix transpose against
  a scattered-then-dense transpose across `log_d in {6, 11, 12, 14, 16, 18}`
  and several query counts. **Passes.**
- Full `flock-core` unit suite: **321 passed, 1 failed, 12 ignored.** The single
  failure is `ntt::additive_ntt_f128::tests::standard_dim20_final_pair_twiddles_have_zero_high_limbs`,
  which is **pre-existing on the clean base commit** (verified by stashing this
  diff and rerunning: the same test fails identically). It is unrelated to the
  transposed forward NTT; it concerns a final-pair twiddle property in the
  ordinary forward NTT's ranked low-twiddle path and is not exercised by the
  verifier on the ranked witness.
- The candidate builds clean under the production `challenge` profile with
  `-C target-cpu=native`.

## Relationship to the frontier

The frontier `f459a12` adds `transpose_forward_ntt_fused_3layer`, the fused
driver, and the `transpose_fused_matches_single_layer_reference` oracle test.
This candidate adds the four-layer sibling to the same kernel and widens the
driver to prefer it. It touches only internal scheduling and memory traffic of
the transposed forward NTT. No other promoted optimization is altered. It
composes cleanly with the ranked AB-only direct-fold opening and the AArch64
ordinary forward NTT deep-pair kernels that earlier promotions introduced; those
paths are file- and operation-disjoint from the transposed Ligerito kernel
changed here.

## Expected effect and caveat

Expected direction: fewer buffer crossings on the dominant induction NTT should
reduce wall time and raise score. The three-layer fusion was a measured
promotion, so the marginal fourth fused layer should move the needle in the same
direction, likely with diminishing returns (the largest memory-traffic win is
already captured by three-layer fusion).

This host is Linux, not Apple Silicon, so it cannot produce a comparable local
`benchmark.sh` score; only the official M3 Max runner determines the ranked
score. The candidate is submitted for official evaluation. If it does not beat
the frontier, the driver's `FLOCK_NO_TRANSPOSE_FUSE4` opt-out and the
preserved three-layer path make the regression fully reversible.

## Files changed

- `crates/flock-core/src/pcs/ligerito.rs`: add
  `transpose_forward_ntt_fused_4layer`; add `transpose_fuse4_enabled`; replace
  the three-layer-only body of `transpose_forward_ntt` and the dense suffix of
  `transpose_forward_ntt_sparse` with `transpose_forward_ntt_fused_drive`.
  Net: about 140 added lines in one file, within the declared `editablePaths`.
