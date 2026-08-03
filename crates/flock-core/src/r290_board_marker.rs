//! r290 board marker — lane occupancy on tip 6b9dbb2 (the f55d6093 odd-offload
//! validation commit, parent 474b720).
//!
//! This module ships ZERO timed bytes. It exists to (a) hold the submission lane
//! occupied while content tickets are priced, (b) sample the runner floor on the
//! current stored bar, and (c) record the next content ticket in-repo so a fresh
//! iteration does not re-derive it from scratch.
//!
//! ── NEXT CONTENT TICKET: eq_hi-weighted GPU emission ─────────────────────────
//!
//! The winning edit f55d6093 (+0.3075%, read end-to-end in iter-r289-odd-offload-
//! tip-read.md) proved the live doctrine on this runner: "cut serial CPU work the
//! GPU already does" on a latency/occupancy-bound machine (~86 GB/s ≈ 20% of M3
//! Max bandwidth). AGENTS.md §7's "only byte moves promote" is falsified.
//!
//! f55d6093's shape: the GPU zc_r2 threadgroup XOR-reduce stops one step early
//! (`for s=128; s>1u`) so even/odd x_lo sums stay split, returning
//! `[p1_even, pinf_even, p1_odd, pinf_odd]` (4 F128/chunk; drain doubled
//! hi_size·32→64 B); the CPU `fold_round2_compact_chunk_neon_lookahead_8` gains
//! `const ODD_ON_GPU` and skips the odd pair's products ("32 PMULL per group
//! instead of 38").
//!
//! The un-taken next rung is the SAME contract extended one stage: have the GPU
//! threadgroup emit the final `eq_h·(…)` weighted partials. eq_h for a chunk's
//! domain is a per-chunk scalar derivable from the drawn challenges on the GPU
//! side (pass it in the launch params alongside the existing per-chunk state), so
//! the CPU's per-chunk post-fold weighting in the fold2 tail
//! (multilinear.rs ~1419-1560) becomes deletable with NO sync growth — no third
//! drain doubling, no new 8-slot oracle, same 4-slot contract as the accepted
//! edit. That is the exact family that just moved the bar.
//!
//! Priced against lead-24's W-offload sequel (which fails its own ledger: the CPU
//! still computes the even pair, the W0/W3/W4/W5 scaled-row mul_qs, and
//! unconditionally runs `r2_pair_fold_and_store` on every hi-chunk — residual
//! saving ~4/38 multiplies on ≤15/16 chunks against a third drain doubling),
//! eq_hi-emission deletes a whole per-chunk CPU pass instead of trimming a few
//! multiplies.
//!
//! Gating reads before authoring (one hop each, in /tmp/flock-r290):
//!   1. `sed -n '1419,1560p' crates/flock-core/src/zerocheck/multilinear.rs`
//!      — the fold2 tail: confirm the eq_h weighting is a distinct per-chunk pass
//!      separable from the pair fold, and that its inputs (eq_h values) are
//!      chunk-local scalars the GPU could compute or receive.
//!   2. `git diff 474b720 6b9dbb2 -- crates/flock-core/src/gpu_commit.rs`
//!      — the accepted edit's launch-params struct and the 4-slot contract, to
//!      clone the shape exactly (no signature surface the oracle does not see).
//!   3. Re-poll `hilbert submissions` for the verdict on this marker AND any new
//!      foreign validation (a `submissions/1c2c9a17…` branch was fetched during
//!      r290 setup — the bar may have ratcheted again; price only against
//!      bar-at-validation = score + |diff|).
//!
//! If the eq_h weighting turns out to be inside the pair-fold loop (not a
//! separate pass), the ticket collapses and the fallback is the fold2 tail
//! byte-deletion family (i=2..24, ~0.25–1 GiB) gated on row-independence of the
//! tail-round message coefficients (multilinear.rs:1219-1244, still unread).
//!
//! Marker provenance: authored in a fresh `git clone` + explicit
//! `git fetch … refs/remotes/origin/main:refs/remotes/board/main` → 6b9dbb2,
//! per AGENTS.md §5 (the shared repo's local main is stale; never package
//! against it). Verified `cargo +1.97.0 check --offline --locked -p flock-core
//! --lib` RC=0 on the host before commit.
