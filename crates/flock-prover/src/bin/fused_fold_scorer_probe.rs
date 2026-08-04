//! fused_fold_scorer_probe — ranked-geometry bit-exactness gate for the
//! in-transpose fold fusion candidate.
//!
//! The live fold (`partial_fold_packed_z_fast_padded`,
//! crates/flock-core/src/lincheck.rs:671) is per-stripe independent with known
//! arithmetic: each worker owns contiguous stripes, folds them into a private
//! length-`k` accumulator via `acc[i_inner] += table[z_byte]` (table = 256-entry
//! subset-sum over the stripe's 8 `eq_outer` entries), and the workers'
//! accumulators are XOR/plus-reduced afterwards (lincheck.rs:695-721). That
//! makes the fold a commutative monoid over stripes, so folding each stripe
//! *inside* the transpose task into a thread-local accumulator and combining
//! afterwards is bit-exact by construction.
//!
//! This probe turns that construction argument into an empirical one at the
//! EXACT scored geometry: it implements the fused kernel independently from the
//! disclosed arithmetic, then asserts bit-equality against the REAL
//! `partial_fold_packed_z_fast_padded` on shared random inputs at
//! (m=32, k_log=14, useful=USEFUL_BITS=15409) — the ranked Blake3
//! configuration — plus the dense case (useful=k) and a geometry sweep.
//!
//! x86-host run only; no timing claims per AGENTS.md §1 (the scored run is
//! Apple silicon; this probe's only job is the byte-ledger + equality gate).

use flock_prover::field::F128;
use flock_prover::lincheck::partial_fold_packed_z_fast_padded;
use flock_prover::r1cs_hashes::blake3::USEFUL_BITS;

/// Minimal xorshift64 — deterministic, no external deps (offline locked build).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// 256-entry subset-sum table over 8 F128 values, mirroring the live fold's
/// `build_sum_table` (lincheck.rs:803): table[b] = Σ_{r: bit r of b set} eq8[r].
/// Written independently from the disclosed recurrence (bottom-up, ctz split).
fn build_sum_table(eq8: &[F128], table: &mut [F128]) {
    table[0] = F128::ZERO;
    for b in 1usize..256 {
        let low = b & (b - 1);
        let r = b.trailing_zeros() as usize;
        table[b] = table[low] + eq8[r];
    }
}

/// The fused kernel: single accumulator, stripe-sequential, folding only
/// `stripe[..useful_bits]` — structurally identical arithmetic to the live
/// fold's per-worker inner loop (lincheck.rs:703-709), minus the worker split
/// (which only reorders the commutative reduce).
fn fused_fold(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let k = 1usize << k_log;
    let n_outer = 1usize << (m - k_log);
    let n_stripes = n_outer / 8;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert!(useful_bits <= k);
    let mut acc = vec![F128::ZERO; k];
    for b in 0..n_stripes {
        let mut table = vec![F128::ZERO; 256];
        build_sum_table(&eq_outer[8 * b..8 * b + 8], &mut table);
        let stripe = &z_packed[b * k..(b + 1) * k];
        for (i, &z_byte) in stripe[..useful_bits].iter().enumerate() {
            acc[i] += table[z_byte as usize];
        }
    }
    acc
}

/// Fill `z_packed` with deterministic pseudo-random bytes (any bit pattern is a
/// valid F128/binary-field witness; the lib's own cast at common.rs:410 treats
/// the data as raw u64 pairs).
fn fill_bytes(buf: &mut [u8], rng: &mut Rng) {
    for w in buf.chunks_exact_mut(8) {
        w.copy_from_slice(&rng.next().to_le_bytes());
    }
}

fn make_eq_outer(n_outer: usize, rng: &mut Rng) -> Vec<F128> {
    // F128 is repr(C) with exactly two u64s (common.rs:408-409), so a bit
    // transmute of a random u64 pair is a valid field element in GF(2^128).
    (0..n_outer)
        .map(|_| unsafe { std::mem::transmute::<[u64; 2], F128>([rng.next(), rng.next()]) })
        .collect()
}

/// One (m, k_log, useful) equality gate against the REAL live fold.
fn assert_fused_equals_real(
    m: usize,
    k_log: usize,
    useful_bits: usize,
    seeds: &[u64],
    label: &str,
) {
    let n_outer = 1usize << (m - k_log);
    for &seed in seeds {
        let mut rng = Rng(seed);
        let mut z_packed = vec![0u8; (1usize << m) / 8];
        fill_bytes(&mut z_packed, &mut rng);
        let eq_outer = make_eq_outer(n_outer, &mut rng);
        let real = partial_fold_packed_z_fast_padded(&z_packed, m, k_log, useful_bits, &eq_outer);
        let fused = fused_fold(&z_packed, m, k_log, useful_bits, &eq_outer);
        assert_eq!(
            fused, real,
            "{label}: fused != real at m={m}, k_log={k_log}, useful={useful_bits}, seed={seed}"
        );
    }
}

/// Ranked-geometry byte ledger for the fusion candidate (transpose-side write
/// volume vs fold-side re-read volume, identical stripe order — see
/// fusion_ledger_probe). `pitch64` = useful_words·64 (transpose writes only
/// full 64-byte words, common.rs:446-458), `pitch8` = ceil(useful/8)·8 (fold
/// reads byte-wise to useful_bits, lincheck.rs:706).
fn print_ledger(m: usize, k_log: usize, useful_bits: usize) {
    let n_stripes = 1usize << (m - k_log - 3);
    let pitch64 = useful_bits.div_ceil(64) * 64;
    let pitch8 = useful_bits.div_ceil(8) * 8;
    let write_bytes = n_stripes * pitch64;
    let read_bytes = n_stripes * pitch8;
    let double_touch = write_bytes + read_bytes;
    let removed = read_bytes; // fused keeps only the write-side touch
    eprintln!(
        "ledger m={m} k_log={k_log} useful={useful_bits}: stripes={n_stripes} \
         transpose_write={write_bytes} fold_read={read_bytes} \
         current_double_touch={double_touch} bytes_removed={removed}"
    );
}

fn main() {
    // Ranked Blake3 configuration (m=32, k=2^14, useful=15409).
    print_ledger(32, 14, USEFUL_BITS);
    // Geometry sweep ledger for context.
    for (m, kl) in [(20usize, 14usize), (24, 14), (26, 14), (28, 14)] {
        print_ledger(m, kl, USEFUL_BITS.min(1usize << kl));
    }
    assert_fused_equals_real(32, 14, USEFUL_BITS, &[0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F], "ranked");
    eprintln!("fused_fold_scorer_probe: ranked-geometry gates PASSED");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_matches_real_at_ranked_geometry() {
        assert_fused_equals_real(32, 14, USEFUL_BITS, &[0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F], "ranked");
    }

    #[test]
    fn fused_matches_real_dense_geometry() {
        // dense useful = k exercises the full stripe; ranked USEFUL_BITS at a
        // smaller m where n_log keeps the tiled dispatch path in play.
        assert_fused_equals_real(24, 14, 1 << 14, &[0xDEADBEEF, 0xFEEDFACE], "dense");
        assert_fused_equals_real(24, 14, USEFUL_BITS, &[0x0123456789ABCDEF], "ranked-useful@24");
    }

    #[test]
    fn fused_matches_real_small_sweep() {
        for (m, kl) in [(16usize, 11usize), (18, 11), (20, 14)] {
            assert_fused_equals_real(m, kl, (1usize << kl) - 3, &[0x11111111, 0x22222222], "sweep");
        }
    }

    #[test]
    fn ledger_ranked_sanity() {
        // Regression anchor: ranked ledger must match the independently
        // measured numbers (fusion_ledger_probe: stripes=32768,
        // fold_read=505151488, transpose_write≈505382175).
        let n_stripes = 1usize << (32 - 14 - 3);
        assert_eq!(n_stripes, 32768);
        assert_eq!(USEFUL_BITS, 15409);
        let pitch8 = USEFUL_BITS.div_ceil(8) * 8;
        assert_eq!(n_stripes * pitch8, 505151488);
    }
}
