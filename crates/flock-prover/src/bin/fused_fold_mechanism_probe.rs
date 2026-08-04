//! Fused transpose+fold mechanism probe.
//!
//! Proves the FUSION MECHANISM in isolation, at mock geometry: when a fold
//! over bit-stripes is per-stripe independent (accumulate f(stripe) into a
//! k-entry accumulator), folding each stripe *inside* the transpose task into
//! a thread-local accumulator and combining the thread-locals afterwards is
//! BIT-EXACT against the two-pass pipeline (materialize all stripes, then
//! fold). The real fold (`partial_fold_packed_z_fast_padded`,
//! crates/flock-core/src/lincheck.rs:671) is what the fusion would replace in
//! the live prover; its per-stripe independence is the code-audit prerequisite
//! this probe cannot establish by itself (documented blocker).
//!
//! This is a mechanism proof + byte-ledger reprint, NOT a timing: this host is
//! x86-64 and cannot predict the Apple M-series score (AGENTS.md §1). The byte
//! ledger (481.97 MiB transpose write vs 481.75 MiB fold re-read at m=32,
//! ~465.75 MiB ≈ 3.03% of the ~15 GiB timed ledger removable) is reproduced
//! from the constants measured by `pitch_shrink_probe`.

/// Toy per-stripe fold used only to prove the fusion mechanism: maps a stripe
/// (u64 words) to a k-entry u64 accumulator via an order-independent hash.
fn toy_fold_stripe(stripe: &[u64], k: usize) -> Vec<u64> {
    let mut acc = vec![0u64; k];
    for (i, &w) in stripe.iter().enumerate() {
        // splitmix64-style mixing; per-stripe independent by construction
        let mut x = w ^ 0x9e37_79b9_7f4a_7c15u64;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        acc[i % k] ^= x;
    }
    acc
}

/// Two-pass reference: materialize every stripe, then fold.
fn two_pass(stripes: &[Vec<u64>], k: usize) -> Vec<u64> {
    let mut acc = vec![0u64; k];
    for s in stripes {
        let a = toy_fold_stripe(s, k);
        for (i, v) in a.iter().enumerate() {
            acc[i] ^= v;
        }
    }
    acc
}

/// Fused: each "transpose task" (here simulated as a sequential slice of
/// stripes, standing in for a worker thread) folds its stripes into a local
/// accumulator; a combine pass merges the thread-locals.
fn fused(stripes: &[Vec<u64>], k: usize, threads: usize) -> Vec<u64> {
    let mut locals = vec![vec![0u64; k]; threads];
    for (idx, s) in stripes.iter().enumerate() {
        let t = idx % threads;
        let a = toy_fold_stripe(s, k);
        for (i, v) in a.iter().enumerate() {
            locals[t][i] ^= v;
        }
    }
    let mut acc = vec![0u64; k];
    for l in &locals {
        for (i, v) in l.iter().enumerate() {
            acc[i] ^= v;
        }
    }
    acc
}

fn main() {
    // Byte ledger at ranked m=32 geometry (constants from pitch_shrink_probe).
    let stripes: u64 = 32768;
    let write_mib: f64 = 481.97;
    let read_mib: f64 = 481.75;
    let mib: f64 = (1u64 << 20) as f64;
    let write_bytes = write_mib * mib;
    let read_bytes = read_mib * mib;
    let k: u64 = 32;
    let combine_bytes = stripes * k * 16; // k F128 per stripe, combine pass
    let current = write_bytes + read_bytes;
    let fused_bytes = write_bytes + combine_bytes as f64;
    let saved = read_bytes - combine_bytes as f64;
    let ledger_gib = 15.0 * (1u64 << 30) as f64;
    println!("FUSION MECHANISM PROBE (x86 host — NOT a timing)");
    println!("current_double_touch={:.2} MiB, fused={:.2} MiB, saved={:.2} MiB = {:.3}% of ledger",
        current / mib, fused_bytes / mib, saved / mib, saved / ledger_gib * 100.0);
    println!("run `cargo test --bin fused_fold_mechanism_probe` for the bit-exactness check");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_stripes(rng_seed: u64, n: usize, words: usize) -> Vec<Vec<u64>> {
        let mut x = rng_seed;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..n)
            .map(|_| (0..words).map(|_| next()).collect())
            .collect()
    }

    #[test]
    fn fused_matches_two_pass_across_geometries() {
        for &(n, words, k, threads) in &[
            (1usize, 1usize, 4usize, 1usize),
            (2, 3, 4, 2),
            (7, 5, 8, 3),
            (64, 16, 32, 8),
            (127, 33, 32, 4),
        ] {
            for seed in 1..5u64 {
                let stripes = random_stripes(seed, n, words);
                assert_eq!(
                    two_pass(&stripes, k),
                    fused(&stripes, k, threads),
                    "mismatch n={n} words={words} k={k} threads={threads} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn empty_stripes_are_identity() {
        let stripes: Vec<Vec<u64>> = vec![];
        assert_eq!(two_pass(&stripes, 8), vec![0u64; 8]);
        assert_eq!(fused(&stripes, 8, 2), vec![0u64; 8]);
    }
}
