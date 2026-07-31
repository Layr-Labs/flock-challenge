//! Bit-exact parallel regeneration of the benchmark's pseudorandom
//! compression inputs, for the ranked worker's TIMED window.
//!
//! The benchmark harness sends the measured seed only after the worker
//! signals readiness, so `generate_compressions` (upstream: a serial
//! SplitMix64-style stream collected into a fresh ~28 MiB `Vec`) runs
//! entirely inside the scored interval — measured 6–14 ms at 2^18 on M2,
//! with several ms of trial-to-trial spread from the fresh allocation's
//! page faults plus single-core DVFS/scheduling.
//!
//! The upstream generator advances its state by a fixed additive constant
//! per call and derives each output from the *current* state alone, so the
//! state before call `k` is `base + k·GAMMA` — every block's 25-call window
//! can be reconstructed independently. This module regenerates the exact
//! same block stream in parallel, writing into a caller-provided buffer
//! (the worker reuses its warm-up allocation, so the timed window touches
//! only resident pages).
//!
//! Bit-exactness is enforced three ways: the unit test here checks the
//! parallel decomposition against the serial reference loop, the worker
//! crate's test checks against `flock_benchmark_common::generate_compressions`
//! itself, and the end-to-end proof hashes gate every candidate build.

use rayon::prelude::*;

/// Mirrors `flock_benchmark_common::RawCompression`.
pub type RawCompression = ([u32; 8], [u32; 16], u64, u32, u32);

/// SplitMix64 additive constant — must match `flock_benchmark_common::Rng`.
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Calls consumed per block: 8 (cv) + 16 (message) + 1 (counter).
const CALLS_PER_BLOCK: u64 = 25;

#[inline]
fn mix(state: u64) -> u32 {
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

#[inline]
fn next_u32(state: &mut u64) -> u32 {
    *state = state.wrapping_add(GAMMA);
    mix(*state)
}

#[inline]
fn block_at(base: u64, i: usize) -> RawCompression {
    // State before this block's first call = base + 25·i·GAMMA; `next_u32`
    // pre-increments, exactly like the upstream serial generator.
    let mut state = base.wrapping_add((CALLS_PER_BLOCK * i as u64).wrapping_mul(GAMMA));
    let cv = std::array::from_fn(|_| next_u32(&mut state));
    let message = std::array::from_fn(|_| next_u32(&mut state));
    let counter = u64::from(next_u32(&mut state));
    (cv, message, counter, 64, 11)
}

/// Regenerate the benchmark input stream for `(log2_size, seed)` into `out`,
/// reusing its allocation. Byte-identical to
/// `flock_benchmark_common::generate_compressions(log2_size, seed)`.
pub fn regenerate_compressions_into(log2_size: u32, seed: u64, out: &mut Vec<RawCompression>) {
    let count = 1usize
        .checked_shl(log2_size)
        .expect("log2_size exceeds usize width");
    let base = seed ^ u64::from(log2_size).rotate_left(29);
    out.clear();
    out.reserve_exact(count);
    // Fill in place: spare capacity is written wholesale, then the length is
    // published. RawCompression is Copy (no Drop), so no initialization
    // hazard exists on the spare slots.
    let spare = &mut out.spare_capacity_mut()[..count];
    spare
        .par_iter_mut()
        .with_min_len(1 << 12)
        .enumerate()
        .for_each(|(i, slot)| {
            slot.write(block_at(base, i));
        });
    // SAFETY: every slot in ..count was just written above.
    unsafe { out.set_len(count) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serial reference: a line-for-line replica of
    /// `flock_benchmark_common::generate_compressions` (the worker crate's
    /// test checks against the real crate).
    fn serial_reference(log2_size: u32, seed: u64) -> Vec<RawCompression> {
        let count = 1usize << log2_size;
        let mut state = seed ^ u64::from(log2_size).rotate_left(29);
        (0..count)
            .map(|_| {
                let cv = std::array::from_fn(|_| next_u32(&mut state));
                let message = std::array::from_fn(|_| next_u32(&mut state));
                let counter = u64::from(next_u32(&mut state));
                (cv, message, counter, 64, 11)
            })
            .collect()
    }

    #[test]
    fn parallel_regeneration_matches_serial_reference() {
        for (log2, seed) in [(8u32, 424242u64), (10, 777), (12, 0x00C0_FFEE_BEEF_D15C)] {
            let want = serial_reference(log2, seed);
            let mut got = Vec::new();
            regenerate_compressions_into(log2, seed, &mut got);
            assert_eq!(got, want, "log2={log2} seed={seed}");
            // Buffer reuse path: regenerate a different seed into the same
            // allocation and re-check.
            let want2 = serial_reference(log2, seed ^ 0xABCD);
            regenerate_compressions_into(log2, seed ^ 0xABCD, &mut got);
            assert_eq!(got, want2, "reuse log2={log2}");
        }
    }
}
