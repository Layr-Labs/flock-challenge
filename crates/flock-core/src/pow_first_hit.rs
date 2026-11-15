//! Unordered proof-of-work search. Only a qualifying nonce can stop a drain.

use super::{HashKind, pow_scan};
use rayon::prelude::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_POW_FIRST_HIT").as_deref()
            != Some(std::ffi::OsStr::new("1"))
    })
}

pub(super) struct Winner {
    nonce: OnceLock<u64>,
    stopped: AtomicBool,
}

impl Winner {
    fn new() -> Self {
        Self { nonce: OnceLock::new(), stopped: AtomicBool::new(false) }
    }

    pub(super) fn offer(&self, nonce: u64) {
        // Publish the payload before cancellation. No nonce is a sentinel.
        if self.nonce.set(nonce).is_ok() {
            self.stopped.store(true, Ordering::Release);
        }
    }

    pub(super) fn get(&self) -> Option<u64> {
        self.nonce.get().copied()
    }
}

fn scan(
    digest: &[u8; 32],
    start: u64,
    len: u64,
    bits: u32,
    kind: HashKind,
    stop: &AtomicBool,
) -> Option<u64> {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if kind == HashKind::Blake3 && (1..=32).contains(&bits) && !super::grind_reg_disabled() {
        return crate::merkle::blake3_pow_scan_reg_until(digest, start, len, bits, stop);
    }
    // Preserve full SIMD batches in the portable kernel. SHA-256 also uses
    // this bounded cancellation interval, without changing its predicate.
    let mut offset = 0;
    while offset < len && !stop.load(Ordering::Relaxed) {
        let count = (super::BLAKE3_POW_BATCH as u64).min(len - offset);
        if let Some(nonce) = pow_scan(digest, start + offset, count, bits, kind) {
            return Some(nonce);
        }
        offset += count;
    }
    None
}

fn search(digest: &[u8; 32], bits: u32, kind: HashKind, winner: &Winner) -> u64 {
    const CHUNK: u64 = 960;
    let block = 1u64 << (bits.min(24) + 1);
    let n_chunks = block.div_ceil(CHUNK) as usize;
    let main_threads = rayon::current_num_threads();
    let mut start = 0u64;
    loop {
        if let Some(nonce) = winner.get() {
            return nonce;
        }
        let next = AtomicUsize::new(0);
        let worker = || {
            while !winner.stopped.load(Ordering::Relaxed) {
                let chunk = next.fetch_add(1, Ordering::Relaxed);
                if chunk >= n_chunks {
                    break;
                }
                let offset = chunk as u64 * CHUNK;
                let len = CHUNK.min(block - offset);
                if let Some(nonce) = scan(
                    digest, start.saturating_add(offset), len, bits, kind, &winner.stopped,
                ) {
                    winner.offer(nonce);
                    break;
                }
            }
        };
        match crate::epool::epool().filter(|_| main_threads > 1 && n_chunks >= 16) {
            Some(ep) => crate::epool::drain_hetero(
                main_threads, &worker, ep, &worker,
                super::grind_relay_enabled() && crate::epool::detached_enabled(),
            ),
            None => (0..main_threads).into_par_iter().with_max_len(1).for_each(|_| worker()),
        }
        if let Some(nonce) = winner.get() {
            return nonce;
        }
        start = start.saturating_add(block);
    }
}

pub(super) fn cpu_nonce(digest: &[u8; 32], bits: u32, kind: HashKind) -> u64 {
    search(digest, bits, kind, &Winner::new())
}

pub(super) fn gpu_nonce(digest: &[u8; 32], bits: u32) -> Result<u64, String> {
    let wide = super::grind_wide_enabled();
    let extra = if wide { super::GPU_GRIND_WIDE_EXTRA_BITS } else { 0 };
    let block_len = 1u32 << (bits + extra).min(24);
    let race = if wide { super::grind_race_enabled() } else { super::grind_hybrid_enabled() };
    if race && rayon::current_num_threads() > 1 {
        let winner = std::sync::Arc::new(Winner::new());
        if let Some(done) = super::grind_worker::dispatch_first_hit(
            digest, 0, block_len, bits, std::sync::Arc::clone(&winner),
        ) {
            let nonce = search(digest, bits, HashKind::Blake3, &winner);
            // The worker owns this job's winner. A losing in-flight command
            // may retire later without accessing the next gate's state.
            drop(done);
            return Ok(nonce);
        }
    }
    let mut start = 0u64;
    loop {
        if let Some(nonce) = crate::gpu_commit::gpu_blake3_pow_scan_first(
            digest, start, block_len, bits,
        )? {
            return Ok(nonce);
        }
        start = start.checked_add(u64::from(block_len))
            .ok_or_else(|| "GPU grind nonce range exhausted".to_string())?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_offer_owns_payload_including_max_nonce() {
        let winner = Winner::new();
        winner.offer(u64::MAX);
        winner.offer(0);
        assert_eq!(winner.get(), Some(u64::MAX));
        assert!(winner.stopped.load(Ordering::Acquire));
    }

    #[test]
    fn stopped_scan_does_no_work() {
        let stop = AtomicBool::new(true);
        for kind in [HashKind::Blake3, HashKind::Sha256] {
            assert_eq!(scan(&[0; 32], 0, 960, 1, kind, &stop), None);
        }
    }

    #[test]
    fn unstopped_scan_matches_ordered_kernel_within_its_chunk() {
        let stop = AtomicBool::new(false);
        for kind in [HashKind::Blake3, HashKind::Sha256] {
            for bits in [1, 7, 13] {
                for start in [0, u32::MAX as u64 - 30] {
                    assert_eq!(
                        scan(&[7; 32], start, 960, bits, kind, &stop),
                        pow_scan(&[7; 32], start, 960, bits, kind),
                    );
                }
            }
        }
    }
}

