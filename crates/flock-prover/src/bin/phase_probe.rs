//! Phase-timing probe: prove the ranked BLAKE3 shape TWICE in one process
//! and report both walls. The first prove is the warmup (latch decisions,
//! OnceLock kernel builds, pool population); the SECOND prove is the shape
//! the scored trials actually run. `FLOCK_PHASE_TIMING` / `FLOCK_COMMIT_TIMING`
//! are set here so per-phase lines print for both proves.
//!
//! Usage: phase_probe [steps_log2 (default 18)] [seed_hex (default 309)]

use flock_prover::challenger::{Challenger, FsChallenger};
use flock_prover::r1cs_hashes::blake3::{Blake3Setup, blake3_compress};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_block(&mut self) -> [u32; 16] {
        std::array::from_fn(|_| self.next_u64() as u32)
    }
}

const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
    0x5BE0CD19,
];

fn main() {
    // SAFETY: single-threaded at this point; set before any prover work.
    unsafe {
        std::env::set_var("FLOCK_PHASE_TIMING", "1");
        std::env::set_var("FLOCK_COMMIT_TIMING", "1");
    }
    let args: Vec<String> = std::env::args().collect();
    let log2: u32 = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(18);
    let seed = u64::from_str_radix(args.get(2).map(String::as_str).unwrap_or("309"), 16).unwrap();
    let steps = 1usize << log2;

    let mut rng = Rng(seed);
    let mut cv = BLAKE3_IV;
    let mut blocks = Vec::with_capacity(steps);
    for _ in 0..steps {
        let m = rng.next_block();
        blocks.push((cv, m, 0u64, 64u32, 0u32));
        let st = blake3_compress(&cv, &m, 0, 64, 0);
        cv = st[0..8].try_into().unwrap();
    }

    let setup = Blake3Setup::with_profile(steps, Default::default());
    for run in 0..2 {
        let mut ch = FsChallenger::new(b"flock_chain-cli");
        let t = Instant::now();
        let (_proof, _commitment) = setup.prove_chain(&blocks, &mut ch);
        // Transcript fingerprint: every prover message passed through the
        // challenger, so any proof-byte difference changes this sample.
        let fp = ch.sample_f128();
        eprintln!(
            "=== prove {} ({}) wall: {:.2} ms fs-fingerprint {:016x}{:016x}",
            run + 1,
            if run == 0 { "warmup" } else { "timed-shape" },
            t.elapsed().as_secs_f64() * 1e3,
            fp.hi,
            fp.lo,
        );
    }
}
