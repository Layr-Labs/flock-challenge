//! Public diagnostic, not a performance candidate.
//! Exactly two randomly assigned process arms; fixed post-proof delay labels
//! each arm without discarding or wrapping slow proof times.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

pub const CONTROL_DELAY_MS: u64 = 40;
pub const TREATMENT_DELAY_MS: u64 = 240;
const UNSET: u8 = u8::MAX;
static ARM: AtomicU8 = AtomicU8::new(UNSET);
static HELD: AtomicBool = AtomicBool::new(false);

fn arm_for_byte(value: u8) -> u8 { value & 1 }
fn hold_delay_ms(arm: u8) -> u64 {
    match arm {
        0 => CONTROL_DELAY_MS,
        1 => TREATMENT_DELAY_MS,
        _ => panic!("unknown diagnostic arm"),
    }
}
fn claim_hold(held: &AtomicBool) -> bool {
    held.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok()
}
fn wait_fixed(delay: Duration) {
    let start = Instant::now();
    loop {
        let elapsed = start.elapsed();
        if elapsed >= delay { return; }
        let remaining = delay - elapsed;
        if remaining > Duration::from_millis(2) {
            std::thread::sleep(remaining - Duration::from_millis(1));
        } else {
            std::hint::spin_loop();
        }
    }
}

/// Called at process initialization, before threads or lazy settings exist.
/// The protected benchmark seed and trial number are never read here.
pub fn init(ranked: bool) {
    if !ranked { return; }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use std::io::Read;
        let forced = std::env::var("FLOCK_DIAG_AB_FORCE_ARM").ok();
        let arm = match forced.as_deref() {
            Some("0") => 0,
            Some("1") => 1,
            Some(_) => panic!("FLOCK_DIAG_AB_FORCE_ARM must be 0 or 1"),
            None => {
                let mut byte = [0u8; 1];
                std::fs::File::open("/dev/urandom")
                    .and_then(|mut file| file.read_exact(&mut byte))
                    .expect("diagnostic process arm needs operating-system randomness");
                arm_for_byte(byte[0])
            }
        };
        // SAFETY: init is called at the top of init_perf_thread_pool, before
        // any worker thread exists. Later calls are suppressed by ONCE.
        unsafe {
            std::env::set_var("FLOCK_SPLIT_TUNE_CPU_BIAS", if arm == 0 { "0" } else { "16" });
        }
        ARM.store(arm, Ordering::Release);
    });
}

/// One fixed hold after the requested proof, never during the fixed warmup.
/// Adopted seed-pipe proofs are held by their producer, not held twice.
pub fn after_timed_proof() {
    let arm = ARM.load(Ordering::Acquire);
    if arm == UNSET || !claim_hold(&HELD) { return; }
    wait_fixed(Duration::from_millis(hold_delay_ms(arm)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_byte_partitions_exactly_in_half() {
        let counts = (0..=u8::MAX).fold([0usize; 2], |mut counts, byte| {
            counts[arm_for_byte(byte) as usize] += 1;
            counts
        });
        assert_eq!(counts, [128, 128]);
    }

    #[test]
    fn delays_are_fixed_per_arm_and_differ_by_two_hundred_ms() {
        assert_eq!(hold_delay_ms(0), CONTROL_DELAY_MS);
        assert_eq!(hold_delay_ms(1), TREATMENT_DELAY_MS);
        assert_eq!(hold_delay_ms(1) - hold_delay_ms(0), 200);
    }

    #[test]
    fn hold_is_claimed_exactly_once() {
        let held = AtomicBool::new(false);
        assert!(claim_hold(&held));
        assert!(!claim_hold(&held));
    }

    #[test]
    fn fixed_wait_never_returns_before_its_deadline() {
        let start = Instant::now();
        wait_fixed(Duration::from_millis(8));
        assert!(start.elapsed() >= Duration::from_millis(8));
    }

    #[test]
    fn uninitialized_nonworker_hold_is_noop() {
        init(false);
        after_timed_proof();
        assert_eq!(ARM.load(Ordering::Acquire), UNSET);
    }
}
