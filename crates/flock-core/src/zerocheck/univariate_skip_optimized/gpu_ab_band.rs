//! One proof's handoff between the ranked A/B producer and the commit arm.

#[doc(hidden)]
#[derive(Default)]
pub struct Round1AbGpuBand {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    armed: std::sync::OnceLock<Armed>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    submit_at_ready: bool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::gpu_commit::r1_ab_band as metal;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::{
    Mutex,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod tail {
    include!("gpu_ab_band_tail.rs");
}

#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
pub(super) fn uses_canonical_inverse(table: &crate::ntt::InvNttTableByteSingleGf8) -> bool {
    core::ptr::eq(
        table,
        crate::ntt::InvNttTableByteSingleGf8::cached_standard_k6(),
    )
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct Armed {
    views: metal::Views,
    jobs: AtomicUsize,
    ready: AtomicU64,
    launch: Mutex<Launch>,
    submit_at_ready: bool,
    trace: bool,
    inject_failure: bool,
    omit_mixed_ones: bool,
    tail_claims: Option<tail::Claims>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Default)]
struct Launch {
    closed: bool,
    job: Option<metal::Job>,
}

impl Round1AbGpuBand {
    #[doc(hidden)]
    pub fn enabled() -> bool {
        static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            crate::is_ranked_worker_process()
                && crate::gpu_commit::gpu_commit_enabled()
                && crate::gpu_commit::gpu_zc_ab_completion_share() != 0
                && std::env::var_os("FLOCK_NO_R1_GPU_AB_BAND").is_none()
        });
        *ENABLED
    }

    #[doc(hidden)]
    pub fn with_precommitted_cb2(cb2_enqueued: bool) -> Self {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                std::env::var_os("FLOCK_NO_R1_GPU_AB_BAND_AT_READY").is_none()
            });
            Self {
                submit_at_ready: cb2_enqueued && *ENABLED,
                ..Self::default()
            }
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            let _ = cb2_enqueued;
            Self::default()
        }
    }

    #[doc(hidden)]
    pub fn at_root(&self) {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(armed) = self.armed.get() {
            Self::try_submit(armed, "root");
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn after_cb3(&self) {
        static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("FLOCK_NO_R1_GPU_AB_BAND_AFTER_CB3").is_none()
        });
        if *ENABLED && let Some(armed) = self.armed.get() {
            Self::try_submit(armed, "post_cb3");
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn try_submit(armed: &Armed, trigger: &str) {
        let ready = armed.ready.load(Ordering::Acquire);
        let trigger_ns = if armed.trace {
            crate::tmark::now_ns()
        } else {
            0
        };
        if ready == 0 {
            if armed.trace {
                eprintln!(
                    "[r1-gpu-ab-band] {trigger} declined: jobs_ready={}",
                    armed.jobs.load(Ordering::Relaxed)
                );
            }
            return;
        }
        let Ok(mut launch) = armed.launch.try_lock() else {
            return;
        };
        if launch.closed {
            if armed.trace {
                eprintln!("[r1-gpu-ab-band] {trigger} declined: AB retired");
            }
            return;
        }
        if launch.job.is_some() {
            return;
        }
        let inject = armed.inject_failure;
        // SAFETY: the acquired readiness stamp follows all raw prefix
        // writes. The AB lease keeps output alive, and no CPU job writes
        // the reserved output until this command has terminated.
        match unsafe { armed.views.submit(armed.omit_mixed_ones, inject) } {
            Ok(job) => {
                launch.job = Some(job);
                if armed.trace {
                    let submit = crate::tmark::now_ns();
                    crate::tmark::mark_ns("r1 gpu ab band ready", ready);
                    crate::tmark::mark_ns("r1 gpu ab band submit", submit);
                    let ready_ms = trigger_ns.saturating_sub(ready) as f64 / 1e6;
                    let submit_ms = (submit - trigger_ns) as f64 / 1e6;
                    let chunks = if inject {
                        metal::CHUNKS / 2
                    } else {
                        metal::CHUNKS
                    };
                    eprintln!(
                        "[r1-gpu-ab-band] launched trigger={trigger} elapsed_since_ready_ms={ready_ms:.3} {trigger}_submit_ms={submit_ms:.3} root_wait_ms=0 chunks={chunks} start=0 omit_mixed_ones={}",
                        armed.omit_mixed_ones,
                    );
                }
            }
            Err(e) => {
                if armed.trace {
                    eprintln!("[r1-gpu-ab-band] {trigger} declined: {e}");
                }
            }
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn arm(
        &self,
        a: usize,
        b: usize,
        out: usize,
        len: usize,
        omit_mixed_ones: bool,
        tail_geometry: Option<(usize, usize)>,
    ) -> Option<Lease<'_>> {
        let trace = crate::gpu_commit::gpu_zc_ab_debug() || crate::tmark::on();
        let start = trace.then(std::time::Instant::now);
        let views = metal::Views::from_warm_cache(a, b, out, len);
        if let Some(start) = start {
            eprintln!(
                "[r1-gpu-ab-band] cache_hit={} arm_ms={:.3}",
                views.is_some(),
                start.elapsed().as_secs_f64() * 1e3
            );
        }
        let views = views?;
        let tail_claims = tail_geometry
            .filter(|_| views.tail_available())
            .map(|(earliest, end)| tail::Claims::new(earliest, end));
        assert!(
            self.armed
                .set(Armed {
                    views,
                    jobs: AtomicUsize::new(0),
                    ready: AtomicU64::new(0),
                    launch: Mutex::new(Launch::default()),
                    submit_at_ready: self.submit_at_ready,
                    trace,
                    inject_failure: std::env::var_os("FLOCK_R1_GPU_AB_BAND_FAIL")
                        .is_some_and(|v| v == "1"),
                    omit_mixed_ones,
                    tail_claims,
                })
                .is_ok()
        );
        Some(Lease(self.armed.get().expect("armed AB-band")))
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) struct Lease<'a>(&'a Armed);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Lease<'_> {
    pub(super) const UNIT_BYTES: usize = 8 * 1024;
    pub(super) const UNITS_PER_JOB: usize = 32;

    /// The producers must cover exactly the byte range the GPU can read.
    pub(super) const JOBS: usize = {
        let job_bytes = Self::UNIT_BYTES * Self::UNITS_PER_JOB;
        assert!(metal::BYTES.is_multiple_of(job_bytes));
        metal::BYTES / job_bytes
    };

    pub(super) fn publish_job(&self) {
        let armed = self.0;
        // AcqRel on the same RMW chain brings all preceding jobs' raw
        // writes to the final publisher of the readiness stamp.
        if armed.jobs.fetch_add(1, Ordering::AcqRel) == Self::JOBS - 1 {
            let ready = if armed.trace {
                crate::tmark::now_ns().max(1)
            } else {
                1
            };
            armed.ready.store(ready, Ordering::Release);
            if armed.submit_at_ready {
                // The streamed commit queued cb2 before the producer began.
                Round1AbGpuBand::try_submit(armed, "raw_ready");
            }
        }
    }

    pub(super) fn finish(&self) -> bool {
        let job = {
            let mut launch = self.0.launch.lock().unwrap_or_else(|e| e.into_inner());
            launch.closed = true;
            launch.job.take()
        };
        job.is_some_and(|job| job.retire(self.0.trace))
    }

    pub(super) fn trace(&self) -> bool {
        self.0.trace
    }

    pub(super) fn tail_cursor(&self) -> Option<&AtomicUsize> {
        self.0.tail_claims.as_ref().map(|claims| &claims.next)
    }

    pub(super) fn finish_tail(&self) -> Option<std::ops::Range<usize>> {
        self.0.tail_claims.as_ref()?.finish(self.0.trace)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        // Unwind path: take and drop the job while the output is still alive.
        if let Some(tail) = &self.0.tail_claims {
            let _ = tail.finish(false);
        }
        let mut launch = self.0.launch.lock().unwrap_or_else(|e| e.into_inner());
        launch.closed = true;
        drop(launch.job.take());
    }
}

#[cfg(test)]
mod tests {
    use super::uses_canonical_inverse;
    use crate::{
        field::F8,
        ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8},
    };

    #[test]
    fn gpu_band_rejects_alternate_k6_inverse_table() {
        assert!(uses_canonical_inverse(
            InvNttTableByteSingleGf8::cached_standard_k6()
        ));
        let source = AdditiveNttGf8::new(6, F8::ZERO);
        let other_coset = AdditiveNttGf8::new(6, F8(1 << 7));
        let alternate = InvNttTableByteSingleGf8::new(&source, &other_coset);
        assert!(!uses_canonical_inverse(&alternate));
    }
}
