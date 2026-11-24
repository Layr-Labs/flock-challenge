//! Proof-local ownership of an unclaimed suffix of the ordinary CPU queue.

#[doc(hidden)]
#[derive(Default)]
pub struct Round1AbTail {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    armed: std::sync::OnceLock<Armed>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::gpu_commit::r1_ab_tail_only as metal;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
struct Armed {
    views: metal::Views,
    next: AtomicUsize,
    published: usize,
    end: usize,
    omit_mixed_ones: bool,
    allow_wide: bool,
    trace: bool,
    probe: Option<std::ffi::OsString>,
    launch: Mutex<Launch>,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[derive(Default)]
struct Launch {
    closed: bool,
    claimed: Option<std::ops::Range<usize>>,
    job: Option<metal::TailJob>,
}

impl Round1AbTail {
    #[doc(hidden)]
    pub fn enabled() -> bool {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                crate::is_ranked_worker_process()
                    && crate::gpu_commit::gpu_zc_ab_completion_share() != 0
                    && std::env::var_os("FLOCK_NO_R1_GPU_AB_TAIL").is_none()
            })
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        false
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(super) fn arm(
        &self,
        out: usize,
        len: usize,
        published: usize,
        end: usize,
        omit_mixed_ones: bool,
    ) -> Option<Lease<'_>> {
        let views = metal::Views::from_warm_cache(out, len)?;
        assert!(
            self.armed
                .set(Armed {
                    views,
                    next: AtomicUsize::new(0),
                    published,
                    end,
                    omit_mixed_ones,
                    allow_wide: std::env::var_os("FLOCK_NO_R1_GPU_AB_TAIL_256").is_none(),
                    trace: crate::gpu_commit::gpu_zc_ab_debug() || crate::tmark::on(),
                    probe: std::env::var_os("FLOCK_R1_GPU_AB_TAIL_PROBE"),
                    launch: Mutex::default(),
                })
                .is_ok()
        );
        Some(Lease(self.armed.get().expect("armed once")))
    }

    /// Offer from the final successful streamed commitment.
    ///
    /// # Safety
    /// Z is the canonical read-only witness matched to the armed A/B
    /// producer. Its backing stays alive through the A/B lease's retirement.
    #[doc(hidden)]
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub unsafe fn at_root(&self, z: &[crate::field::F128]) {
        let Some(armed) = self.armed.get() else {
            if crate::gpu_commit::gpu_zc_ab_debug() || crate::tmark::on() {
                eprintln!("[r1-tail-only] considered=0 result=not_armed claimed=0");
            }
            return;
        };
        let mut next = armed.next.load(Ordering::Relaxed);
        if let Err(reason) = armed.eligible_jobs(next) {
            armed.report(reason, next, 0);
            return;
        }
        let Ok(mut launch) = armed.launch.try_lock() else {
            armed.report("tail_lock_busy", next, 0);
            return;
        };
        if launch.closed || launch.claimed.is_some() {
            armed.report("tail_closed", next, 0);
            return;
        }
        let probe = armed.probe.as_deref().and_then(std::ffi::OsStr::to_str);
        if probe == Some("refuse") {
            armed.report("forced_refusal", next, 0);
            return;
        }
        let Some(stage) = armed
            .views
            .prepare(z.as_ptr() as usize, core::mem::size_of_val(z))
        else {
            armed.report("staging_unavailable", next, 0);
            return;
        };
        for _ in 0..8 {
            let jobs = match armed.eligible_jobs(next) {
                Ok(jobs) => jobs,
                Err(reason) => {
                    armed.report(reason, next, 0);
                    return;
                }
            };
            match armed.next.compare_exchange(
                next,
                next + jobs,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Record the exact ownership before any fallible encoding.
                    // CPU claims can never be returned to this cursor.
                    launch.claimed = Some(next..next + jobs);
                    if probe == Some("encode-error") {
                        armed.report("forced_encode_error", next, jobs);
                        return;
                    }
                    match unsafe {
                        armed.views.submit(
                            stage,
                            next,
                            jobs,
                            armed.omit_mixed_ones,
                            probe == Some("short"),
                        )
                    } {
                        Ok(job) => {
                            launch.job = Some(job);
                            if armed.trace {
                                crate::tmark::mark("r1 tail-only gpu submit");
                            }
                            armed.report("submitted", next, jobs);
                        }
                        Err(e) => {
                            armed.report("submit_error", next, jobs);
                            if armed.trace {
                                eprintln!("[r1-tail-only] submit_error={e}");
                            }
                        }
                    }
                    return;
                }
                Err(observed) => next = observed,
            }
        }
        armed.report("contention", next, 0);
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Armed {
    fn eligible_jobs(&self, next: usize) -> Result<usize, &'static str> {
        if next < self.published {
            return Err("before_publication_cut");
        }
        let remaining = self.end.saturating_sub(next);
        if self.allow_wide && remaining >= 2 * metal::WIDE_JOBS {
            Ok(metal::WIDE_JOBS)
        } else if remaining >= 2 * metal::JOBS {
            Ok(metal::JOBS)
        } else {
            Err("insufficient_tail")
        }
    }

    fn report(&self, reason: &str, next: usize, claimed: usize) {
        if self.trace {
            eprintln!(
                "[r1-tail-only] considered=1 result={reason} next={next} unclaimed={} claimed={claimed}",
                self.end.saturating_sub(next),
            );
        }
    }

    fn finish(&self, trace: bool) -> Option<std::ops::Range<usize>> {
        let (claimed, job) = {
            let mut launch = self.launch.lock().unwrap_or_else(|e| e.into_inner());
            launch.closed = true;
            (launch.claimed.take(), launch.job.take())
        };
        let claimed = claimed?;
        if job.is_some_and(|job| job.retire(trace)) {
            None
        } else {
            Some(claimed)
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) struct Lease<'a>(&'a Armed);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Lease<'_> {
    pub(super) fn cursor(&self) -> &AtomicUsize {
        &self.0.next
    }

    pub(super) fn finish(&self) -> Option<std::ops::Range<usize>> {
        let redo = self.0.finish(self.0.trace)?;
        if self.0.trace {
            eprintln!(
                "[r1-tail-only] cpu_redo first={} jobs={}",
                redo.start,
                redo.len()
            );
        }
        Some(redo)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for Lease<'_> {
    fn drop(&mut self) {
        // Also closes an in-flight command on panic, before the output
        // allocation or scoped producer can be destroyed.
        let _ = self.0.finish(false);
    }
}
