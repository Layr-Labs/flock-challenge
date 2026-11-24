use super::*;
use crate::field::F128;
use std::ops::Range;

const MIN_UNCLAIMED: usize = 2 * metal::TAIL_JOBS;

pub(super) struct Claims {
    pub(super) next: AtomicUsize,
    earliest: usize,
    end: usize,
    short: bool,
    launch: Mutex<TailLaunch>,
}

#[derive(Default)]
struct TailLaunch {
    closed: bool,
    first: Option<usize>,
    job: Option<metal::TailJob>,
}

impl Claims {
    pub(super) fn new(earliest: usize, end: usize) -> Self {
        Self {
            next: AtomicUsize::new(0),
            earliest,
            end,
            short: std::env::var_os("FLOCK_R1_GPU_AB_BAND_FAIL").is_some_and(|v| v == "tail"),
            launch: Mutex::new(TailLaunch::default()),
        }
    }

    fn check(&self, next: usize) -> Result<(), &'static str> {
        if next < self.earliest {
            return Err("before_publication_cut");
        }
        if self.end.saturating_sub(next) < MIN_UNCLAIMED {
            return Err("insufficient_tail");
        }
        Ok(())
    }

    // The same RMW order applies to every CPU fetch_add and this one packet
    // claim. A failed CAS never changes the range any CPU worker can take.
    fn reserve(&self, mut next: usize) -> Result<usize, (&'static str, usize)> {
        for _ in 0..8 {
            self.check(next).map_err(|why| (why, next))?;
            match self.next.compare_exchange(
                next,
                next + metal::TAIL_JOBS,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(next),
                Err(observed) => next = observed,
            }
        }
        Err(("contention", next))
    }

    // Close before inspecting the job. An error path owns the whole packet
    // again only after any device writer has terminated.
    pub(super) fn finish(&self, trace: bool) -> Option<Range<usize>> {
        let (first, job) = {
            let mut launch = self.launch.lock().unwrap_or_else(|e| e.into_inner());
            launch.closed = true;
            (launch.first.take(), launch.job.take())
        };
        let first = first?;
        if job.is_some_and(|job| job.retire(trace)) {
            None
        } else {
            Some(first..first + metal::TAIL_JOBS)
        }
    }
}

fn report(reason: &str, next: usize, end: usize, claimed: usize) {
    eprintln!(
        "[r1-ab-tail-gpu] considered=1 result={reason} next={next} unclaimed={} claimed={claimed}",
        end.saturating_sub(next),
    );
}

impl Round1AbGpuBand {
    /// Offer a packet only after the commitment root is final.
    ///
    /// # Safety
    /// Z is the canonical readonly BLAKE3 witness of the same active
    /// A/B lease. Its allocation must live until that lease is retired.
    pub unsafe fn offer_tail_at_root(&self, z: &[F128]) {
        let Some(armed) = self.armed.get() else {
            if crate::tmark::on() || crate::gpu_commit::gpu_zc_ab_debug() {
                eprintln!("[r1-ab-tail-gpu] considered=0 result=not_armed claimed=0");
            }
            return;
        };
        let Some(claims) = &armed.tail_claims else {
            if armed.trace {
                eprintln!("[r1-ab-tail-gpu] considered=0 result=no_tail_receipt claimed=0");
            }
            return;
        };
        let log = |reason, next, claimed| {
            if armed.trace {
                report(reason, next, claims.end, claimed)
            }
        };
        let observed = claims.next.load(Ordering::Relaxed);
        if let Err(why) = claims.check(observed) {
            log(why, observed, 0);
            return;
        }
        let Ok(mut launch) = claims.launch.try_lock() else {
            log("tail_lock_busy", observed, 0);
            return;
        };
        if launch.closed || launch.first.is_some() {
            log("tail_closed", observed, 0);
            return;
        }
        let primary_ready = armed.launch.try_lock().ok().is_some_and(|primary| {
            primary
                .job
                .as_ref()
                .is_some_and(metal::Job::completed_successfully)
        });
        if !primary_ready {
            log("primary_not_complete", observed, 0);
            return;
        }
        let Some(stage) = armed
            .views
            .prepare_tail(z.as_ptr() as usize, core::mem::size_of_val(z))
        else {
            log("stage_unavailable", observed, 0);
            return;
        };
        let first = match claims.reserve(observed) {
            Ok(first) => first,
            Err((why, next)) => {
                log(why, next, 0);
                return;
            }
        };
        // Publish ownership before any fallible encoder call. Do not rewind
        // the cursor: other workers already own jobs after this packet.
        launch.first = Some(first);
        match unsafe {
            armed
                .views
                .submit_tail(stage, first, armed.omit_mixed_ones, claims.short)
        } {
            Ok(job) => {
                launch.job = Some(job);
                if armed.trace {
                    crate::tmark::mark("r1 gpu ab tail submit");
                }
                log("submitted", first, metal::TAIL_JOBS);
            }
            Err(e) => {
                log("submit_error", first, metal::TAIL_JOBS);
                if armed.trace {
                    eprintln!("[r1-ab-tail-gpu] submit_error={e}");
                }
            }
        }
    }
}
