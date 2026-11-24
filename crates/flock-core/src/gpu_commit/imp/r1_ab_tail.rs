use super::{Gpu, Id, NIL, Views};
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) const JOBS: usize = 128;
pub(crate) const JOB_BYTES: usize = 32 * 8 * 1024;
const BYTES: usize = JOBS * JOB_BYTES;

pub(super) struct TailPipeline {
    forward: Id,
    staging: [Id; 2],
    in_use: AtomicBool,
}

impl TailPipeline {
    pub(super) unsafe fn new(gpu: &Gpu, forward: Id) -> Option<Self> {
        if forward.is_null() {
            return None;
        }
        let mut staging = [NIL; 2];
        let ready = unsafe {
            let (_, width, max_threads) = gpu.pipeline_resources(forward);
            width == 32
                && max_threads >= 128
                && staging.iter_mut().all(|slot| {
                    gpu.new_buffer(BYTES).is_ok_and(|buffer| {
                        *slot = buffer;
                        true
                    })
                })
        };
        if !ready {
            for handle in staging.into_iter().chain([forward]) {
                unsafe { gpu.release(handle) };
            }
            return None;
        }
        Some(Self {
            forward,
            staging,
            in_use: AtomicBool::new(false),
        })
    }
}

pub(crate) struct TailStage {
    gpu: &'static Gpu,
    pipeline: &'static TailPipeline,
    z: Id,
}

// SAFETY: Metal handles are immutable. The staging lease serializes writers
// and the owner releases it only after the command retires.
unsafe impl Send for TailStage {}

impl Drop for TailStage {
    fn drop(&mut self) {
        unsafe { self.gpu.release(self.z) };
        self.pipeline.in_use.store(false, Ordering::Release);
    }
}

impl Views {
    /// Get the two warm staging buffers and this proof's exact read-only Z
    /// mapping. The caller keeps the underlying Z backing alive.
    pub(crate) fn prepare_tail(&self, addr: usize, len: usize) -> Option<TailStage> {
        use super::super::{LATCH, LatchState};
        if len != 1 << 29 {
            return None;
        }
        let pipeline = self.pipeline.tail.as_ref()?;
        let latch = LATCH.try_lock().ok()?;
        let LatchState::On(state) = &*latch else {
            return None;
        };
        let &(_, _, z) = state
            .wraps
            .iter()
            .find(|&&(p, n, _)| p == addr && n == len)?;
        if pipeline.in_use.swap(true, Ordering::Acquire) {
            return None;
        }
        let gpu = self.pipeline.gpu;
        Some(TailStage {
            gpu,
            pipeline,
            z: unsafe { gpu.retain(z) },
        })
    }

    /// The caller owns the full output packet after its CPU queue claim.
    /// Keep the proof's Z and output allocations alive through retirement.
    pub(crate) unsafe fn submit_tail(
        &self,
        stage: TailStage,
        first: usize,
        omit_mixed_ones: bool,
        short: bool,
    ) -> Result<TailJob, String> {
        let gpu = self.pipeline.gpu;
        let bytes = if short { BYTES / 2 } else { BYTES };
        let offset = first * JOB_BYTES;
        assert!(offset <= (1 << 29) - BYTES);
        let blocks = bytes / 2048;
        unsafe {
            let pool = gpu.pool_push();
            let result = (|| {
                let cb = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb)?;
                gpu.set_pipeline(enc, stage.pipeline.forward);
                gpu.set_buffer(enc, stage.z, offset, 0);
                gpu.set_buffer(enc, stage.pipeline.staging[0], 0, 1);
                gpu.set_buffer(enc, stage.pipeline.staging[1], 0, 2);
                gpu.set_bytes(enc, &(blocks as u32).to_le_bytes(), 3);
                gpu.dispatch(enc, blocks.div_ceil(128) as u64, 128);
                gpu.end_encoding(enc);

                let enc = gpu.compute_encoder(cb)?;
                gpu.set_pipeline(enc, self.pipeline.pso);
                gpu.set_buffer(enc, stage.pipeline.staging[0], 0, 0);
                gpu.set_buffer(enc, stage.pipeline.staging[1], 0, 1);
                gpu.set_buffer(enc, self.pipeline.table, 0, 2);
                gpu.set_buffer(enc, self.buffers[2], offset, 3);
                gpu.set_bytes(enc, &u32::from(omit_mixed_ones).to_ne_bytes(), 4);
                gpu.dispatch(enc, (bytes / 1024) as u64, 256);
                gpu.end_encoding(enc);
                let cb = gpu.retain(cb);
                gpu.commit_async(cb);
                Ok(TailJob {
                    cb,
                    short,
                    first,
                    stage,
                })
            })();
            gpu.pool_pop(pool);
            result
        }
    }
}

pub(crate) struct TailJob {
    cb: Id,
    short: bool,
    first: usize,
    stage: TailStage,
}

// SAFETY: the command is waited and released on one owner after transfer.
unsafe impl Send for TailJob {}

impl TailJob {
    pub(crate) fn retire(mut self, trace: bool) -> bool {
        let cb = std::mem::replace(&mut self.cb, NIL);
        let gpu = self.stage.gpu;
        unsafe {
            let status = gpu.wait_cb(cb);
            if trace {
                let (start, end) = super::super::cb_gpu_interval(gpu, cb);
                crate::tmark::mark_gpu("r1 gpu ab tail start", "r1 gpu ab tail end", start, end);
                crate::tmark::mark("r1 gpu ab tail retire");
                eprintln!(
                    "[r1-ab-tail-gpu] retired first={} dispatched={} accepted={} device_ms={:.3}",
                    self.first,
                    if self.short { JOBS / 2 } else { JOBS },
                    usize::from(status.is_ok() && !self.short),
                    (end - start).max(0.0) * 1e3,
                );
            }
            gpu.release(cb);
            status.is_ok() && !self.short
        }
    }
}

impl Drop for TailJob {
    fn drop(&mut self) {
        if !self.cb.is_null() {
            unsafe {
                let _ = self.stage.gpu.wait_cb(self.cb);
                self.stage.gpu.release(self.cb);
            }
        }
    }
}
