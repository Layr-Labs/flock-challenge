use super::{
    Gpu, Id, LATCH, LatchState, NIL, ZC_AB_STATE, compile_supplemental_entry,
    compile_supplemental_library, gpu,
};
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};

pub(crate) const JOBS: usize = 128;
pub(crate) const WIDE_JOBS: usize = 2 * JOBS;
pub(crate) const JOB_BYTES: usize = 32 * 8 * 1024;
pub(crate) const FULL_BYTES: usize = 1 << 29;
const BYTES: usize = JOBS * JOB_BYTES;

struct Pipeline {
    gpu: &'static Gpu,
    transform: Id,
    forward: Id,
    table: Id,
    staging: [Id; 2],
    in_use: AtomicBool,
}

// SAFETY: the handles are immutable and live for the process. Only a
// TailStage with the atomic slot may access the two writable staging buffers.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

static PIPELINE: OnceLock<Option<Pipeline>> = OnceLock::new();

pub(crate) fn prewarm() {
    PIPELINE.get_or_init(|| {
        let gpu = gpu().ok()?;
        let trace = crate::gpu_commit::gpu_zc_ab_debug() || crate::tmark::on();
        let starts = trace.then(std::time::Instant::now);
        let built = unsafe { build(gpu) };
        if let Some(starts) = starts {
            eprintln!(
                "[r1-tail-only] prewarm ready={} setup_ms={:.3}",
                built.is_ok(),
                starts.elapsed().as_secs_f64() * 1e3,
            );
            if let Err(ref error) = built {
                eprintln!("[r1-tail-only] prewarm declined: {error}");
            }
        }
        built.ok()
    });
}

unsafe fn build(gpu: &'static Gpu) -> Result<Pipeline, String> {
    const SOURCE: &str = concat!(
        include_str!("../r1_ab_band.metal"),
        "\n",
        include_str!("../r2_z_forward.metal"),
    );
    unsafe {
        let pool = gpu.pool_push();
        let mut handles = [NIL; 5];
        let result = (|| -> Result<Pipeline, String> {
            let library = compile_supplemental_library(gpu, SOURCE)?;
            let compiled = (|| -> Result<(), String> {
                handles[0] = compile_supplemental_entry(gpu, library, "ab_pre_rows")?;
                handles[1] = compile_supplemental_entry(gpu, library, "r2_reconstruct_raw_from_z")?;
                Ok(())
            })();
            gpu.release(library);
            compiled?;
            let (_, width, threads) = gpu.pipeline_resources(handles[1]);
            if width != 32 || threads < 128 || gpu.pipeline_resources(handles[0]).2 < 256 {
                return Err("incompatible R1 tail pipeline resources".into());
            }
            handles[2] = gpu.new_buffer(256 * 64)?;
            handles[3] = gpu.new_buffer(BYTES)?;
            handles[4] = gpu.new_buffer(BYTES)?;
            let inverse = crate::ntt::InvNttTableByteSingleGf8::cached_standard_k6();
            std::ptr::copy_nonoverlapping(
                inverse.data_ptr(),
                gpu.buffer_contents(handles[2]),
                256 * 64,
            );
            Ok(Pipeline {
                gpu,
                transform: handles[0],
                forward: handles[1],
                table: handles[2],
                staging: [handles[3], handles[4]],
                in_use: AtomicBool::new(false),
            })
        })();
        if result.is_err() {
            for handle in handles {
                gpu.release(handle);
            }
        }
        gpu.pool_pop(pool);
        result
    }
}

pub(crate) struct Views {
    pipeline: &'static Pipeline,
    out: Id,
    z: Id,
    z_address: usize,
    out_address: usize,
}

// SAFETY: the retained handles are immutable. The live proof owns the
// allocations, and every command retires before the owning frame returns.
unsafe impl Send for Views {}
unsafe impl Sync for Views {}

static WARM_VIEWS: Mutex<Option<Views>> = Mutex::new(None);

pub(crate) fn warm_views_empty() -> bool {
    let Ok(latch) = LATCH.try_lock() else { return false };
    if matches!(&*latch, LatchState::On(state) if !state.wraps.is_empty()) {
        return false;
    }
    match ZC_AB_STATE.get().and_then(|s| s.as_ref()) {
        Some(state) => state.try_lock().ok().is_some_and(|s| s.wrap.is_none()),
        None => true,
    }
}

fn live_z_view(address: usize, len: usize) -> Option<Id> {
    let latch = LATCH.try_lock().ok()?;
    let LatchState::On(state) = &*latch else { return None };
    state.wraps.iter().find_map(|&(p, n, z)| (p == address && n == len).then_some(z))
}

fn live_out_view(address: usize, len: usize) -> Option<Id> {
    let state = ZC_AB_STATE.get()?.as_ref()?.try_lock().ok()?;
    let (_, _, out) = state.wrap.filter(|&(p, n, _)| p == address && n == len)?;
    Some(out)
}

/// Called on the first proof while it owns both live allocations and before
/// it donates the transform backing. No other prove runs in this worker.
pub(crate) fn record_warm(z_address: usize, out_address: usize, len: usize) {
    if len != FULL_BYTES {
        return;
    }
    let Some(pipeline) = PIPELINE.get().and_then(|p| p.as_ref()) else { return };
    let Ok(latch) = LATCH.try_lock() else { return };
    let LatchState::On(latched) = &*latch else { return };
    let Some(&(_, _, z)) = latched.wraps.iter().find(|&&(p, n, _)| p == z_address && n == len)
    else { return };
    let Some(state) = ZC_AB_STATE.get().and_then(|s| s.as_ref()) else { return };
    let Ok(state) = state.try_lock() else { return };
    let Some((_, _, out)) = state.wrap.filter(|&(p, n, _)| p == out_address && n == len)
    else { return };
    let Ok(mut slot) = WARM_VIEWS.try_lock() else { return };
    if slot.is_none() {
        *slot = Some(Views {
            pipeline,
            out: unsafe { pipeline.gpu.retain(out) },
            z: unsafe { pipeline.gpu.retain(z) },
            z_address,
            out_address,
        });
        crate::tmark::mark("r1 high warm views recorded");
    }
}

pub(crate) fn discard_warm() {
    if let Ok(mut slot) = WARM_VIEWS.try_lock() {
        slot.take();
    }
}

impl Views {
    /// The caller verifies the current output's provenance and the current
    /// Z role-token hit before taking the one-use warm receipt.
    pub(crate) fn take_warm(out_address: usize, len: usize) -> Option<Self> {
        let warm = WARM_VIEWS.try_lock().ok()?.take()?;
        if len != FULL_BYTES || warm.out_address != out_address
            || live_out_view(out_address, len)? != warm.out
        {
            return None;
        }
        crate::tmark::mark("r1 high warm output matched");
        Some(warm)
    }

    pub(crate) fn prepare(&self, address: usize, len: usize) -> Option<TailStage> {
        if len != FULL_BYTES || address != self.z_address || live_z_view(address, len)? != self.z {
            return None;
        }
        let pipeline = self.pipeline;
        if pipeline.in_use.swap(true, Ordering::Acquire) {
            return None;
        }
        crate::tmark::mark("r1 high warm Z matched");
        Some(TailStage {
            pipeline,
            z: unsafe { pipeline.gpu.retain(self.z) },
        })
    }

    /// The proof owns the complete output packet and both underlying
    /// allocations live until this job is retired.
    pub(crate) unsafe fn submit(
        &self,
        stage: TailStage,
        first: usize,
        jobs: usize,
        omit_mixed_ones: bool,
        short: bool,
    ) -> Result<TailJob, String> {
        let pipeline = self.pipeline;
        assert!(core::ptr::eq(stage.pipeline, pipeline));
        assert!(matches!(jobs, JOBS | WIDE_JOBS));
        assert!(first <= FULL_BYTES / JOB_BYTES - jobs);
        let gpu = pipeline.gpu;
        // For a wide short probe, complete one batch and part of the next.
        let dispatched = if short { jobs - JOBS / 2 } else { jobs };
        unsafe {
            let pool = gpu.pool_push();
            let result = (|| -> Result<Id, String> {
                let cb = gpu.command_buffer()?;
                // The same directly allocated, hazard-tracked scratch is
                // bound in each encoder. Metal orders each write after the
                // previous batch's read, and each read after its own write.
                // Only the Z and output offsets change between batches.
                for batch in (0..dispatched).step_by(JOBS) {
                    let bytes = (dispatched - batch).min(JOBS) * JOB_BYTES;
                    let offset = (first + batch) * JOB_BYTES;
                    let blocks = bytes / 2048;
                    let enc = gpu.compute_encoder(cb)?;
                    gpu.set_pipeline(enc, pipeline.forward);
                    gpu.set_buffer(enc, stage.z, offset, 0);
                    gpu.set_buffer(enc, pipeline.staging[0], 0, 1);
                    gpu.set_buffer(enc, pipeline.staging[1], 0, 2);
                    gpu.set_bytes(enc, &(blocks as u32).to_le_bytes(), 3);
                    gpu.dispatch(enc, blocks.div_ceil(128) as u64, 128);
                    gpu.end_encoding(enc);

                    let enc = gpu.compute_encoder(cb)?;
                    gpu.set_pipeline(enc, pipeline.transform);
                    gpu.set_buffer(enc, pipeline.staging[0], 0, 0);
                    gpu.set_buffer(enc, pipeline.staging[1], 0, 1);
                    gpu.set_buffer(enc, pipeline.table, 0, 2);
                    gpu.set_buffer(enc, self.out, offset, 3);
                    gpu.set_bytes(enc, &u32::from(omit_mixed_ones).to_ne_bytes(), 4);
                    gpu.dispatch(enc, (bytes / 1024) as u64, 256);
                    gpu.end_encoding(enc);
                }
                // A failure while encoding either batch leaves this buffer
                // uncommitted and drops the staging lease after pool_pop.
                let cb = gpu.retain(cb);
                gpu.commit_async(cb);
                Ok(cb)
            })();
            gpu.pool_pop(pool);
            result.map(|cb| TailJob {
                cb,
                jobs,
                dispatched,
                first,
                stage,
            })
        }
    }
}

impl Drop for Views {
    fn drop(&mut self) {
        unsafe {
            self.pipeline.gpu.release(self.out);
            self.pipeline.gpu.release(self.z);
        }
    }
}

pub(crate) struct TailStage {
    pipeline: &'static Pipeline,
    z: Id,
}

impl Drop for TailStage {
    fn drop(&mut self) {
        unsafe { self.pipeline.gpu.release(self.z) };
        self.pipeline.in_use.store(false, Ordering::Release);
    }
}

pub(crate) struct TailJob {
    cb: Id,
    jobs: usize,
    dispatched: usize,
    first: usize,
    stage: TailStage,
}

// SAFETY: one owner waits for the command and releases the read-only
// view and staging lease after the command has terminated.
unsafe impl Send for TailJob {}

impl TailJob {
    pub(crate) fn retire(mut self, trace: bool) -> bool {
        let cb = std::mem::replace(&mut self.cb, NIL);
        let gpu = self.stage.pipeline.gpu;
        unsafe {
            let status = gpu.wait_cb(cb);
            let accepted = status.is_ok() && self.dispatched == self.jobs;
            if trace {
                let (start, end) = super::cb_gpu_interval(gpu, cb);
                crate::tmark::mark_gpu(
                    "r1 tail-only gpu start",
                    "r1 tail-only gpu end",
                    start,
                    end,
                );
                crate::tmark::mark("r1 tail-only gpu retire");
                eprintln!(
                    "[r1-tail-only] retired first={} dispatched={} accepted={accepted} device_ms={:.3}",
                    self.first,
                    self.dispatched,
                    (end - start).max(0.0) * 1e3,
                );
            }
            gpu.release(cb);
            accepted
        }
    }
}

impl Drop for TailJob {
    fn drop(&mut self) {
        if !self.cb.is_null() {
            let gpu = self.stage.pipeline.gpu;
            unsafe {
                let _ = gpu.wait_cb(self.cb);
                gpu.release(self.cb);
            }
        }
    }
}
