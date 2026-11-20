use super::{Gpu, Id, NIL, ZC_AB_STATE, ZC_R2_STATE, compile_supplemental_pipeline, gpu};
use std::sync::OnceLock;

pub(crate) const CHUNKS: usize = 1 << 15;
pub(crate) const BYTES: usize = CHUNKS * 1024;

struct Pipeline {
    gpu: &'static Gpu,
    pso: Id,
    table: Id,
}

// SAFETY: immutable handles to a process-lifetime pipeline and shared constant table.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

static PIPELINE: OnceLock<Option<Pipeline>> = OnceLock::new();

pub(crate) fn prewarm() {
    PIPELINE.get_or_init(|| {
        let trace = super::super::gpu_zc_ab_debug() || crate::tmark::on();
        let started = trace.then(std::time::Instant::now);
        let gpu = gpu().ok()?;
        let result: Result<Pipeline, String> = unsafe {
            let pool = gpu.pool_push();
            let result = (|| {
                let pso = compile_supplemental_pipeline(
                    gpu,
                    include_str!("../r1_ab_band.metal"),
                    "ab_pre_rows",
                )?;
                let max_threads: u64 = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, super::Sel) -> u64,
                    pso,
                    c"maxTotalThreadsPerThreadgroup"
                );
                if max_threads < 256 {
                    gpu.release(pso);
                    return Err("AB-band pipeline has fewer than 256 threads per group".into());
                }
                let table = match gpu.new_buffer(256 * 64) {
                    Ok(table) => table,
                    Err(e) => {
                        gpu.release(pso);
                        return Err(e);
                    }
                };
                let inverse = crate::ntt::InvNttTableByteSingleGf8::cached_standard_k6();
                std::ptr::copy_nonoverlapping(
                    inverse.data_ptr(),
                    gpu.buffer_contents(table),
                    256 * 64,
                );
                Ok(Pipeline { gpu, pso, table })
            })();
            gpu.pool_pop(pool);
            result
        };
        if let Some(started) = started {
            eprintln!(
                "[r1-gpu-ab-band] prewarm ready={} setup_ms={:.3} finished_ns={}",
                result.is_ok(),
                started.elapsed().as_secs_f64() * 1e3,
                crate::tmark::now_ns(),
            );
        }
        match result {
            Ok(pipeline) => Some(pipeline),
            Err(error) => {
                if trace {
                    eprintln!("[r1-gpu-ab-band] prewarm declined: {error}");
                }
                None
            }
        }
    });
}

pub(crate) struct Views {
    pipeline: &'static Pipeline,
    buffers: [Id; 3],
}

// SAFETY: the retained Metal handles are immutable. A proof-local lease
// drains the only command that can access these views before its buffers die.
unsafe impl Send for Views {}
unsafe impl Sync for Views {}

impl Views {
    pub(crate) fn from_warm_cache(a: usize, b: usize, out: usize, len: usize) -> Option<Self> {
        let pipeline = PIPELINE.get()?.as_ref()?;
        let inputs = ZC_R2_STATE.get()?.as_ref()?.try_lock().ok()?;
        let output = ZC_AB_STATE.get()?.as_ref()?.try_lock().ok()?;
        let input = |address| {
            inputs
                .wraps
                .iter()
                .find(|&&(ptr, bytes, _)| ptr == address && bytes == len)
                .map(|&(_, _, buffer)| buffer)
        };
        let output = output
            .wrap
            .filter(|&(ptr, bytes, _)| ptr == out && bytes == len)?
            .2;
        let buffers = [input(a)?, input(b)?, output];
        // SAFETY: the cache owners still hold the corresponding handles.
        let buffers = buffers.map(|buffer| unsafe { pipeline.gpu.retain(buffer) });
        Some(Self { pipeline, buffers })
    }

    /// Launch only after the proof's CPU producers publish all prefix rows.
    /// The caller holds the proof-local output lease through retirement.
    pub(crate) unsafe fn submit(&self, inject_failure: bool) -> Result<Job, String> {
        let gpu = self.pipeline.gpu;
        let chunks = if inject_failure { CHUNKS / 2 } else { CHUNKS };
        unsafe {
            let pool = gpu.pool_push();
            let result = (|| {
                let cb = gpu.command_buffer()?;
                let enc = gpu.compute_encoder(cb)?;
                gpu.set_pipeline(enc, self.pipeline.pso);
                gpu.set_buffer(enc, self.buffers[0], 0, 0);
                gpu.set_buffer(enc, self.buffers[1], 0, 1);
                gpu.set_buffer(enc, self.pipeline.table, 0, 2);
                gpu.set_buffer(enc, self.buffers[2], 0, 3);
                gpu.dispatch(enc, chunks as u64, 256);
                gpu.end_encoding(enc);
                let cb = gpu.retain(cb);
                gpu.commit_async(cb);
                Ok(Job {
                    gpu,
                    cb,
                    inject_failure,
                    chunks,
                })
            })();
            gpu.pool_pop(pool);
            result
        }
    }
}

impl Drop for Views {
    fn drop(&mut self) {
        for buffer in self.buffers {
            unsafe { self.pipeline.gpu.release(buffer) };
        }
    }
}

pub(crate) struct Job {
    gpu: &'static Gpu,
    cb: Id,
    inject_failure: bool,
    chunks: usize,
}

// SAFETY: Metal command-buffer handles may be waited on from a different thread.
unsafe impl Send for Job {}

impl Job {
    pub(crate) fn retire(mut self, trace: bool) -> bool {
        let start = trace.then(std::time::Instant::now);
        let cb = std::mem::replace(&mut self.cb, NIL);
        // Both success and error return only after Metal terminates the command.
        let status = unsafe { self.gpu.wait_cb(cb) };
        if let Some(start) = start {
            let wait_ms = start.elapsed().as_secs_f64() * 1e3;
            let (gpu_start, gpu_end) = unsafe { super::cb_gpu_interval(self.gpu, cb) };
            crate::tmark::mark_gpu(
                "r1 gpu ab band start",
                "r1 gpu ab band end",
                gpu_start,
                gpu_end,
            );
            crate::tmark::mark("r1 gpu ab band retire");
            let gpu_ms = (gpu_end - gpu_start).max(0.0) * 1e3;
            eprintln!(
                "[r1-gpu-ab-band] retire chunks={} ab_wait_ms={wait_ms:.3} gpu_ms={gpu_ms:.3} status={} forced_failure={}",
                self.chunks,
                if status.is_ok() { "ok" } else { "error" },
                self.inject_failure,
            );
        }
        unsafe { self.gpu.release(cb) };
        status.is_ok() && !self.inject_failure
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        if !self.cb.is_null() {
            unsafe {
                let _ = self.gpu.wait_cb(self.cb);
                self.gpu.release(self.cb);
            }
        }
    }
}
