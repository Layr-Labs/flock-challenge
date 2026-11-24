use super::{Gpu, Id, NIL, ZC_AB_STATE, ZC_R2_STATE, compile_supplemental_pipeline, gpu};
use std::sync::OnceLock;

mod tail {
    include!("r1_ab_tail.rs");
}
pub(crate) use tail::{JOB_BYTES as TAIL_JOB_BYTES, JOBS as TAIL_JOBS, TailJob};

pub(crate) const CHUNKS: usize = 1 << 16;
pub(crate) const BYTES: usize = CHUNKS * 1024;

struct Pipeline {
    gpu: &'static Gpu,
    pso: Id,
    table: Id,
    tail: Option<tail::TailPipeline>,
}

// SAFETY: immutable handles to process-lifetime pipelines and the constant
// table. The tail slot serializes the use of its staging buffers.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

static PIPELINE: OnceLock<Option<Pipeline>> = OnceLock::new();

pub(crate) fn tail_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_R1_GPU_AB_TAIL").is_none())
}

unsafe fn compile_band(gpu: &Gpu) -> Result<(Id, Id), String> {
    const R1: &str = include_str!("../r1_ab_band.metal");
    const COMBINED: &str = concat!(
        include_str!("../r1_ab_band.metal"),
        "\n",
        include_str!("../r2_z_forward.metal"),
    );
    unsafe {
        if tail_enabled()
            && let Ok(library) = super::compile_supplemental_library(gpu, COMBINED)
        {
            let primary = super::compile_supplemental_entry(gpu, library, "ab_pre_rows");
            let forward = if primary.is_ok() {
                super::compile_supplemental_entry(gpu, library, "r2_reconstruct_raw_from_z").unwrap_or(NIL)
            } else {
                NIL
            };
            gpu.release(library);
            if let Ok(primary) = primary {
                return Ok((primary, forward));
            }
        }
        compile_supplemental_pipeline(gpu, R1, "ab_pre_rows").map(|primary| (primary, NIL))
    }
}

pub(crate) fn prewarm() {
    PIPELINE.get_or_init(|| {
        let trace = super::super::gpu_zc_ab_debug() || crate::tmark::on();
        let started = trace.then(std::time::Instant::now);
        let gpu = gpu().ok()?;
        let result: Result<Pipeline, String> = unsafe {
            let pool = gpu.pool_push();
            let result = (|| {
                let (pso, forward) = compile_band(gpu)?;
                let max_threads: u64 = send!(
                    gpu.api,
                    unsafe extern "C" fn(Id, super::Sel) -> u64,
                    pso,
                    c"maxTotalThreadsPerThreadgroup"
                );
                if max_threads < 256 {
                    gpu.release(pso);
                    gpu.release(forward);
                    return Err("AB-band pipeline has fewer than 256 threads per group".into());
                }
                let table = match gpu.new_buffer(256 * 64) {
                    Ok(table) => table,
                    Err(e) => {
                        gpu.release(pso);
                        gpu.release(forward);
                        return Err(e);
                    }
                };
                let inverse = crate::ntt::InvNttTableByteSingleGf8::cached_standard_k6();
                std::ptr::copy_nonoverlapping(
                    inverse.data_ptr(),
                    gpu.buffer_contents(table),
                    256 * 64,
                );
                let tail = tail::TailPipeline::new(gpu, forward);
                Ok(Pipeline { gpu, pso, table, tail })
            })();
            gpu.pool_pop(pool);
            result
        };
        if let Some(started) = started {
            eprintln!(
                "[r1-gpu-ab-band] prewarm ready={} tail_ready={} setup_ms={:.3} finished_ns={}",
                result.is_ok(),
                result.as_ref().is_ok_and(|p| p.tail.is_some()),
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
    pub(crate) fn tail_available(&self) -> bool {
        self.pipeline.tail.is_some()
    }

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
    pub(crate) unsafe fn submit(
        &self,
        omit_mixed_ones: bool,
        inject_failure: bool,
    ) -> Result<Job, String> {
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
                gpu.set_bytes(enc, &u32::from(omit_mixed_ones).to_ne_bytes(), 4);
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
    pub(crate) fn completed_successfully(&self) -> bool {
        !self.inject_failure && unsafe { self.gpu.cb_status(self.cb) == 4 }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{F8, gf2_8::gf8_reduce};
    use crate::ntt::InvNttTableByteSingleGf8;
    use crate::r1cs::ROUND1_MIXED_ONE_ROWS;

    fn scalar_row(a: &[u8], b: &[u8], chunk: usize, row: usize, omit_mixed_ones: bool) -> [u8; 64] {
        let table = InvNttTableByteSingleGf8::cached_standard_k6();
        let base = chunk * 1024 + row * 64;
        let mut accum = [0u16; 64];
        let mut ta = [F8::ZERO; 64];
        let mut tb = [F8::ZERO; 64];
        for k in 0..8 {
            let word = (16 * (chunk % 2) + row) * 8 + k;
            if omit_mixed_ones && ROUND1_MIXED_ONE_ROWS.contains(&word) {
                continue;
            }
            let at = base + 8 * k;
            table.apply_scalar(&a[at..at + 8], &mut ta);
            table.apply_scalar(&b[at..at + 8], &mut tb);
            for lane in 0..64 {
                accum[lane] ^= ((ta[lane] * tb[lane]).0 as u16) << k;
            }
        }
        accum.map(gf8_reduce)
    }

    #[test]
    fn gpu_ab_band_matches_independent_scalar_in_both_representations() {
        prewarm();
        let pipeline = PIPELINE
            .get()
            .and_then(Option::as_ref)
            .expect("real Metal band pipeline");
        let gpu = pipeline.gpu;
        let buffers = [(); 3].map(|()| unsafe { gpu.new_buffer(BYTES).expect("test buffer") });
        let views = Views { pipeline, buffers };
        let [pa, pb, po] = views.buffers.map(|buf| unsafe { gpu.buffer_contents(buf) });
        let samples = [0, 1, CHUNKS / 2 - 1, CHUNKS / 2, CHUNKS - 2, CHUNKS - 1];
        let mut seed = 0x640d_baf8_11e3_ae2du64;
        {
            unsafe {
                std::ptr::write_bytes(pa, 0x73, BYTES);
                std::ptr::write_bytes(pb, 0x35, BYTES);
            }
            let a = unsafe { std::slice::from_raw_parts_mut(pa, BYTES) };
            let b = unsafe { std::slice::from_raw_parts_mut(pb, BYTES) };
            for chunk in samples {
                let start = chunk * 1024;
                for x in a[start..start + 1024]
                    .iter_mut()
                    .chain(&mut b[start..start + 1024])
                {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    *x = seed as u8;
                }
            }
            for chunk in 0..CHUNKS {
                let first_word = (chunk % 2) * 128;
                for (i, &word) in ROUND1_MIXED_ONE_ROWS
                    .iter()
                    .filter(|&&word| word / 128 == chunk % 2)
                    .enumerate()
                {
                    let at = chunk * 1024 + (word - first_word) * 8;
                    // Force a nonzero difference at both mixed rows.
                    a[at..at + 8].fill(if i == 0 { 0xff } else { 0 });
                    b[at..at + 8].fill(0xff);
                }
            }
        }
        let a = unsafe { std::slice::from_raw_parts(pa, BYTES) };
        let b = unsafe { std::slice::from_raw_parts(pb, BYTES) };
        for omit_mixed_ones in [false, true] {
            unsafe { std::ptr::write_bytes(po, 0xa5, BYTES) };
            let job = unsafe { views.submit(omit_mixed_ones, false) }.expect("launch");
            assert!(job.retire(false));
            let out = unsafe { std::slice::from_raw_parts(po, BYTES) };
            for chunk in samples {
                let row_span = if chunk % 2 == 0 { 2..16 } else { 0..14 };
                for row in 0..16 {
                    let expected = if row_span.contains(&row) {
                        scalar_row(a, b, chunk, row, omit_mixed_ones)
                    } else {
                        [0xa5; 64]
                    };
                    let at = chunk * 1024 + row * 64;
                    assert_eq!(
                        &out[at..at + 64],
                        &expected,
                        "chunk {chunk}, row {row}, omit={omit_mixed_ones}"
                    );
                    if omit_mixed_ones && row == if chunk % 2 == 0 { 2 } else { 13 } {
                        assert_ne!(expected, scalar_row(a, b, chunk, row, false));
                    }
                }
            }
        }
    }
}
