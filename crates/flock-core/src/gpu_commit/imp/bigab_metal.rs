//! Metal backend of the round-one AB transform's GPU arm (see
//! `crate::gpu_commit::ENV_NO_GPU_AB_ARM`).
//!
//! The kernels ship precompiled in `gpu_commit/bigab.metallib`, built from
//! `gpu_commit/bigab.metal` with `xcrun -sdk macosx metal -c -O3
//! -std=metal3.0 -mmacosx-version-min=13.0 bigab.metal -o bigab.air &&
//! xcrun -sdk macosx metallib bigab.air -o bigab.metallib` (the guard test
//! below pins the source the library was built from). Nothing is compiled
//! from source at run time; a library or pipeline failure leaves the arm
//! off (every unit on the CPU). The sandboxed worker gets no Metal shader
//! cache, so every process pays the backend compile of both pipelines: the
//! generator is one G body in a loop (a straight-line form compiled ten
//! times slower), and the pipelines are built on a helper thread at the
//! first GPU use.
//!
//! - `bigab_gen`: one thread per compression. Its inputs (25 words the
//!   prover writes per launch into a shared buffer: cv, message, counter_lo),
//!   the compression with the witness add stream, and the instance's live a
//!   and b rows (bits `[1024, 15360)`, 448 words each, the producer's rows)
//!   into a staging slab, word-interleaved across the batch: a rolling
//!   LSB-first writer whose bit positions are the same in every thread.
//! - `bigab_xform`: one thread per (instance, live 64-byte block, 8-lane
//!   chunk): the collapsed inverse-NTT byte table in threadgroup memory,
//!   GF(2^8) products, Horner over the block's eight 64-bit words; the
//!   block's 64 bytes land at `instance * 2048 + 128 + 64 j` of the output,
//!   exactly where the CPU writes them.
//!
//! One command buffer per launch: per batch of [`BIGAB_BATCH`] instances a
//! gen dispatch then an xform dispatch (serial encoder, so each batch's
//! xform reads its own staged rows). Every write is the CPU's bytes.

use super::*;

const BIGAB_METALLIB: &[u8] = include_bytes!("../bigab.metallib");

/// FNV-1a (64-bit) of `bigab.metal` when `bigab.metallib` was built.
#[cfg(test)]
const BIGAB_MSL_FNV1A: u64 = 0x9abf_fc05_45a1_392a;

#[cfg(test)]
mod bigab_metallib_guard_tests {
    #[test]
    fn embedded_bigab_metallib_matches_its_source() {
        // On failure: rebuild bigab.metallib from bigab.metal (module doc)
        // and update `BIGAB_MSL_FNV1A`.
        let src = include_str!("../bigab.metal");
        assert_eq!(super::super::fnv1a64(src), super::BIGAB_MSL_FNV1A, "bigab.metallib is stale");
        assert!(!super::BIGAB_METALLIB.is_empty());
    }
}

/// Instances per staging batch (3.5 KiB of staged rows each).
pub(crate) const BIGAB_BATCH: usize = 2048;
/// Live 64-byte transform blocks per instance.
const LIVE_BLOCKS: usize = 28;
/// Staged u32 words per instance (448 of a, 448 of b).
const STAGE_WORDS: usize = 896;
/// Output ring (oracle samples, tuner replays), in instances.
pub(crate) const BIGAB_RING: usize = 2 * BIGAB_BATCH;
/// Most instances one launch takes (half of the ranked transform).
const BIGAB_MAX_INSTANCES: usize = 1 << 17;
/// 32-item groups per xform threadgroup (amortises its table load).
const XFORM_ITERS: usize = 8;

struct BigAb {
    gen_pso: Id,
    xform: Id,
    tab: Id,
    stage: Id,
    ring: Id,
    /// Compression inputs of a launch, 25 words per instance, in batch order.
    inputs: Id,
    /// A queue of the arm's own (`FLOCK_GPU_AB_QUEUE2=1`, diagnostic): the
    /// launch may then run beside the commit graph's buffers instead of
    /// behind them. NIL = the commit queue.
    queue2: Id,
}

// SAFETY: Metal pipelines and buffers are thread-safe. Every launch goes
// on the commit queue, which runs command buffers in commit order, so the
// staging slab is never used by two launches at once.
unsafe impl Send for BigAb {}
unsafe impl Sync for BigAb {}

static BIGAB: OnceLock<Option<BigAb>> = OnceLock::new();

fn bigab_state() -> Option<&'static BigAb> {
    BIGAB
        .get_or_init(|| {
            let gpu = gpu().ok()?;
            let t = std::time::Instant::now();
            let r = unsafe { bigab_init(gpu) };
            if crate::gpu_commit::gpu_ab_debug() {
                match &r {
                    Ok(_) => eprintln!(
                        "[gpu-ab] library + pipelines + buffers in {:.1} ms",
                        t.elapsed().as_secs_f64() * 1e3
                    ),
                    Err(e) => eprintln!("[gpu-ab] init failed (arm off): {e}"),
                }
            }
            r.ok()
        })
        .as_ref()
}

/// The arm's library, pipelines and buffers are built (and usable).
pub(crate) fn bigab_ready() -> bool {
    BIGAB.get().is_some_and(Option::is_some)
}

/// Build the arm's library, pipelines and buffers now (a helper thread at
/// the first GPU use of a process).
pub(crate) fn bigab_prewarm() {
    let _ = bigab_state();
}

unsafe fn bigab_init(gpu: &Gpu) -> Result<BigAb, String> {
    unsafe {
        let api = &gpu.api;
        let create = api.dispatch_data_create.ok_or("dispatch_data_create unavailable")?;
        let pool = gpu.pool_push();
        let r = (|| -> Result<BigAb, String> {
            let data = create(BIGAB_METALLIB.as_ptr().cast(), BIGAB_METALLIB.len(), NIL, NIL);
            if data.is_null() {
                return Err("dispatch_data_create failed".into());
            }
            let mut err: Id = NIL;
            let library: Id = send!(
                api,
                unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id,
                gpu.device,
                c"newLibraryWithData:error:",
                data,
                &mut err
            );
            if let Some(release) = api.dispatch_release {
                release(data);
            }
            if library.is_null() {
                return Err(format!("bigab library: {}", api.error_string(err)));
            }
            let build = |name: &str| -> Result<Id, String> {
                let ns = api.nsstring(name)?;
                let f: Id = send!(api, unsafe extern "C" fn(Id, Sel, Id) -> Id, library, c"newFunctionWithName:", ns);
                if f.is_null() {
                    return Err(format!("bigab kernel {name} not found"));
                }
                let mut pso_err: Id = NIL;
                let pso: Id = send!(
                    api,
                    unsafe extern "C" fn(Id, Sel, Id, *mut Id) -> Id,
                    gpu.device,
                    c"newComputePipelineStateWithFunction:error:",
                    f,
                    &mut pso_err
                );
                send!(api, unsafe extern "C" fn(Id, Sel) -> Id, f, c"release");
                if pso.is_null() {
                    Err(format!("bigab pipeline {name}: {}", api.error_string(pso_err)))
                } else {
                    Ok(pso)
                }
            };
            let gen_pso = build("bigab_gen");
            let xform = build("bigab_xform");
            send!(api, unsafe extern "C" fn(Id, Sel) -> Id, library, c"release");
            let (gen_pso, xform) = match (gen_pso, xform) {
                (Ok(g), Ok(x)) => (g, x),
                (g, x) => {
                    let e = g.as_ref().err().or(x.as_ref().err()).cloned().unwrap_or_default();
                    for p in [g.ok(), x.ok()].into_iter().flatten() {
                        gpu.release(p);
                    }
                    return Err(e);
                }
            };
            let release_all = |ids: &[Id]| {
                for &b in ids {
                    gpu.release(b);
                }
            };
            let tab = match gpu.new_buffer(256 * 64) {
                Ok(b) => b,
                Err(e) => {
                    release_all(&[gen_pso, xform]);
                    return Err(e);
                }
            };
            let table = crate::ntt::InvNttTableByteSingleGf8::cached_standard_k6();
            debug_assert_eq!(table.ell, 64);
            std::ptr::copy_nonoverlapping(table.data_ptr(), gpu.buffer_contents(tab), 256 * 64);
            // GPU-only staging (private storage: never read by the CPU).
            let stage: Id = send!(
                api,
                unsafe extern "C" fn(Id, Sel, u64, u64) -> Id,
                gpu.device,
                c"newBufferWithLength:options:",
                (BIGAB_BATCH * STAGE_WORDS * 4) as u64,
                32u64 // MTLResourceStorageModePrivate
            );
            if stage.is_null() {
                release_all(&[gen_pso, xform, tab]);
                return Err("bigab staging buffer".into());
            }
            let ring = match gpu.new_buffer(BIGAB_RING * 2048) {
                Ok(b) => b,
                Err(e) => {
                    release_all(&[gen_pso, xform, tab, stage]);
                    return Err(e);
                }
            };
            let inputs = match gpu.new_buffer(BIGAB_MAX_INSTANCES * 100) {
                Ok(b) => b,
                Err(e) => {
                    release_all(&[gen_pso, xform, tab, stage, ring]);
                    return Err(e);
                }
            };
            let queue2: Id = if std::env::var_os("FLOCK_GPU_AB_QUEUE2").is_some() {
                send!(api, unsafe extern "C" fn(Id, Sel) -> Id, gpu.device, c"newCommandQueue")
            } else {
                NIL
            };
            Ok(BigAb { gen_pso, xform, tab, stage, ring, inputs, queue2 })
        })();
        gpu.pool_pop(pool);
        r
    }
}

/// One batch of a launch: instances `first..first + count` (`count <=
/// BIGAB_BATCH`), written as output instances `out_inst..out_inst + count`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BigAbBatch {
    pub(crate) first: usize,
    pub(crate) count: usize,
    pub(crate) out_inst: usize,
}

/// Where a launch writes.
#[derive(Clone, Copy)]
pub(crate) enum BigAbOut {
    /// The arm's own ring (oracle samples, tuner replays).
    Ring,
    /// A no-copy view of the round-one AB output storage.
    View(Id),
}

/// An in-flight GPU AB launch.
pub(crate) struct BigAbJob {
    gpu: &'static Gpu,
    cb: Id,
    sig: Option<DoneSig>,
}

// SAFETY: the command buffer is retained and only waited/released through
// this job; Metal command buffers are thread-safe.
unsafe impl Send for BigAbJob {}

/// Encode and commit one launch on the commit queue. `inputs(first, count,
/// dst)` writes each batch's compression inputs (25 words each) before the
/// commit. `build`: create the arm's state if it does not exist yet
/// (untimed proves); otherwise only an already built state is used. `None` =
/// nothing launched (the caller's CPU computes the units).
pub(crate) fn bigab_launch(
    inputs: &(dyn Fn(usize, usize, &mut [u32]) + Sync),
    batches: &[BigAbBatch],
    out: BigAbOut,
    build: bool,
) -> Option<BigAbJob> {
    let st = if build { bigab_state()? } else { BIGAB.get()?.as_ref()? };
    let gpu = gpu().ok()?;
    if batches.iter().map(|b| b.count).sum::<usize>() > BIGAB_MAX_INSTANCES {
        return None;
    }
    // Every earlier launch has been joined (one in flight at a time), so the
    // input buffer is free to rewrite.
    let inp_words: &mut [u32] = unsafe {
        std::slice::from_raw_parts_mut(gpu.buffer_contents(st.inputs).cast::<u32>(), BIGAB_MAX_INSTANCES * 25)
    };
    let mut at = 0usize;
    for b in batches {
        inputs(b.first, b.count, &mut inp_words[25 * at..25 * (at + b.count)]);
        at += b.count;
    }
    unsafe {
        let pool = gpu.pool_push();
        let r = (|| -> Result<BigAbJob, String> {
            let cb = if st.queue2.is_null() {
                gpu.command_buffer()?
            } else {
                let cb: Id = send!(gpu.api, unsafe extern "C" fn(Id, Sel) -> Id, st.queue2, c"commandBuffer");
                if cb.is_null() {
                    return Err("commandBuffer (queue 2) failed".into());
                }
                cb
            };
            let enc = gpu.compute_encoder(cb)?;
            let out_buf = match out {
                BigAbOut::Ring => st.ring,
                BigAbOut::View(b) => b,
            };
            #[repr(C)]
            struct GenP {
                inp: u32,
                pad0: u32,
                first: u32,
                count: u32,
                stride: u32,
                pad: [u32; 3],
            }
            #[repr(C)]
            struct XformP {
                stride: u32,
                count: u32,
                out_inst: u32,
                iters: u32,
            }
            let mut inp = 0usize;
            for b in batches {
                debug_assert!(b.count > 0 && b.count <= BIGAB_BATCH);
                let gp = GenP {
                    inp: inp as u32,
                    pad0: 0,
                    first: b.first as u32,
                    count: b.count as u32,
                    stride: BIGAB_BATCH as u32,
                    pad: [0; 3],
                };
                inp += b.count;
                gpu.set_pipeline(enc, st.gen_pso);
                gpu.set_buffer(enc, st.stage, 0, 0);
                gpu.set_buffer(enc, st.inputs, 0, 2);
                gpu.set_bytes(
                    enc,
                    std::slice::from_raw_parts((&raw const gp).cast::<u8>(), core::mem::size_of::<GenP>()),
                    1,
                );
                gpu.dispatch(enc, b.count.div_ceil(64) as u64, 64);
                let xp = XformP {
                    stride: BIGAB_BATCH as u32,
                    count: b.count as u32,
                    out_inst: b.out_inst as u32,
                    iters: XFORM_ITERS as u32,
                };
                gpu.set_pipeline(enc, st.xform);
                gpu.set_buffer(enc, st.stage, 0, 0);
                gpu.set_buffer(enc, st.tab, 0, 1);
                gpu.set_buffer(enc, out_buf, 0, 2);
                gpu.set_bytes(
                    enc,
                    std::slice::from_raw_parts((&raw const xp).cast::<u8>(), core::mem::size_of::<XformP>()),
                    3,
                );
                gpu.dispatch(enc, (b.count * LIVE_BLOCKS).div_ceil(32 * XFORM_ITERS) as u64, 256);
            }
            gpu.end_encoding(enc);
            let sig = gpu.arm_done(cb);
            let cb = gpu.retain(cb);
            gpu.commit_async(cb);
            crate::tmark::mark("gpu ab committed");
            Ok(BigAbJob { gpu, cb, sig })
        })();
        gpu.pool_pop(pool);
        match r {
            Ok(job) => Some(job),
            Err(e) => {
                if crate::gpu_commit::gpu_ab_debug() {
                    eprintln!("[gpu-ab] launch failed ({e})");
                }
                None
            }
        }
    }
}

impl BigAbJob {
    /// The launch has completed (or failed).
    pub(crate) fn done(&self) -> bool {
        unsafe { self.gpu.cb_done(self.cb, self.sig.as_ref()) }
    }

    /// Wait for the launch (helping the pool first), then `Ok((gpu ms, GPU
    /// end time in seconds))` when it completed, `Err` when Metal reported a
    /// failure.
    pub(crate) fn wait(self) -> Result<(f64, f64), String> {
        crate::gpu_commit::help_pool_until(|| self.done());
        let mut spins = 0u32;
        while !self.done() {
            spins = spins.wrapping_add(1);
            if spins % 64 == 0 {
                std::thread::yield_now();
            } else {
                core::hint::spin_loop();
            }
        }
        unsafe {
            // The signal can fire before the buffer reports completion; its
            // GPU interval is final only then (bounded: 2 ms).
            let t = std::time::Instant::now();
            while self.gpu.cb_status(self.cb) < 4 && t.elapsed() < std::time::Duration::from_millis(2) {
                core::hint::spin_loop();
            }
            let r = self.gpu.done_status(self.cb);
            let (s, e) = cb_gpu_interval(self.gpu, self.cb);
            crate::tmark::mark("gpu ab waited");
            crate::tmark::mark_gpu("gpu ab gpu start", "gpu ab gpu end", s, e);
            r.map(|()| ((e - s) * 1e3, e))
        }
    }

    /// The output ring's bytes (valid after [`Self::wait`] of a ring launch,
    /// until the next launch).
    pub(crate) fn ring_bytes() -> Option<&'static [u8]> {
        let st = BIGAB.get()?.as_ref()?;
        let gpu = gpu().ok()?;
        unsafe { Some(std::slice::from_raw_parts(gpu.buffer_contents(st.ring), BIGAB_RING * 2048)) }
    }
}

impl Drop for BigAbJob {
    fn drop(&mut self) {
        unsafe {
            // A signal is reused only once its buffer is known complete.
            if self.done() {
                Gpu::recycle_done(self.sig.take());
            }
            self.gpu.release(self.cb);
        }
    }
}

/// The no-copy view of the round-one AB output storage `ptr..ptr + len`:
/// the round-one AB completion arm's cached view (it wraps the allocation in
/// the warm-up prove and keeps one view per allocation). Only on a
/// provenance hit of the exact allocation and while that arm is live (a
/// disabled arm stops refreshing its view). The arm never wraps memory
/// itself: a second view over the same allocation is not allowed (the
/// zerocheck K pass may wrap it later), and a wrap inside a timed prove
/// costs page wiring. `None`: the CPU drains every unit.
pub(crate) fn bigab_storage_view(ptr: usize, len: usize, hit: bool) -> Option<Id> {
    if !hit || crate::gpu_commit::gpu_zc_ab_completion_share() == 0 {
        return None;
    }
    if ZC_AB_MODE.load(std::sync::atomic::Ordering::Relaxed) == ZC_AB_DISABLED {
        return None;
    }
    let m = ZC_AB_STATE.get()?.as_ref()?;
    let st = m.lock().ok()?;
    match st.wrap {
        Some((p, l, buf)) if p == ptr && l == len => Some(buf),
        _ => None,
    }
}
