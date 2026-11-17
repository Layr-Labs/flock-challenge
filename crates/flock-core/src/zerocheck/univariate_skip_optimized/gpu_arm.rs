//! Host side of the round-one AB transform's GPU arm (see
//! `crate::gpu_commit::ENV_NO_GPU_AB_ARM`; Metal side:
//! `gpu_commit::imp::bigab_metal`).
//!
//! The GPU takes the units `[n - g, n)` of the drain (a unit is one quad:
//! four compressions, 8 KiB of A and of B, 8 KiB of output), `g` even and
//! never below the first unit the prover leaves unpublished for round two.
//! The CPU drains `[0, n - g)` as before. Three modes:
//! - oracle (the process's first transform, i.e. the untimed warm-up): the
//!   CPU drains every unit; the GPU computes two batches of the GPU range
//!   into its own ring, compared byte for byte with the CPU's blocks after
//!   the drain. The verdict admits or disables the arm for the process.
//! - admitted: the GPU writes its units into the output storage through
//!   the cached no-copy view of the allocation (never a new view inside an
//!   admitted prove); a Metal failure makes the CPU drain them.
//! - replay (the split tuner's exact-contention replay): the GPU computes
//!   its units into its ring (the replay discards its output), so the sweep
//!   prices the window with the arm in it.

/// Instances per unit (quad).
const UNIT_INSTANCES: usize = 4;
/// Bytes of A per unit.
const UNIT_BYTES: usize = 8192;
/// Units of the ranked transform (`2^18` compressions).
const RANKED_UNITS: usize = 1 << 16;

/// The prover's input writer: `inputs(first, count, dst)` writes the inputs
/// of compressions `first..first + count` into `dst`, 25 words each (cv[8],
/// m[16], counter_lo).
pub type GpuAbInputs = dyn Fn(usize, usize, &mut [u32]) + Sync;

/// One prove's registration (see [`gpu_ab_seed_scope`]).
#[derive(Clone, Copy)]
struct Seed {
    a: usize,
    b: usize,
    len: usize,
    /// Lifetime erased: reachable only while the registering scope lives.
    inputs: &'static GpuAbInputs,
    min_unit: usize,
    id: u64,
}

static SEEDS: std::sync::Mutex<Vec<Seed>> = std::sync::Mutex::new(Vec::new());
static SEED_NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Registration of a prove's GPU AB inputs; dropping it withdraws it.
pub struct GpuAbSeedScope<'a> {
    id: u64,
    _inputs: core::marker::PhantomData<&'a ()>,
}

impl Drop for GpuAbSeedScope<'_> {
    fn drop(&mut self) {
        let mut v = SEEDS.lock().unwrap_or_else(|e| e.into_inner());
        v.retain(|s| s.id != self.id);
    }
}

/// Register the GPU AB arm's inputs for the prove whose packed A/B buffers
/// are `a_packed` / `b_packed` (`len` bytes each): unit `u` of their rows is
/// compressions `4u..4u + 4`, whose inputs `inputs` writes, and units below
/// `min_unit` must stay with the CPU (their rows are published for round
/// two). Without a registration the arm never runs for those buffers.
///
/// # Safety
/// While the scope lives, the rows of every unit at or past `min_unit` in
/// both buffers (written or deferred) are exactly the witness rows of the
/// compressions `inputs` describes (counter_hi 0, block_len 64, flags 11).
pub unsafe fn gpu_ab_seed_scope<'a>(
    a_packed: *const u8,
    b_packed: *const u8,
    len: usize,
    inputs: &'a (dyn Fn(usize, usize, &mut [u32]) + Sync + 'a),
    min_unit: usize,
) -> GpuAbSeedScope<'a> {
    let id = SEED_NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // SAFETY: the registration (the only holder of this reference) is
    // withdrawn when the scope drops, before `'a` ends.
    let inputs: &'static GpuAbInputs = unsafe {
        core::mem::transmute::<&'a (dyn Fn(usize, usize, &mut [u32]) + Sync + 'a), &'static GpuAbInputs>(inputs)
    };
    SEEDS.lock().unwrap_or_else(|e| e.into_inner()).push(Seed {
        a: a_packed as usize,
        b: b_packed as usize,
        len,
        inputs,
        min_unit,
        id,
    });
    GpuAbSeedScope { id, _inputs: core::marker::PhantomData }
}

/// First unit of the admitted GPU AB arm that is running now (`usize::MAX`:
/// none). Set before the CPU drain starts, cleared before the join's
/// fallback, so a reader on the drain's workers sees it for the whole drain.
static GPU_FIRST_UNIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// While an admitted GPU AB arm runs: the first unit it owns (it owns every
/// unit from there to the end). The stripe producer uses it to fill the
/// GPU units' stripe groups from z on the drain's idle workers.
pub fn gpu_ab_first_unit() -> Option<usize> {
    match GPU_FIRST_UNIT.load(std::sync::atomic::Ordering::Acquire) {
        usize::MAX => None,
        u => Some(u),
    }
}

fn seed_for(a: *const u8, b: *const u8, len: usize) -> Option<Seed> {
    let v = SEEDS.lock().unwrap_or_else(|e| e.into_inner());
    v.iter().rev().find(|s| s.a == a as usize && s.b == b as usize && s.len == len).copied()
}

/// What the precompute body hands the arm.
pub(super) struct GpuArmCtx {
    pub(super) a_packed: *const u8,
    pub(super) b_packed: *const u8,
    pub(super) total_bytes: usize,
    pub(super) out: *mut u8,
    pub(super) producer_unit_bytes: Option<usize>,
    pub(super) provenance_hit: bool,
    pub(super) replay: bool,
    /// Ranked BLAKE3 shape with the identity-prefix and output-window rows
    /// omitted, and a store flavor that leaves the dead rows alone.
    pub(super) shape_ok: bool,
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod imp {
    use super::*;
    use crate::gpu_commit::{BIGAB_BATCH, BIGAB_RING, BigAbBatch, BigAbJob, BigAbOut};

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Mode {
        Oracle,
        /// Oracle whose GPU half runs after the drain on a thread of its own
        /// (the pipelines were still building when the drain started).
        OracleDeferred,
        Admitted,
        Replay,
    }

    /// Instances per oracle batch (two batches: the GPU range's first and
    /// last instances).
    const ORACLE_INST: usize = 1024;

    pub(in super::super) struct GpuArm {
        job: Option<BigAbJob>,
        inputs: &'static GpuAbInputs,
        mode: Mode,
        cpu_units: usize,
        n_units: usize,
        out: *mut u8,
        batches: Vec<BigAbBatch>,
        started: std::time::Instant,
    }

    /// Batches covering instances `first..end`, written as output instances
    /// `out_base + (i - first)` (`None`: each batch at `first` itself).
    fn batches(first: usize, end: usize, ring: bool) -> Vec<BigAbBatch> {
        (first..end)
            .step_by(BIGAB_BATCH)
            .map(|f| BigAbBatch { first: f, count: BIGAB_BATCH.min(end - f), out_inst: if ring { 0 } else { f } })
            .collect()
    }

    /// Compare the GPU's oracle batches in the ring with the CPU's blocks:
    /// `cpu(j, k)` = byte `k` of the `j`-th sample instance's 1,792 live
    /// bytes (batch order). True only on a completed launch and equal bytes.
    fn oracle_verdict(
        r: Result<f64, String>,
        batches: &[BigAbBatch],
        cpu: impl Fn(usize, usize) -> u8,
        debug: bool,
    ) -> bool {
        let (gpu_ms, ring) = match (r, BigAbJob::ring_bytes()) {
            (Ok(ms), Some(ring)) => (ms, ring),
            (r, _) => {
                if debug {
                    eprintln!("[gpu-ab] oracle launch failed ({r:?})");
                }
                return false;
            }
        };
        let mut bad = 0usize;
        let mut j = 0usize;
        for b in batches {
            for i in 0..b.count {
                let g = &ring[(b.out_inst + i) * 2048 + 128..(b.out_inst + i) * 2048 + 1920];
                if let Some(k) = (0..1792).find(|&k| g[k] != cpu(j, k)) {
                    if debug && bad == 0 {
                        eprintln!("[gpu-ab] oracle: instance {} differs first at live byte {k}", b.first + i);
                    }
                    bad += 1;
                }
                j += 1;
            }
        }
        if debug {
            eprintln!(
                "[gpu-ab] oracle: {bad} of {j} instances differ; sample gpu {gpu_ms:.2} ms ({:.3} us/instance)",
                gpu_ms * 1e3 / j.max(1) as f64
            );
        }
        bad == 0
    }

    impl GpuArm {
        pub(in super::super) fn start(ctx: GpuArmCtx) -> Option<Self> {
            let share = crate::gpu_commit::gpu_ab_share64();
            let verdict = crate::gpu_commit::gpu_ab_verdict();
            let debug = crate::gpu_commit::gpu_ab_debug();
            let skip = |why: &str| {
                if debug {
                    eprintln!("[gpu-ab] no arm: {why}");
                }
            };
            if share == 0 || verdict == 2 || !ctx.shape_ok {
                skip(&format!("share {share}, verdict {verdict}, shape {}", ctx.shape_ok));
                return None;
            }
            if ctx.total_bytes != RANKED_UNITS * UNIT_BYTES || ctx.producer_unit_bytes.is_some_and(|u| u != UNIT_BYTES) {
                skip("not the ranked transform size");
                return None;
            }
            // The oracle runs only in the process's first eligible transform
            // (the untimed warm-up prove); without a verdict by then the arm
            // stays off, so no timed prove ever runs the oracle.
            static FIRST_SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            let first = !FIRST_SEEN.swap(true, std::sync::atomic::Ordering::AcqRel);
            if verdict == 0 && !first && !ctx.replay {
                // A deferred oracle may still be running: no arm for this
                // transform, its verdict decides later ones.
                if !crate::gpu_commit::gpu_ab_oracle_pending(None) {
                    crate::gpu_commit::gpu_ab_verdict_off_if_unset();
                }
                skip("no oracle verdict");
                return None;
            }
            let Some(seed) = seed_for(ctx.a_packed, ctx.b_packed, ctx.total_bytes) else {
                skip("no seed registered for these buffers");
                return None;
            };
            let n_units = RANKED_UNITS;
            let cap = n_units.saturating_sub(seed.min_unit) & !1;
            let g = ((n_units * share / 64) & !1).min(cap);
            if g < 2 * BIGAB_BATCH / UNIT_INSTANCES {
                skip(&format!("share too small (g {g}, min unit {})", seed.min_unit));
                return None;
            }
            let lo_inst = (n_units - g) * UNIT_INSTANCES;
            let end_inst = n_units * UNIT_INSTANCES;
            let (mode, batches, out, build) = if ctx.replay {
                if verdict != 1 {
                    skip("replay without an admitted arm");
                    return None;
                }
                crate::gpu_commit::gpu_ab_note_replay_gpu_end(0.0);
                (Mode::Replay, batches(lo_inst, end_inst, true), BigAbOut::Ring, false)
            } else if verdict == 0 {
                // Oracle: the first and the last batch of the GPU range, into
                // the ring's two halves. The CPU drains every unit.
                let sample = vec![
                    BigAbBatch { first: lo_inst, count: ORACLE_INST, out_inst: 0 },
                    BigAbBatch { first: end_inst - ORACLE_INST, count: ORACLE_INST, out_inst: ORACLE_INST },
                ];
                debug_assert!(2 * ORACLE_INST <= BIGAB_RING && ORACLE_INST <= BIGAB_BATCH);
                if !crate::gpu_commit::bigab_ready() {
                    // Never wait for the pipelines here: the GPU half of the
                    // oracle runs after the drain, on its own thread.
                    if debug {
                        eprintln!("[gpu-ab] oracle deferred (pipelines still building)");
                    }
                    let cpu_units = n_units;
                    return Some(Self {
                        job: None,
                        inputs: seed.inputs,
                        mode: Mode::OracleDeferred,
                        cpu_units,
                        n_units,
                        out: ctx.out,
                        batches: sample,
                        started: std::time::Instant::now(),
                    });
                }
                (Mode::Oracle, sample, BigAbOut::Ring, false)
            } else {
                let Some(view) =
                    crate::gpu_commit::bigab_storage_view(ctx.out as usize, ctx.total_bytes, ctx.provenance_hit)
                else {
                    if debug {
                        eprintln!("[gpu-ab] no cached view of the AB output (hit={}); CPU drains all", ctx.provenance_hit);
                    }
                    return None;
                };
                (Mode::Admitted, batches(lo_inst, end_inst, false), BigAbOut::View(view), false)
            };
            let started = std::time::Instant::now();
            let Some(job) = crate::gpu_commit::bigab_launch(seed.inputs, &batches, out, build) else {
                skip("launch failed");
                return None;
            };
            let cpu_units = if mode == Mode::Oracle { n_units } else { n_units - g };
            if mode == Mode::Admitted {
                GPU_FIRST_UNIT.store(cpu_units, std::sync::atomic::Ordering::Release);
            }
            if debug {
                eprintln!(
                    "[gpu-ab] {mode:?}: gpu units {}..{n_units} ({} instances in {} batches), cpu units 0..{cpu_units}, launch {:.2} ms",
                    n_units - g,
                    batches.iter().map(|b| b.count).sum::<usize>(),
                    batches.len(),
                    started.elapsed().as_secs_f64() * 1e3
                );
            }
            Some(Self { job: Some(job), inputs: seed.inputs, mode, cpu_units, n_units, out: ctx.out, batches, started })
        }

        /// The in-flight launch (`None` once joined), for the drop guard.
        pub(in super::super) fn job_mut(&mut self) -> &mut Option<BigAbJob> {
            &mut self.job
        }

        /// Units the CPU drains: `[0, cpu_units)`.
        pub(in super::super) fn cpu_units(&self) -> usize {
            self.cpu_units
        }

        /// Join the GPU after the CPU's drain. A failed admitted launch is
        /// drained by `cpu(lo, hi)`; the oracle compares and records its
        /// verdict.
        pub(in super::super) fn finish(mut self, cpu: &dyn Fn(usize, usize)) {
            let debug = crate::gpu_commit::gpu_ab_debug();
            let cpu_done_ms = self.started.elapsed().as_secs_f64() * 1e3;
            let job = self.job.take();
            let batches = std::mem::take(&mut self.batches);
            let (inputs, mode, cpu_units, n_units, out, started) =
                (self.inputs, self.mode, self.cpu_units, self.n_units, self.out, self.started);
            GPU_FIRST_UNIT.store(usize::MAX, std::sync::atomic::Ordering::Release);
            let Some(job) = job else {
                // Deferred oracle: keep the CPU's sample blocks, then let a
                // thread of its own wait for the pipelines, run the GPU half
                // and record the verdict (the split tuner waits for it).
                let mut expect = Vec::with_capacity(batches.iter().map(|b| b.count).sum::<usize>() * 1792);
                for b in &batches {
                    for i in 0..b.count {
                        // SAFETY: the CPU drain has joined; in bounds.
                        expect.extend_from_slice(unsafe {
                            core::slice::from_raw_parts(out.add((b.first + i) * 2048 + 128), 1792)
                        });
                    }
                }
                // The inputs too: the registration ends with this prove.
                let mut inp = vec![0u32; batches.iter().map(|b| b.count).sum::<usize>() * 25];
                let mut at = 0usize;
                for b in &batches {
                    inputs(b.first, b.count, &mut inp[at..at + 25 * b.count]);
                    at += 25 * b.count;
                }
                let firsts: Vec<(usize, usize)> = {
                    let mut at = 0usize;
                    batches.iter().map(|b| { let r = (b.first, at); at += 25 * b.count; r }).collect()
                };
                crate::gpu_commit::gpu_ab_oracle_pending(Some(true));
                let spawned = std::thread::Builder::new().name("gpu-ab-oracle".into()).spawn(move || {
                    let saved = |first: usize, count: usize, dst: &mut [u32]| {
                        if let Some(&(_, at)) = firsts.iter().find(|f| f.0 == first) {
                            dst.copy_from_slice(&inp[at..at + 25 * count]);
                        }
                    };
                    let r = crate::gpu_commit::bigab_launch(&saved, &batches, BigAbOut::Ring, true)
                        .ok_or_else(|| "launch".to_string())
                        .and_then(BigAbJob::wait)
                        .map(|(ms, _)| ms);
                    let pass = oracle_verdict(r, &batches, |inst, k| expect[inst * 1792 + k], debug);
                    crate::gpu_commit::gpu_ab_set_verdict(pass);
                    crate::gpu_commit::gpu_ab_oracle_pending(Some(false));
                });
                if spawned.is_err() {
                    crate::gpu_commit::gpu_ab_set_verdict(false);
                    crate::gpu_commit::gpu_ab_oracle_pending(Some(false));
                }
                return;
            };
            let r = job.wait();
            let gpu_end = r.as_ref().map_or(0.0, |&(_, e)| e);
            let r = r.map(|(ms, _)| ms);
            let wait_ms = started.elapsed().as_secs_f64() * 1e3 - cpu_done_ms;
            match (mode, r) {
                (Mode::Admitted, Ok(gpu_ms)) => {
                    if debug {
                        eprintln!(
                            "[gpu-ab] admitted: gpu {gpu_ms:.2} ms, cpu drain done {cpu_done_ms:.2} ms after launch, then waited {wait_ms:.2} ms"
                        );
                    }
                }
                (Mode::Admitted, Err(e)) => {
                    eprintln!("[gpu-ab] GPU AB arm failed ({e}); CPU drains units {cpu_units}..{n_units}");
                    crate::gpu_commit::gpu_ab_set_verdict(false);
                    cpu(cpu_units, n_units);
                }
                (Mode::OracleDeferred, _) => unreachable!("no job"),
                (Mode::Replay, r) => {
                    if debug {
                        eprintln!("[gpu-ab] replay: gpu {r:?} ms, cpu drain done {cpu_done_ms:.2} ms, waited {wait_ms:.2} ms");
                    }
                    match r {
                        Ok(_) => crate::gpu_commit::gpu_ab_note_replay_gpu_end(gpu_end),
                        Err(_) => crate::gpu_commit::gpu_ab_set_verdict(false),
                    }
                }
                (Mode::Oracle, r) => {
                    // The CPU's sample blocks, instance by instance in batch order.
                    let mut index = Vec::new();
                    for b in &batches {
                        for i in 0..b.count {
                            index.push((b.first + i) * 2048 + 128);
                        }
                    }
                    // SAFETY: the CPU drain has joined; every offset is in bounds.
                    let cpu = |inst: usize, k: usize| unsafe { *out.add(index[inst] + k) };
                    let pass = oracle_verdict(r, &batches, cpu, debug);
                    crate::gpu_commit::gpu_ab_set_verdict(pass);
                }
            }
        }
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(super) use imp::GpuArm;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl Drop for imp::GpuArm {
    /// Unwinding past an un-finished arm (a panic in the CPU drain): the
    /// GPU may still be writing into the output storage, so wait for the
    /// launch before the storage can be freed, and clear the stripe flag.
    fn drop(&mut self) {
        GPU_FIRST_UNIT.store(usize::MAX, std::sync::atomic::Ordering::Release);
        if let Some(job) = self.job_mut().take() {
            while !job.done() {
                std::thread::yield_now();
            }
        }
    }
}

/// No GPU arm off Apple silicon.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(super) struct GpuArm;

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
impl GpuArm {
    pub(super) fn start(_ctx: GpuArmCtx) -> Option<Self> {
        let _ = (UNIT_INSTANCES, UNIT_BYTES, RANKED_UNITS);
        let _ = seed_for;
        None
    }

    pub(super) fn cpu_units(&self) -> usize {
        usize::MAX
    }

    pub(super) fn finish(self, _cpu: &dyn Fn(usize, usize)) {}
}
