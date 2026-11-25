// The proof owns a reservation. Each submitted command owns the slot until
// terminal, even if its output is declined or the caller unwinds.
use super::{
    Api, BuildJob, Gpu, Id, LATCH, LatchState, NIL, Sel, ZC_R2_CS_ADMITTED, ZC_R2_CS_MODE,
    ZC_R2_POISONED, ZC_R2_TUNED, cb_gpu_interval, compile_supplemental_pipeline_on, gpu, r2cs_fast,
};
use crate::{field::F128, tmark};
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

pub(super) const OFFER: usize = 880;
const CHUNK_BYTES: usize = 256 * 1024;
const BLOCK_BYTES: usize = 2048;
const TG: usize = 128;
const Z_BYTES: usize = 512 * 1024 * 1024;
const COMPLETED: u64 = 4;
const ERROR: u64 = 5;
const FORWARD_SOURCE: &str = include_str!("r2_raw_stage.metal");
const FORWARD_ENTRY: &str = "r2_reconstruct_raw_from_z";

#[derive(Clone, Copy, PartialEq, Eq)]
enum Probe {
    Normal,
    PreflightRefuse,
    RootRefuse,
    Short,
    StageDiscard,
    R2Refuse,
    R2Discard,
    R2HeadBad,
}

/// A +1 Metal reference usable before the GPU context exists. It also drops
/// safely if the initializer exits without joining the optional build.
pub(super) struct EarlyMetal {
    api: Api,
    id: Id,
}

impl EarlyMetal {
    fn id(&self) -> Id {
        self.id
    }

    fn into_id(mut self) -> Id {
        std::mem::replace(&mut self.id, NIL)
    }
}

impl Drop for EarlyMetal {
    fn drop(&mut self) {
        if !self.id.is_null() {
            unsafe { send!(self.api, unsafe extern "C" fn(Id, Sel) -> Id, self.id, c"release") };
        }
    }
}

pub(super) fn start_early_pso(
    api: &Api,
    device: Id,
    fan: bool,
) -> Option<BuildJob<Option<EarlyMetal>>> {
    if !fan || !enabled() || !crate::is_ranked_worker_process() {
        return None;
    }
    let refuse = match std::env::var("FLOCK_R2_RAW880_EARLY_PSO").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("1") => false,
        Ok("refuse") => true,
        _ => return None,
    };
    let retained = unsafe { send!(*api, unsafe extern "C" fn(Id, Sel) -> Id, device, c"retain") };
    let device = EarlyMetal { api: *api, id: retained };
    let api = *api;
    Some(BuildJob::start(&api, fan, move || {
        let mut host_ns = warm_timing_enabled().then(|| Vec::with_capacity(6));
        let mut mark = |label| {
            if let Some(host_ns) = &mut host_ns {
                host_ns.push((label, tmark::now_ns()));
            }
        };
        mark("build_start");
        let result = if refuse {
            Err("forced initializer build refusal".into())
        } else {
            unsafe {
                compile_supplemental_pipeline_on(
                    &api, device.id(), FORWARD_SOURCE, FORWARD_ENTRY, &mut mark,
                )
            }
        };
        mark("build_end");
        if let Some(host_ns) = host_ns {
            let status = if result.is_ok() { "ready" } else { "refused" };
            receipt(format_args!("[raw880-early-pso] build={status} host_ns={host_ns:?}\n"));
        }
        match result {
            Ok(id) => Some(EarlyMetal { api, id }),
            Err(e) => {
                receipt(format_args!("[raw880-early-pso] {e}; use late builder\n"));
                None
            }
        }
    }))
}

pub(super) fn join_early_pso(job: Option<BuildJob<Option<EarlyMetal>>>) -> Option<EarlyMetal> {
    let job = job?;
    let threaded = matches!(&job, BuildJob::Running(_));
    let start = warm_timing_enabled().then(tmark::now_ns);
    let pso = job.get();
    if let Some(start) = start {
        let end = tmark::now_ns();
        receipt(format_args!(
            "[raw880-early-pso] join_start_ns={start} join_end_ns={end} threaded={threaded} ready={}\n",
            pso.is_some()
        ));
    }
    pso
}

// All handles are +1 Metal references on the process device.
struct Owned {
    gpu: &'static Gpu,
    raw: usize,
}

impl Owned {
    fn new(gpu: &'static Gpu, raw: Id) -> Self {
        Self {
            gpu,
            raw: raw as usize,
        }
    }

    fn id(&self) -> Id {
        self.raw as Id
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        unsafe { self.gpu.release(self.id()) };
    }
}

fn wait_terminal(gpu: &Gpu, cb: Id) -> u64 {
    loop {
        let status = unsafe { gpu.cb_status(cb) };
        if matches!(status, COMPLETED | ERROR) {
            return status;
        }
        let _ = unsafe { gpu.wait_cb(cb) };
    }
}

struct Warm {
    gpu: &'static Gpu,
    pso: Owned,
    a: Owned,
    b: Owned,
    busy: AtomicBool,
    probe: Probe,
}

static WARM: OnceLock<Option<Warm>> = OnceLock::new();

fn cached_wrap(gpu: &'static Gpu, z: &[F128]) -> Option<Owned> {
    let latch = LATCH.try_lock().ok()?;
    let LatchState::On(state) = &*latch else {
        return None;
    };
    let address = z.as_ptr() as usize;
    let len = core::mem::size_of_val(z);
    let &(_, _, view) = state
        .wraps
        .iter()
        .find(|&&(p, l, _)| p == address && l == len)?;
    Some(Owned::new(gpu, unsafe { gpu.retain(view) }))
}

fn selected() -> bool {
    super::super::gpu_zc_r2_enabled()
        && super::super::zc_r2_cs_enabled()
        && !ZC_R2_POISONED.load(Ordering::Relaxed)
        && ZC_R2_TUNED.load(Ordering::Relaxed) == OFFER
        && ZC_R2_CS_MODE.load(Ordering::Relaxed) == ZC_R2_CS_ADMITTED
        && r2cs_fast::fast2_selected()
}

fn private_buffer(gpu: &'static Gpu) -> Result<Owned, String> {
    let id: Id = unsafe {
        send!(
            gpu.api,
            unsafe extern "C" fn(Id, Sel, u64, u64) -> Id,
            gpu.device,
            c"newBufferWithLength:options:",
            (OFFER * CHUNK_BYTES) as u64,
            2u64 << 4 // MTLResourceStorageModePrivate
        )
    };
    if id == NIL {
        return Err("220 MiB GPU-private allocation failed".into());
    }
    Ok(Owned::new(gpu, id))
}

fn encode(warm: &Warm, z: Id, chunks: usize) -> Result<Owned, String> {
    let gpu = warm.gpu;
    unsafe {
        let cb = gpu.command_buffer()?;
        let enc = gpu.compute_encoder(cb)?;
        gpu.set_pipeline(enc, warm.pso.id());
        gpu.set_buffer(enc, z, 0, 0);
        gpu.set_buffer(enc, warm.a.id(), 0, 1);
        gpu.set_buffer(enc, warm.b.id(), 0, 2);
        let blocks = (chunks * CHUNK_BYTES / BLOCK_BYTES) as u32;
        gpu.set_bytes(enc, &blocks.to_ne_bytes(), 3);
        gpu.dispatch(enc, u64::from(blocks.div_ceil(TG as u32)), TG as u64);
        gpu.end_encoding(enc);
        Ok(Owned::new(gpu, gpu.retain(cb)))
    }
}

struct WarmTiming {
    host_ns: Vec<(&'static str, u64)>,
    gpu_ns: Option<(u64, u64)>,
}

impl WarmTiming {
    fn mark(&mut self, label: &'static str) {
        self.host_ns.push((label, tmark::now_ns()));
    }
}

fn make_warm(
    z: &[F128],
    probe: Probe,
    early_pso: Option<EarlyMetal>,
    mut timing: Option<&mut WarmTiming>,
) -> Result<Warm, String> {
    if core::mem::size_of_val(z) != Z_BYTES || !selected() {
        return Err("requires the admitted exact raw880 FAST2 warm proof".into());
    }
    let gpu = gpu()?;
    let z_view = cached_wrap(gpu, z).ok_or("warm Z wrap unavailable")?;
    let pool = unsafe { gpu.pool_push() };
    let result = (|| {
        let mut mark = |label| {
            if let Some(timing) = timing.as_deref_mut() {
                timing.mark(label);
            }
        };
        let pso = if let Some(early_pso) = early_pso {
            mark("adopt_start");
            let pso = Owned::new(gpu, early_pso.into_id());
            mark("adopt_end");
            pso
        } else {
            mark("compile_start");
            let pso = Owned::new(gpu, unsafe {
                compile_supplemental_pipeline_on(
                    &gpu.api, gpu.device, FORWARD_SOURCE, FORWARD_ENTRY, &mut mark,
                )?
            });
            mark("compile_end");
            pso
        };
        let (_, width, max_threads) = unsafe { gpu.pipeline_resources(pso.id()) };
        if width != 32 || max_threads < TG as u64 {
            return Err("full forward threadgroup unavailable".into());
        }
        mark("alloc_a_start");
        let a = private_buffer(gpu)?;
        mark("alloc_a_end");
        mark("alloc_b_start");
        let b = private_buffer(gpu)?;
        mark("alloc_b_end");
        let warm = Warm {
            gpu,
            pso,
            a,
            b,
            busy: AtomicBool::new(false),
            probe,
        };
        mark("encode_start");
        let cb = encode(&warm, z_view.id(), OFFER)?;
        mark("encode_end");
        mark("commit_start");
        unsafe { gpu.commit_async(cb.id()) };
        mark("commit_end");
        let status = wait_terminal(gpu, cb.id());
        mark("terminal");
        if status != COMPLETED {
            return Err("full forward warm command failed".into());
        }
        if let Some(timing) = timing.as_deref_mut() {
            let (start, end) = unsafe { cb_gpu_interval(gpu, cb.id()) };
            if start.is_finite() && end.is_finite() && start > 0.0 && end > start {
                timing.gpu_ns = Some(((start * 1e9).round() as u64, (end * 1e9).round() as u64));
            }
        }
        Ok(warm)
    })();
    unsafe { gpu.pool_pop(pool) };
    result
}

#[doc(hidden)]
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !super::super::root_packets_disabled()
            && std::env::var_os("FLOCK_R2_RAW880_PRIVATE_AB")
                .map_or_else(crate::is_ranked_worker_process, |value| value == "1")
    })
}

/// Warm the pipeline and both private buffers with the genuine completed
/// fixed proof's Z, while its Rust allocation remains live.
pub fn warm(z: &[F128]) {
    WARM.get_or_init(|| {
        if !enabled() {
            return None;
        }
        // This job came from the very same initialized context used by make_warm.
        // Take it once, so a declined warm check also releases the program.
        let early_pso = super::GPU.get()
            .and_then(|result| result.as_ref().ok())
            .and_then(|gpu| gpu.raw880_early_pso.lock().ok()?.take());
        let probe = match std::env::var("FLOCK_R2_RAW880_PRIVATE_AB_PROBE").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("") => Probe::Normal,
            Ok("preflight-refuse") => Probe::PreflightRefuse,
            Ok("root-refuse") => Probe::RootRefuse,
            Ok("short") => Probe::Short,
            Ok("stage-discard") => Probe::StageDiscard,
            Ok("r2-refuse") => Probe::R2Refuse,
            Ok("r2-discard") => Probe::R2Discard,
            Ok("r2-head-bad") => Probe::R2HeadBad,
            _ => {
                receipt(format_args!("[raw880] unknown probe\n"));
                return None;
            }
        };
        let mut timing = warm_timing_enabled().then(|| WarmTiming {
            host_ns: Vec::with_capacity(17),
            gpu_ns: None,
        });
        let start = (timing.is_some() || tmark::on()).then(std::time::Instant::now);
        if let Some(timing) = timing.as_mut() {
            timing.mark("hook_start");
        }
        let result = make_warm(z, probe, early_pso, timing.as_mut());
        let total_ms = start.map_or(0.0, |start| start.elapsed().as_secs_f64() * 1e3);
        if let Some(mut timing) = timing {
            timing.mark("hook_end");
            receipt(format_args!(
                "[raw880-warm-timing] total_ms={total_ms:.3} host_ns={:?} gpu_ns={:?}\n",
                timing.host_ns, timing.gpu_ns
            ));
        }
        match result {
            Ok(warm) => {
                receipt(format_args!(
                    "[raw880] warm ready bytes=461373440 ms={total_ms:.3}\n",
                ));
                Some(warm)
            }
            Err(e) => {
                receipt(format_args!("[raw880] warm refused: {e}\n"));
                None
            }
        }
    });
}

fn warm_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FLOCK_R2_RAW880_WARM_TIMING").is_some_and(|value| value == "1"))
}

fn receipt(args: std::fmt::Arguments<'_>) {
    if !warm_timing_enabled() && !tmark::on() {
        return;
    }
    use std::io::Write;
    let _ = std::io::stderr().lock().write_fmt(args);
}

struct Slot {
    warm: &'static Warm,
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.warm.busy.store(false, Ordering::Release);
    }
}

/// Exclusive proof-local slot plus the preflight Z allocation identity.
/// The live typed Z borrow is established by `submit` before any GPU read.
#[doc(hidden)]
pub struct Reservation {
    slot: Slot,
    addr: usize,
    len: usize,
}

pub fn try_reserve(z: &[F128]) -> Option<Reservation> {
    let warm = WARM.get()?.as_ref()?;
    let len = core::mem::size_of_val(z);
    if len != Z_BYTES
        || !selected()
        || warm.probe == Probe::PreflightRefuse
        || cached_wrap(warm.gpu, z).is_none()
    {
        tmark::mark("raw880 preflight declined");
        return None;
    }
    if warm.busy.swap(true, Ordering::Acquire) {
        tmark::mark("raw880 preflight busy");
        return None;
    }
    tmark::mark("raw880 preflight reserved");
    Some(Reservation {
        slot: Slot { warm },
        addr: z.as_ptr() as usize,
        len,
    })
}

impl Reservation {
    /// Called immediately after the existing C hook at the actual root.
    pub fn submit(self, z: &[F128]) -> Option<StageJob<'_>> {
        tmark::mark("raw880 root stage entry");
        let warm = self.slot.warm;
        if self.addr != z.as_ptr() as usize
            || self.len != core::mem::size_of_val(z)
            || warm.probe == Probe::RootRefuse
        {
            tmark::mark("raw880 root stage refused");
            return None;
        }
        let Some(z_view) = cached_wrap(warm.gpu, z) else {
            tmark::mark("raw880 root Z view missed");
            return None;
        };
        let chunks = if warm.probe == Probe::Short {
            OFFER - 1
        } else {
            OFFER
        };
        let gpu = warm.gpu;
        let pool = unsafe { gpu.pool_push() };
        let encoded = encode(warm, z_view.id(), chunks);
        unsafe { gpu.pool_pop(pool) };
        let Ok(cb) = encoded else {
            tmark::mark("raw880 root encode refused");
            return None;
        };
        tmark::mark("raw880 root commit begin");
        unsafe { gpu.commit_async(cb.id()) };
        let job = StageJob {
            flight: Some(StageFlight {
                cb,
                _z_view: z_view,
                slot: self.slot,
                chunks,
            }),
            _z_owner: z,
        };
        tmark::mark("raw880 root commit return");
        if chunks != OFFER {
            tmark::mark("raw880 root short stage");
        }
        Some(job)
    }
}

struct StageFlight {
    cb: Owned,
    _z_view: Owned,
    slot: Slot,
    chunks: usize,
}

/// Owns the command and a read-only borrow of the current Z until terminal.
#[doc(hidden)]
pub struct StageJob<'z> {
    flight: Option<StageFlight>,
    _z_owner: &'z [F128],
}

fn mark_interval(cb: &Owned, start: &'static str, end: &'static str) {
    if tmark::on() {
        let (s, e) = unsafe { cb_gpu_interval(cb.gpu, cb.id()) };
        if s.is_finite() && e.is_finite() && s > 0.0 && e > s {
            tmark::mark_gpu(start, end, s, e);
        }
    }
}

impl StageJob<'_> {
    /// One nonblocking eligibility check at R2. A pending stage stays with
    /// the proof owner until the late safe retirement point.
    pub(super) fn ready(&self, gpu: &Gpu) -> bool {
        let Some(flight) = self.flight.as_ref() else {
            return false;
        };
        if !std::ptr::eq(gpu, flight.slot.warm.gpu) {
            tmark::mark("raw880 R2 stage device mismatch");
            return false;
        }
        if flight.chunks != OFFER {
            tmark::mark("raw880 R2 short stage refused");
            return false;
        }
        match unsafe { gpu.cb_status(flight.cb.id()) } {
            COMPLETED => match flight.slot.warm.probe {
                Probe::StageDiscard => {
                    tmark::mark("raw880 R2 completed stage discarded");
                    false
                }
                Probe::R2Refuse => {
                    tmark::mark("raw880 R2 test late admission refused");
                    false
                }
                _ => true,
            },
            ERROR => {
                tmark::mark("raw880 R2 stage failed");
                false
            }
            _ => {
                tmark::mark("raw880 R2 stage pending");
                false
            }
        }
    }

    // Only the R2 launcher can transfer the slot. It calls ready first and
    // supplies the proven full output views before taking this receipt.
    pub(super) fn take_ready(&mut self) -> Option<Ready> {
        let StageFlight {
            cb, _z_view, slot, ..
        } = self.flight.take()?;
        mark_interval(&cb, "raw880 stage gpu start", "raw880 stage gpu end");
        tmark::mark("raw880 stage terminal success");
        Some(Ready(slot))
    }
}

impl Drop for StageJob<'_> {
    fn drop(&mut self) {
        if let Some(flight) = &self.flight {
            let cb = &flight.cb;
            let gpu = cb.gpu;
            let before = unsafe { gpu.cb_status(cb.id()) };
            if !matches!(before, COMPLETED | ERROR) {
                tmark::mark("raw880 stage retire wait");
            }
            if wait_terminal(gpu, cb.id()) == COMPLETED {
                mark_interval(cb, "raw880 stage gpu start", "raw880 stage gpu end");
                tmark::mark("raw880 stage retired unused");
            } else {
                tmark::mark("raw880 stage retired error");
            }
        }
    }
}

pub(super) struct Ready(Slot);

impl Ready {
    pub(super) fn buffers(&self) -> (Id, Id) {
        let w = self.0.warm;
        (w.a.id(), w.b.id())
    }
}

// Only launches in an opt-in process take the shared R2 lease. The caller
// acquires it under the ZC_R2_STATE mutex before any scratch writer runs;
// lifetime covers terminal and host output collection, even for raw jobs.
static STATE_BUSY: AtomicBool = AtomicBool::new(false);

pub(super) fn state_busy() -> bool {
    STATE_BUSY.load(Ordering::Acquire)
}

pub(super) struct StateLease;

impl StateLease {
    pub(super) fn try_take() -> Result<Option<Self>, ()> {
        if !enabled() {
            return Ok(None);
        }
        if STATE_BUSY.swap(true, Ordering::Acquire) {
            return Err(());
        }
        Ok(Some(Self))
    }

    pub(super) fn bind(self, gpu: &'static Gpu, cb: Id, private: Option<Ready>) -> R2Flight {
        R2Flight {
            // Take the command's +1 reference returned by zc_r2_submit.
            cb: Owned::new(gpu, cb),
            oracle: None,
            private,
            _state: self,
        }
    }
}

impl Drop for StateLease {
    fn drop(&mut self) {
        STATE_BUSY.store(false, Ordering::Release);
    }
}

pub(super) struct R2Flight {
    cb: Owned,
    oracle: Option<Owned>,
    private: Option<Ready>,
    _state: StateLease,
}

impl R2Flight {
    pub(super) fn set_oracle(&mut self, cb: Id) {
        if cb != NIL {
            self.oracle = Some(Owned::new(self.cb.gpu, cb));
        }
    }

    pub(super) fn discard(&self) -> bool {
        self.private
            .as_ref()
            .is_some_and(|slot| slot.0.warm.probe == Probe::R2Discard)
    }

    pub(super) fn head_bad(&self) -> bool {
        self.private
            .as_ref()
            .is_some_and(|slot| slot.0.warm.probe == Probe::R2HeadBad)
    }
}

impl Drop for R2Flight {
    fn drop(&mut self) {
        let gpu = self.cb.gpu;
        let terminal = wait_terminal(gpu, self.cb.id());
        if let Some(cb) = &self.oracle {
            wait_terminal(gpu, cb.id());
        }
        if self.private.is_some() {
            if terminal == COMPLETED {
                mark_interval(&self.cb, "raw880 R2 gpu start", "raw880 R2 gpu end");
                tmark::mark("raw880 R2 terminal success");
            } else {
                tmark::mark("raw880 R2 terminal error");
            }
        }
    }
}
