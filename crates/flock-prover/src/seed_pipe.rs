use std::ffi::OsStr;
use std::io::{self, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use flock_core::pcs::Commitment;
use flock_core::proof::{R1csClaim, R1csProofLigerito};
use rayon::prelude::*;

use crate::r1cs_hashes::blake3::Compression;

pub type ProveOut = (R1csProofLigerito, Commitment, R1csClaim);

pub const BENCH_DOMAIN: &[u8] = b"flock-bench-v0";

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
const DRAWS_PER_BLOCK: usize = 25;
const ZERO_COMPRESSION: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

#[inline(always)]
fn mix(mut z: u64) -> u32 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

pub fn generate_compressions_par(log2_size: u32, seed: u64) -> Vec<Compression> {
    let mut out = zero_blocks(1usize << log2_size);
    fill_compressions_par(&mut out, log2_size, seed);
    out
}

fn zero_blocks(count: usize) -> Vec<Compression> {
    vec![ZERO_COMPRESSION; count]
}

pub(crate) fn prefaulted_blocks(count: usize) -> Vec<Compression> {
    let mut v = zero_blocks(count);
    let bytes = std::mem::size_of_val(v.as_slice());
    let base = v.as_mut_ptr().cast::<u8>();
    let mut offset = 0usize;
    while offset < bytes {
        unsafe { base.add(offset).write_volatile(0) };
        offset += 4096;
    }
    v
}

#[inline(always)]
pub(crate) fn generator_init(log2_size: u32, seed: u64) -> u64 {
    seed ^ u64::from(log2_size).rotate_left(29)
}

#[inline(always)]
pub(crate) fn gen_block(init: u64, block: usize) -> Compression {
    gen_block_with(init, block, !gen_block_ilp_killed())
}

pub(crate) fn gen_block_ilp_killed() -> bool {
    std::env::var("FLOCK_NO_GEN_BLOCK_SIMD").is_ok_and(|v| v == "1")
}

#[inline(always)]
pub(crate) fn gen_block_with(init: u64, block: usize, ilp: bool) -> Compression {
    if ilp {
        gen_block_ilp(init, block)
    } else {
        gen_block_scalar(init, block)
    }
}

#[inline(always)]
pub(crate) fn gen_block_scalar(init: u64, block: usize) -> Compression {
    let mut s = init.wrapping_add(((DRAWS_PER_BLOCK * block) as u64).wrapping_mul(GOLDEN));
    let mut cv = [0u32; 8];
    for word in cv.iter_mut() {
        s = s.wrapping_add(GOLDEN);
        *word = mix(s);
    }
    let mut message = [0u32; 16];
    for word in message.iter_mut() {
        s = s.wrapping_add(GOLDEN);
        *word = mix(s);
    }
    s = s.wrapping_add(GOLDEN);
    (cv, message, u64::from(mix(s)), 64, 11)
}

#[inline(always)]
pub(crate) fn gen_block_ilp(init: u64, block: usize) -> Compression {
    let base = init.wrapping_add(((DRAWS_PER_BLOCK * block) as u64).wrapping_mul(GOLDEN));
    #[inline(always)]
    fn draw(base: u64, d: usize) -> u32 {
        mix(base.wrapping_add(((d + 1) as u64).wrapping_mul(GOLDEN)))
    }
    let mut out = [0u32; DRAWS_PER_BLOCK];
    let mut d = 0usize;
    while d + 4 <= DRAWS_PER_BLOCK {
        let (o0, o1, o2, o3) = (
            draw(base, d),
            draw(base, d + 1),
            draw(base, d + 2),
            draw(base, d + 3),
        );
        out[d] = o0;
        out[d + 1] = o1;
        out[d + 2] = o2;
        out[d + 3] = o3;
        d += 4;
    }
    while d < DRAWS_PER_BLOCK {
        out[d] = draw(base, d);
        d += 1;
    }
    let cv: [u32; 8] = out[0..8].try_into().expect("draw layout");
    let message: [u32; 16] = out[8..24].try_into().expect("draw layout");
    (cv, message, u64::from(out[24]), 64, 11)
}

#[repr(align(16))]
pub(crate) struct CompressionQuadSoa {
    pub(crate) cv: [[u32; 4]; 8],
    pub(crate) message: [[u32; 4]; 16],
    pub(crate) counter_lo: [u32; 4],
    pub(crate) counter_hi: [u32; 4],
    pub(crate) block_len: [u32; 4],
    pub(crate) flags: [u32; 4],
}

#[inline(always)]
pub(crate) fn gen_quad_soa(init: u64, first_block: usize) -> CompressionQuadSoa {
    let first_state =
        init.wrapping_add(((DRAWS_PER_BLOCK * first_block) as u64).wrapping_mul(GOLDEN));
    let block_stride = (DRAWS_PER_BLOCK as u64).wrapping_mul(GOLDEN);
    let mut states = [
        first_state,
        first_state.wrapping_add(block_stride),
        first_state.wrapping_add(block_stride.wrapping_mul(2)),
        first_state.wrapping_add(block_stride.wrapping_mul(3)),
    ];

    #[inline(always)]
    fn draw_row(states: &mut [u64; 4]) -> [u32; 4] {
        states[0] = states[0].wrapping_add(GOLDEN);
        states[1] = states[1].wrapping_add(GOLDEN);
        states[2] = states[2].wrapping_add(GOLDEN);
        states[3] = states[3].wrapping_add(GOLDEN);
        [
            mix(states[0]),
            mix(states[1]),
            mix(states[2]),
            mix(states[3]),
        ]
    }

    let cv = std::array::from_fn(|_| draw_row(&mut states));
    let message = std::array::from_fn(|_| draw_row(&mut states));
    let counter_lo = draw_row(&mut states);
    CompressionQuadSoa {
        cv,
        message,
        counter_lo,
        counter_hi: [0; 4],
        block_len: [64; 4],
        flags: [11; 4],
    }
}

fn generated_quads_match_blocks(init: u64, blocks: &[Compression]) -> bool {
    if !blocks.len().is_multiple_of(4) {
        return false;
    }
    (0..blocks.len() / 4).into_par_iter().all(|quad_index| {
        let first = 4 * quad_index;
        let quad = gen_quad_soa(init, first);
        (0..4).all(|lane| {
            let (cv, message, counter, block_len, flags) = &blocks[first + lane];
            quad.cv
                .iter()
                .zip(cv)
                .all(|(words, expected)| words[lane] == *expected)
                && quad
                    .message
                    .iter()
                    .zip(message)
                    .all(|(words, expected)| words[lane] == *expected)
                && (u64::from(quad.counter_lo[lane]) | (u64::from(quad.counter_hi[lane]) << 32))
                    == *counter
                && quad.block_len[lane] == *block_len
                && quad.flags[lane] == *flags
        })
    })
}

fn fill_compressions_par(out: &mut [Compression], log2_size: u32, seed: u64) {
    fill_compressions_from_init(out, generator_init(log2_size, seed));
}

fn fill_compressions_from_init(out: &mut [Compression], init: u64) {
    let ilp = !gen_block_ilp_killed();
    out.par_chunks_mut(4096)
        .enumerate()
        .for_each(|(chunk_index, dst)| {
            let base = chunk_index * 4096;
            for (offset, slot) in dst.iter_mut().enumerate() {
                *slot = gen_block_with(init, base + offset, ilp);
            }
        });
}

static SPEC_BASE: AtomicUsize = AtomicUsize::new(0);
static SPEC_LEN: AtomicUsize = AtomicUsize::new(0);
static SPEC_INIT: AtomicU64 = AtomicU64::new(0);

fn lazy_blocks_killed() -> bool {
    std::env::var("FLOCK_NO_SPEC_LAZY_BLOCKS").is_ok_and(|v| v == "1")
}

#[inline]
pub(crate) fn spec_gen_init(blocks: &[Compression]) -> Option<u64> {
    let base = SPEC_BASE.load(Ordering::Acquire);
    if base == 0
        || base != blocks.as_ptr() as usize
        || SPEC_LEN.load(Ordering::Relaxed) != blocks.len()
    {
        return None;
    }
    Some(SPEC_INIT.load(Ordering::Relaxed))
}

pub(crate) fn materialize_spec_blocks(blocks: &[Compression]) -> Option<Vec<Compression>> {
    let init = spec_gen_init(blocks)?;
    let mut generated = zero_blocks(blocks.len());
    fill_compressions_from_init(&mut generated, init);
    Some(generated)
}

fn blocks_eq(a: &[Compression], b: &[Compression]) -> bool {
    const _: () = assert!(std::mem::size_of::<Compression>() == 112);
    if a.len() != b.len() {
        return false;
    }
    a.par_chunks(8192)
        .zip(b.par_chunks(8192))
        .all(|(x, y)| bytes_of(x) == bytes_of(y))
}

fn blocks_eq_serial(a: &[Compression], b: &[Compression]) -> bool {
    const _: () = assert!(std::mem::size_of::<Compression>() == 112);
    a.len() == b.len() && bytes_of(a) == bytes_of(b)
}

fn bytes_of(v: &[Compression]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

#[derive(Default)]
struct State {
    blocks: Option<Arc<Vec<Compression>>>,
    endpoints: Option<(Compression, Compression)>,
    result: Option<ProveOut>,
    dead: bool,
    seed_at: Option<std::time::Instant>,
    blocks_at: Option<std::time::Instant>,
}

struct Pipe {
    state: Mutex<State>,
    signal: Condvar,
}

const WARMUP_SEED: u64 = 0x00C0_FFEE_BEEF_D15C;

static GENERATOR_VERIFIED: AtomicBool = AtomicBool::new(false);

static PIPE: OnceLock<Pipe> = OnceLock::new();
static ARMED: AtomicBool = AtomicBool::new(false);
static SHADOW_QOS: AtomicBool = AtomicBool::new(false);

struct ShadowQosGuard(bool);

impl Drop for ShadowQosGuard {
    fn drop(&mut self) {
        if self.0 {
            flock_core::set_calling_thread_prover_qos();
        }
    }
}

fn shared() -> &'static Pipe {
    PIPE.get_or_init(|| Pipe {
        state: Mutex::new(State::default()),
        signal: Condvar::new(),
    })
}

fn mark_dead() {
    flock_core::cpu_keepalive::pool_spin_release();
    let mut state = shared().state.lock().unwrap_or_else(|e| e.into_inner());
    state.dead = true;
    shared().signal.notify_all();
}

unsafe extern "C" {
    fn dup(fd: i32) -> i32;
    fn dup2(from: i32, to: i32) -> i32;
    #[link_name = "pipe"]
    fn sys_pipe(fds: *mut i32) -> i32;
    fn close(fd: i32) -> i32;
    fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn poll(fds: *mut PollFd, nfds: u32, timeout: i32) -> i32;
}

#[repr(C)]
struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

const HOT_WAIT_BUDGET: std::time::Duration = std::time::Duration::from_secs(2);

fn hot_wait_readable(fd: i32) {
    if std::env::var_os("FLOCK_NO_HOT_SEED_WAIT").is_some() {
        return;
    }
    let deadline = std::time::Instant::now() + HOT_WAIT_BUDGET;
    let mut spins = 0u32;
    loop {
        let mut p = PollFd {
            fd,
            events: 0x0001,
            revents: 0,
        };
        let rc = unsafe { poll(&mut p, 1, 0) };
        if rc != 0 {
            return;
        }
        spins = spins.wrapping_add(1);
        if spins % 1024 == 0 && std::time::Instant::now() >= deadline {
            return;
        }
        std::hint::spin_loop();
    }
}

fn read_line_fd(fd: i32) -> Option<Vec<u8>> {
    let mut line = Vec::with_capacity(64);
    let mut chunk = [0u8; 64];
    loop {
        let n = unsafe { read(fd, chunk.as_mut_ptr(), chunk.len()) };
        match n {
            n if n > 0 => {
                line.extend_from_slice(&chunk[..n as usize]);
                if line.contains(&b'\n') || line.len() >= 256 {
                    return Some(line);
                }
            }
            0 => return (!line.is_empty()).then_some(line),
            _ => return None,
        }
    }
}

fn write_all_fd(fd: i32, mut buf: &[u8]) -> bool {
    while !buf.is_empty() {
        let n = unsafe { write(fd, buf.as_ptr(), buf.len()) };
        if n <= 0 {
            return false;
        }
        buf = &buf[n as usize..];
    }
    true
}

struct SeedForwardFallback<'a> {
    writer: i32,
    bytes: &'a [u8],
    pending: bool,
    completed: bool,
}

impl<'a> SeedForwardFallback<'a> {
    fn new(writer: i32, bytes: &'a [u8]) -> Self {
        Self {
            writer,
            bytes,
            pending: true,
            completed: false,
        }
    }

    fn forward(&mut self) -> bool {
        if !self.pending {
            return true;
        }
        self.pending = false;
        if write_all_fd(self.writer, self.bytes) {
            true
        } else {
            unsafe { close(self.writer) };
            false
        }
    }

    fn complete(&mut self) {
        self.completed = true;
    }
}

impl Drop for SeedForwardFallback<'_> {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        let _ = self.forward();
        mark_dead();
    }
}

pub(crate) fn is_ranked_worker() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let mut args = std::env::args_os();
        let Some(exe) = args.next() else {
            return false;
        };
        if args.len() != 3 {
            return false;
        }
        std::path::Path::new(&exe)
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("flock-benchmark-worker"))
    })
}

fn speculative_publish_enabled_for(ranked: bool, kill: Option<&OsStr>) -> bool {
    ranked && kill != Some(OsStr::new("1"))
}

fn speculative_publish_enabled() -> bool {
    speculative_publish_enabled_for(
        is_ranked_worker(),
        std::env::var_os("FLOCK_NO_SPEC_PUBLISH").as_deref(),
    )
}

fn ranked_proof_path() -> Option<PathBuf> {
    static CACHED: OnceLock<Option<PathBuf>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            if !is_ranked_worker() {
                return None;
            }
            let path = PathBuf::from(std::env::args_os().nth(3)?);
            (!path.as_os_str().is_empty()).then_some(path)
        })
        .clone()
}

fn speculative_temporary_path(proof_path: &Path) -> PathBuf {
    let mut temporary = proof_path.as_os_str().to_os_string();
    temporary.push(".spec.tmp");
    temporary.into()
}

fn publish_bytes_atomically(proof_path: &Path, bytes: &[u8]) -> io::Result<()> {
    if proof_path.as_os_str().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "empty proof path",
        ));
    }
    let temporary = speculative_temporary_path(proof_path);
    let result = (|| {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.flush()?;
        drop(file);
        std::fs::rename(&temporary, proof_path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn preopen_publish_enabled() -> bool {
    preopen_publish_enabled_for(std::env::var_os("FLOCK_NO_PREOPEN_PUBLISH").as_deref())
}

fn preopen_publish_enabled_for(kill: Option<&OsStr>) -> bool {
    kill != Some(OsStr::new("1"))
}

fn late_close_enabled() -> bool {
    std::env::var_os("FLOCK_NO_LATE_CLOSE").as_deref() != Some(OsStr::new("1"))
}

const PREOPEN_WARM_BYTES: usize = 458_752;

struct PreopenedPublish {
    file: Option<std::fs::File>,
    temporary: PathBuf,
}

impl Drop for PreopenedPublish {
    fn drop(&mut self) {
        self.file.take();
        let _ = std::fs::remove_file(&self.temporary);
    }
}

fn preopen_publish_file(proof_path: &Path) -> Option<PreopenedPublish> {
    if !preopen_publish_enabled() {
        return None;
    }
    let temporary = speculative_temporary_path(proof_path);
    let attempt = || -> io::Result<PreopenedPublish> {
        let mut file = std::fs::File::create(&temporary)?;
        let pattern = [0u8; 8192];
        let mut remaining = PREOPEN_WARM_BYTES;
        while remaining > 0 {
            let n = remaining.min(pattern.len());
            file.write_all(&pattern[..n])?;
            remaining -= n;
        }
        file.set_len(0)?;
        file.seek(io::SeekFrom::Start(0))?;
        Ok(PreopenedPublish {
            file: Some(file),
            temporary: temporary.clone(),
        })
    };
    match attempt() {
        Ok(pre) => Some(pre),
        Err(_) => {
            let _ = std::fs::remove_file(&temporary);
            None
        }
    }
}

fn publish_bytes_through_preopened(
    proof_path: &Path,
    bytes: &[u8],
    preopened: PreopenedPublish,
) -> io::Result<()> {
    publish_through_preopened_with(proof_path, preopened, |file| file.write_all(bytes))
}

fn publish_through_preopened_with<F>(
    proof_path: &Path,
    mut preopened: PreopenedPublish,
    fill: F,
) -> io::Result<()>
where
    F: FnOnce(&mut std::fs::File) -> io::Result<()>,
{
    let file = preopened
        .file
        .take()
        .ok_or_else(|| io::Error::other("pre-opened publish fd already consumed"));
    let result = match file {
        Ok(mut file) => fill(&mut file).and_then(|()| {
            flock_core::tmark::mark("written");
            if !late_close_enabled() {
                drop(file);
                return std::fs::rename(&preopened.temporary, proof_path);
            }
            let renamed = std::fs::rename(&preopened.temporary, proof_path);
            flock_core::tmark::mark("renamed");
            drop(file);
            renamed
        }),
        Err(error) => Err(error),
    };
    if result.is_err() {
        let _ = std::fs::remove_file(&preopened.temporary);
    }
    result
}

fn publish_speculative_result(
    out: &ProveOut,
    preopened: Option<PreopenedPublish>,
) -> io::Result<bool> {
    if !speculative_publish_enabled() {
        return Ok(false);
    }
    let proof_path = ranked_proof_path()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing ranked proof path"))?;
    let mut preopened = preopened;
    if preopened.is_some() {
        if let Some(gather) = crate::proof_io::r1cs_parts_gather(&out.1, &out.0) {
            flock_core::tmark::mark("encoded");
            let pre = preopened.take().expect("checked above");
            if publish_through_preopened_with(&proof_path, pre, |file| gather.write_all_to(file))
                .is_ok()
            {
                return Ok(true);
            }
        }
    }
    let bytes = crate::proof_io::r1cs_parts_to_bytes(&out.1, &out.0);
    flock_core::tmark::mark("encoded");
    if let Some(pre) = preopened {
        if publish_bytes_through_preopened(&proof_path, &bytes, pre).is_ok() {
            return Ok(true);
        }
    }
    publish_bytes_atomically(&proof_path, &bytes)?;
    Ok(true)
}

fn catch_speculative_publish<F>(publish: F) -> bool
where
    F: FnOnce() -> io::Result<bool>,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(publish))
        .ok()
        .and_then(|result| result.ok())
        .unwrap_or(false)
}

pub(crate) fn verify_generator_at_warmup(log2_size: u32, warmup_blocks: &[Compression]) {
    if std::env::var_os("FLOCK_NO_WARMUP_GENCHECK").is_some() || !is_ranked_worker() {
        return;
    }
    if warmup_blocks.len() != 1usize << log2_size {
        return;
    }
    let ours = generate_compressions_par(log2_size, WARMUP_SEED);
    let init = generator_init(log2_size, WARMUP_SEED);
    if blocks_eq_serial(&ours, warmup_blocks) && generated_quads_match_blocks(init, warmup_blocks) {
        GENERATOR_VERIFIED.store(true, Ordering::SeqCst);
    }
}

pub(crate) fn arm(log2_size: u32, setup_addr: usize, run: fn(usize, &[Compression]) -> ProveOut) {
    if log2_size < 14 || std::env::var_os("FLOCK_NO_SEED_PIPE").is_some() || !is_ranked_worker() {
        return;
    }
    if ARMED.swap(true, Ordering::SeqCst) {
        return;
    }

    let (real_stdin, writer) = unsafe {
        let real = dup(0);
        if real < 0 {
            ARMED.store(false, Ordering::SeqCst);
            return;
        }
        let mut fds = [0i32; 2];
        if sys_pipe(fds.as_mut_ptr()) != 0 {
            close(real);
            ARMED.store(false, Ordering::SeqCst);
            return;
        }
        if dup2(fds[0], 0) < 0 {
            close(real);
            close(fds[0]);
            close(fds[1]);
            ARMED.store(false, Ordering::SeqCst);
            return;
        }
        close(fds[0]);
        (real, fds[1])
    };

    let _ = shared();
    let scratch = prefaulted_blocks(1usize << log2_size);
    let warm = Arc::new((Mutex::new(false), Condvar::new()));
    let warm_tx = Arc::clone(&warm);
    let spawned = std::thread::Builder::new()
        .name("flock-seed-pipe".into())
        .stack_size(32 << 20)
        .spawn(move || {
            speculative_main(
                real_stdin, writer, log2_size, setup_addr, run, scratch, warm_tx,
            )
        });

    if spawned.is_err() {
        unsafe {
            dup2(real_stdin, 0);
            close(real_stdin);
            close(writer);
        }
        ARMED.store(false, Ordering::SeqCst);
        return;
    }

    let (lock, cv) = &*warm;
    let mut done = lock.lock().unwrap_or_else(|e| e.into_inner());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
    while !*done {
        let now = std::time::Instant::now();
        if now >= deadline {
            break;
        }
        done = cv
            .wait_timeout(done, deadline - now)
            .unwrap_or_else(|e| e.into_inner())
            .0;
    }

    if std::env::var_os("FLOCK_SHADOW_QOS").is_some() {
        SHADOW_QOS.store(true, Ordering::SeqCst);
        flock_core::set_calling_thread_shadow_qos();
    }
}

#[allow(clippy::too_many_arguments)]
fn speculative_main(
    real_stdin: i32,
    writer: i32,
    log2_size: u32,
    setup_addr: usize,
    run: fn(usize, &[Compression]) -> ProveOut,
    scratch: Vec<Compression>,
    warm: Arc<(Mutex<bool>, Condvar)>,
) {
    flock_core::set_calling_thread_prover_qos();
    let mut scratch = scratch;

    let n_total = 1usize << log2_size;
    if scratch.len() == n_total && std::env::var_os("FLOCK_NO_SPEC_WARMUP").is_none() {
        let warm_passes = if log2_size >= 16 { 2usize } else { 1usize };
        let lazy_warm = GENERATOR_VERIFIED.load(Ordering::SeqCst) && !lazy_blocks_killed();
        for _ in 0..warm_passes {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if lazy_warm {
                    let init = generator_init(log2_size, WARMUP_SEED);
                    let base = scratch.as_ptr() as usize;
                    SPEC_INIT.store(init, Ordering::Relaxed);
                    SPEC_LEN.store(scratch.len(), Ordering::Relaxed);
                    SPEC_BASE.store(base, Ordering::Release);
                    let _ = std::hint::black_box(run(setup_addr, &scratch));
                    crate::proof_io::clear_stashed_r1cs_prefix();
                } else {
                    fill_compressions_par(&mut scratch, log2_size, WARMUP_SEED);
                    let _ = std::hint::black_box(run(setup_addr, &scratch));
                    crate::proof_io::clear_stashed_r1cs_prefix();
                }
            }));
        }
    }

    let preopened = speculative_publish_enabled()
        .then(ranked_proof_path)
        .flatten()
        .and_then(|proof_path| preopen_publish_file(&proof_path));

    {
        let (lock, cv) = &*warm;
        *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
        cv.notify_all();
    }
    drop(warm);

    hot_wait_readable(real_stdin);
    let line = read_line_fd(real_stdin);
    flock_core::tmark::mark("seed read");

    let defer_join =
        std::env::var_os("FLOCK_NO_KEEPALIVE_DEFER").as_deref() != Some(std::ffi::OsStr::new("1"));
    if defer_join {
        flock_core::cpu_keepalive::keepalive_signal();
    } else {
        flock_core::cpu_keepalive::keepalive_stop();
    }

    let Some(bytes) = line.as_deref() else {
        unsafe { close(writer) };
        mark_dead();
        return;
    };
    let mut seed_forward = SeedForwardFallback::new(writer, bytes);

    let defer_forward = speculative_publish_enabled();
    if !defer_forward && !seed_forward.forward() {
        return;
    }
    if !defer_forward && defer_join {
        flock_core::cpu_keepalive::keepalive_join();
    }

    let parsed = std::str::from_utf8(bytes)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok());
    let Some(seed) = parsed else {
        if defer_forward {
            let _ = seed_forward.forward();
        }
        mark_dead();
        seed_forward.complete();
        return;
    };

    let seed_at = std::time::Instant::now();
    if defer_forward && defer_join {
        flock_core::cpu_keepalive::keepalive_join();
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut buf = std::mem::take(&mut scratch);
        let n_total = 1usize << log2_size;
        let lazy = buf.len() == n_total
            && GENERATOR_VERIFIED.load(Ordering::SeqCst)
            && !lazy_blocks_killed();
        let (blocks, endpoints) = if lazy {
            let init = generator_init(log2_size, seed);
            let endpoints = (gen_block(init, 0), gen_block(init, n_total - 1));
            let base = buf.as_ptr() as usize;
            SPEC_INIT.store(init, Ordering::Relaxed);
            SPEC_LEN.store(buf.len(), Ordering::Relaxed);
            SPEC_BASE.store(base, Ordering::Release);
            (Arc::new(buf), Some(endpoints))
        } else if buf.len() == n_total {
            fill_compressions_par(&mut buf, log2_size, seed);
            (Arc::new(buf), None)
        } else {
            (Arc::new(generate_compressions_par(log2_size, seed)), None)
        };
        {
            let mut state = shared().state.lock().unwrap_or_else(|e| e.into_inner());
            state.seed_at = Some(seed_at);
            state.blocks_at = Some(std::time::Instant::now());
            state.endpoints = endpoints;
            state.blocks = Some(Arc::clone(&blocks));
            shared().signal.notify_all();
        }
        let out = run(setup_addr, &blocks);
        flock_core::tmark::mark("prove done");
        out
    }));

    match outcome {
        Ok(out) => {
            if defer_forward {
                let published =
                    catch_speculative_publish(|| publish_speculative_result(&out, preopened));
                if published {
                    flock_core::tmark::mark("published");
                    flock_core::tmark::dump();
                    unsafe { close(writer) };
                    seed_forward.complete();
                    return;
                }
            }
            {
                let mut state = shared().state.lock().unwrap_or_else(|e| e.into_inner());
                state.result = Some(out);
                shared().signal.notify_all();
            }
            flock_core::tmark::mark("published");
            flock_core::tmark::dump();
            if defer_forward && !seed_forward.forward() {
                return;
            }
            seed_forward.complete();
        }
        Err(_) => {
            if defer_forward {
                let _ = seed_forward.forward();
            }
            mark_dead();
            seed_forward.complete();
        }
    }
}

pub(crate) fn try_adopt(blocks: &[Compression]) -> Option<ProveOut> {
    if !ARMED.load(Ordering::SeqCst) {
        return None;
    }
    let shadow = ShadowQosGuard(SHADOW_QOS.swap(false, Ordering::SeqCst));
    let shared = shared();
    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());

    while state.blocks.is_none() && !state.dead {
        state = shared.signal.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    if state.dead {
        return None;
    }
    let speculative = Arc::clone(state.blocks.as_ref()?);
    let endpoints = state.endpoints;
    let seed_at = state.seed_at;
    let blocks_at = state.blocks_at;
    drop(state);

    let fast_gate = GENERATOR_VERIFIED.load(Ordering::SeqCst);
    let matched = if let Some((first, last)) = &endpoints {
        speculative.len() == blocks.len()
            && blocks.first() == Some(first)
            && blocks.last() == Some(last)
    } else if fast_gate {
        speculative.len() == blocks.len()
            && speculative.first() == blocks.first()
            && speculative.last() == blocks.last()
    } else if shadow.0 {
        blocks_eq_serial(&speculative, blocks)
    } else {
        blocks_eq(&speculative, blocks)
    };

    drop(shadow);

    if std::env::var_os("FLOCK_SEED_PIPE_DEBUG").is_some() {
        if let (Some(seed_at), Some(blocks_at)) = (seed_at, blocks_at) {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            eprintln!(
                "[seed-pipe] par-gen {:.3} ms, head start {:.3} ms, blocks matched={matched}, gate={}",
                ms(blocks_at - seed_at),
                ms(seed_at.elapsed()),
                if fast_gate { "fast" } else { "full" },
            );
        }
    }

    let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    while state.result.is_none() && !state.dead {
        state = shared.signal.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    let result = state.result.take();
    if state.dead || !matched {
        return None;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    static SPEC_IDENTITY_TEST_LOCK: Mutex<()> = Mutex::new(());
    static PUBLICATION_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    fn publication_test_dir(label: &str) -> PathBuf {
        let id = PUBLICATION_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "flock-seed-pipe-{label}-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create publication test directory");
        path
    }

    struct SpecIdentityReset;

    impl Drop for SpecIdentityReset {
        fn drop(&mut self) {
            SPEC_BASE.store(0, Ordering::Release);
            SPEC_LEN.store(0, Ordering::Relaxed);
            SPEC_INIT.store(0, Ordering::Relaxed);
        }
    }

    fn reference(log2_size: u32, seed: u64) -> Vec<Compression> {
        struct Rng(u64);
        impl Rng {
            fn next_u32(&mut self) -> u32 {
                self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (z ^ (z >> 31)) as u32
            }
        }
        let count = 1usize << log2_size;
        let mut rng = Rng(seed ^ u64::from(log2_size).rotate_left(29));
        (0..count)
            .map(|_| {
                let cv = std::array::from_fn(|_| rng.next_u32());
                let message = std::array::from_fn(|_| rng.next_u32());
                let counter = u64::from(rng.next_u32());
                (cv, message, counter, 64, 11)
            })
            .collect()
    }

    #[test]
    fn seed_pipe_matches_reference_generator() {
        for &log2 in &[8u32, 12, 13] {
            for &seed in &[0u64, 1, 0x00C0_FFEE_BEEF_D15C, u64::MAX, 0x5DEE_CE66_D_u64] {
                assert_eq!(
                    generate_compressions_par(log2, seed),
                    reference(log2, seed),
                    "log2={log2} seed={seed}"
                );
            }
        }
    }

    #[test]
    fn seed_pipe_matches_reference_at_ranked_size() {
        let seed = 0x1234_5678_9ABC_DEF0;
        assert_eq!(generate_compressions_par(18, seed), reference(18, seed));
    }

    #[test]
    fn seed_pipe_block_comparison_is_exact() {
        let a = generate_compressions_par(10, 7);
        let mut b = a.clone();
        assert!(blocks_eq(&a, &b));
        b[900].1[3] ^= 1;
        assert!(!blocks_eq(&a, &b));
        assert!(!blocks_eq(&a, &a[..a.len() - 1]));
    }

    #[test]
    fn blocks_eq_serial_agrees_with_the_parallel_form() {
        let a = generate_compressions_par(12, 7);
        let b = generate_compressions_par(12, 7);
        let mut c = b.clone();
        assert!(blocks_eq_serial(&a, &b));
        assert_eq!(blocks_eq_serial(&a, &b), blocks_eq(&a, &b));
        c.last_mut().expect("non-empty").2 ^= 1;
        assert!(!blocks_eq_serial(&a, &c));
        assert_eq!(blocks_eq_serial(&a, &c), blocks_eq(&a, &c));
        assert!(!blocks_eq_serial(&a, &b[..b.len() - 1]));
    }

    #[test]
    fn warmup_generator_check_is_fail_closed() {
        verify_generator_at_warmup(10, &generate_compressions_par(10, WARMUP_SEED));
        assert!(!GENERATOR_VERIFIED.load(Ordering::SeqCst));
        verify_generator_at_warmup(10, &[]);
        assert!(!GENERATOR_VERIFIED.load(Ordering::SeqCst));
        let a = generate_compressions_par(10, WARMUP_SEED);
        let b = generate_compressions_par(10, WARMUP_SEED ^ 1);
        assert_ne!(a.first(), b.first());
        assert_ne!(a.last(), b.last());
    }

    #[test]
    fn gen_block_matches_the_parallel_fill_at_every_index() {
        let (log2, seed) = (13u32, 0xDEAD_BEEF_1234_5678u64);
        let all = generate_compressions_par(log2, seed);
        let init = generator_init(log2, seed);
        for &i in &[0usize, 1, 7, 8, 4095, 4096, all.len() - 2, all.len() - 1] {
            assert_eq!(gen_block(init, i), all[i], "block {i}");
            assert_eq!(gen_block_scalar(init, i), all[i], "scalar block {i}");
            assert_eq!(gen_block_ilp(init, i), all[i], "ilp block {i}");
        }
        for &(lg, sd) in &[
            (10u32, 0u64),
            (13, 0xDEAD_BEEF_1234_5678),
            (18, u64::MAX),
            (18, 0x0123_4567_89AB_CDEF),
            (32, 0x9E37_79B9_7F4A_7C15),
        ] {
            let init = generator_init(lg, sd);
            let n = 1usize << lg.min(18);
            for i in (0..64).chain([n / 2 - 1, n / 2, n - 3, n - 2, n - 1]) {
                assert_eq!(
                    gen_block_scalar(init, i),
                    gen_block_ilp(init, i),
                    "scalar/ilp divergence lg={lg} sd={sd:#x} block {i}"
                );
            }
        }
    }

    #[test]
    fn gen_quad_soa_matches_four_reference_blocks() {
        assert_eq!(std::mem::size_of::<CompressionQuadSoa>(), 4 * 112);
        assert_eq!(std::mem::align_of::<CompressionQuadSoa>(), 16);
        for &(log2, seed) in &[
            (8u32, 0u64),
            (13, 0xDEAD_BEEF_1234_5678),
            (18, u64::MAX),
            (32, 0x0123_4567_89AB_CDEF),
        ] {
            let init = generator_init(log2, seed);
            let n = 1usize << log2.min(18);
            for &first in &[0usize, 4, 4092, 4096, n / 2, n - 4] {
                let quad = gen_quad_soa(init, first);
                let blocks: [Compression; 4] =
                    std::array::from_fn(|lane| gen_block_scalar(init, first + lane));
                for lane in 0..4 {
                    assert_eq!(
                        std::array::from_fn::<_, 8, _>(|word| quad.cv[word][lane]),
                        blocks[lane].0,
                        "cv log2={log2} seed={seed:#x} block={}",
                        first + lane
                    );
                    assert_eq!(
                        std::array::from_fn::<_, 16, _>(|word| quad.message[word][lane]),
                        blocks[lane].1,
                        "message log2={log2} seed={seed:#x} block={}",
                        first + lane
                    );
                    assert_eq!(u64::from(quad.counter_lo[lane]), blocks[lane].2);
                    assert_eq!(quad.counter_hi[lane], 0);
                    assert_eq!(quad.block_len[lane], blocks[lane].3);
                    assert_eq!(quad.flags[lane], blocks[lane].4);
                }
            }
        }

        let (log2, seed) = (8u32, 0xA55A_1234_5678_9ABC);
        let init = generator_init(log2, seed);
        let mut protected = reference(log2, seed);
        assert!(generated_quads_match_blocks(init, &protected));
        protected[77].1[5] ^= 1;
        assert!(!generated_quads_match_blocks(init, &protected));
        assert!(!generated_quads_match_blocks(init, &protected[..255]));
    }

    #[test]
    fn lazy_identity_is_inert_for_foreign_slices() {
        let _serial = SPEC_IDENTITY_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let blocks = generate_compressions_par(8, 42);
        assert!(spec_gen_init(&blocks).is_none());
        assert!(materialize_spec_blocks(&blocks).is_none());
    }

    #[test]
    fn lazy_materialization_keeps_the_shared_backing_immutable() {
        let _serial = SPEC_IDENTITY_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _reset = SpecIdentityReset;
        let (log2, seed) = (8u32, 0x0123_4567_89AB_CDEF);
        let init = generator_init(log2, seed);
        let backing = Arc::new(prefaulted_blocks(1usize << log2));
        assert!(backing.iter().all(|block| *block == ZERO_COMPRESSION));

        SPEC_INIT.store(init, Ordering::Relaxed);
        SPEC_LEN.store(backing.len(), Ordering::Relaxed);
        SPEC_BASE.store(backing.as_ptr() as usize, Ordering::Release);

        assert_eq!(spec_gen_init(backing.as_slice()), Some(init));
        let generated = materialize_spec_blocks(backing.as_slice())
            .expect("live lazy identity must produce owned blocks");
        assert_eq!(generated, reference(log2, seed));
        let expected_witness =
            crate::r1cs_hashes::blake3::generate_witness_batch_major(&generated, log2 as usize);
        let lazy_witness = crate::r1cs_hashes::blake3::generate_witness_batch_major(
            backing.as_slice(),
            log2 as usize,
        );
        assert_eq!(lazy_witness, expected_witness);
        assert!(backing.iter().all(|block| *block == ZERO_COMPRESSION));
        assert_eq!(spec_gen_init(backing.as_slice()), Some(init));
    }

    #[test]
    fn seed_pipe_stays_disarmed_outside_the_ranked_worker() {
        assert!(!is_ranked_worker());
        assert!(try_adopt(&[]).is_none());
    }

    #[test]
    fn speculative_publish_kill_is_literal_one() {
        assert!(!speculative_publish_enabled_for(false, None));
        assert!(speculative_publish_enabled_for(true, None));
        assert!(!speculative_publish_enabled_for(
            true,
            Some(OsStr::new("1"))
        ));
        for value in ["", "0", "01", "true", " 1", "1 "] {
            assert!(
                speculative_publish_enabled_for(true, Some(OsStr::new(value))),
                "unexpected kill for {value:?}"
            );
        }
    }

    #[test]
    fn speculative_publish_catches_every_failure_mode() {
        assert!(catch_speculative_publish(|| Ok(true)));
        assert!(!catch_speculative_publish(|| Ok(false)));
        assert!(!catch_speculative_publish(|| Err(io::Error::other("io"))));
        assert!(!catch_speculative_publish(|| -> io::Result<bool> {
            panic!("serialization invariant")
        }));
    }

    #[test]
    fn speculative_atomic_publish_never_exposes_partial_bytes() {
        let dir = publication_test_dir("atomic");
        let proof_path = dir.join("proof.bin");
        let expected: Vec<u8> = (0..(1 << 20)).map(|i| (i * 131 + 17) as u8).collect();
        let reader_path = proof_path.clone();
        let reader = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                match std::fs::read(&reader_path) {
                    Ok(bytes) => return bytes,
                    Err(error)
                        if error.kind() == io::ErrorKind::NotFound
                            && std::time::Instant::now() < deadline =>
                    {
                        std::thread::yield_now();
                    }
                    Err(error) => panic!("atomic publication reader failed: {error}"),
                }
            }
        });

        publish_bytes_atomically(&proof_path, &expected).expect("atomic publication");
        assert_eq!(reader.join().expect("reader thread"), expected);
        assert!(!speculative_temporary_path(&proof_path).exists());
        std::fs::remove_dir_all(&dir).expect("remove publication test directory");
    }

    #[test]
    fn speculative_atomic_publish_cleans_up_after_rename_failure() {
        let dir = publication_test_dir("failure");
        let proof_path = dir.join("proof.bin");
        std::fs::create_dir(&proof_path).expect("create conflicting destination");
        assert!(publish_bytes_atomically(&proof_path, b"complete proof").is_err());
        assert!(!speculative_temporary_path(&proof_path).exists());
        assert!(proof_path.is_dir());
        std::fs::remove_dir_all(&dir).expect("remove publication test directory");
    }

    #[test]
    fn preopen_publish_kill_is_literal_one() {
        assert!(preopen_publish_enabled_for(None));
        assert!(!preopen_publish_enabled_for(Some(OsStr::new("1"))));
        for value in ["", "0", "01", "true", " 1", "1 "] {
            assert!(
                preopen_publish_enabled_for(Some(OsStr::new(value))),
                "unexpected kill for {value:?}"
            );
        }
    }

    #[test]
    fn preopen_parks_the_fd_as_a_fresh_create_would() {
        let dir = publication_test_dir("preopen-parked");
        let proof_path = dir.join("proof.bin");
        let mut pre = preopen_publish_file(&proof_path).expect("pre-open in a writable scratch");
        assert!(speculative_temporary_path(&proof_path).exists());
        assert_eq!(
            std::fs::metadata(&speculative_temporary_path(&proof_path))
                .expect("temporary metadata")
                .len(),
            0
        );
        let mut file = pre.file.take().expect("parked fd");
        assert_eq!(file.stream_position().expect("parked offset"), 0);
        file.write_all(b"probe").expect("parked fd is writable");
        assert_eq!(
            std::fs::metadata(&speculative_temporary_path(&proof_path))
                .expect("temporary metadata")
                .len(),
            5
        );
        drop(file);
        drop(pre);
        assert!(!speculative_temporary_path(&proof_path).exists());
        assert!(!proof_path.exists());
        std::fs::remove_dir_all(&dir).expect("remove publication test directory");
    }

    #[test]
    fn preopen_publish_is_byte_identical_to_the_incumbent_path() {
        let dir = publication_test_dir("preopen-publish");
        let proof_path = dir.join("proof.bin");
        let expected: Vec<u8> = (0..(1 << 20)).map(|i| (i * 131 + 17) as u8).collect();
        let reader_path = proof_path.clone();
        let reader = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                match std::fs::read(&reader_path) {
                    Ok(bytes) => return bytes,
                    Err(error)
                        if error.kind() == io::ErrorKind::NotFound
                            && std::time::Instant::now() < deadline =>
                    {
                        std::thread::yield_now();
                    }
                    Err(error) => panic!("pre-opened publication reader failed: {error}"),
                }
            }
        });

        let pre = preopen_publish_file(&proof_path).expect("pre-open in a writable scratch");
        publish_bytes_through_preopened(&proof_path, &expected, pre)
            .expect("publication through the pre-opened fd");
        assert_eq!(reader.join().expect("reader thread"), expected);
        assert!(!speculative_temporary_path(&proof_path).exists());
        std::fs::remove_dir_all(&dir).expect("remove publication test directory");
    }

    #[test]
    fn preopen_publish_falls_back_and_cleans_up_on_failure() {
        let dir = publication_test_dir("preopen-failure");
        let proof_path = dir.join("proof.bin");
        std::fs::create_dir(&proof_path).expect("create conflicting destination");
        let pre = preopen_publish_file(&proof_path).expect("pre-open in a writable scratch");
        assert!(
            publish_bytes_through_preopened(&proof_path, b"complete proof", pre).is_err(),
            "rename over a directory must fail"
        );
        assert!(!speculative_temporary_path(&proof_path).exists());
        assert!(publish_bytes_atomically(&proof_path, b"complete proof").is_err());
        assert!(!speculative_temporary_path(&proof_path).exists());
        assert!(proof_path.is_dir());
        std::fs::remove_dir_all(&dir).expect("remove publication test directory");
    }

    #[test]
    fn preopen_skips_when_the_scratch_directory_is_missing() {
        let missing = std::env::temp_dir().join(format!(
            "flock-seed-pipe-preopen-missing-{}-{}",
            std::process::id(),
            PUBLICATION_TEST_ID.fetch_add(1, Ordering::Relaxed)
        ));
        let proof_path = missing.join("proof.bin");
        assert!(!missing.exists());
        assert!(preopen_publish_file(&proof_path).is_none());
        assert!(!speculative_temporary_path(&proof_path).exists());
    }
}
