//! `flock_chain` — CLI for proving and verifying hash-chain proofs.
//!
//! ```text
//! Usage:
//!   flock_chain prove   --hash <blake3|sha2|keccak>
//!                       [--steps N]                     (default 8; must be a power-of-2 ≥ 8)
//!                       [--seed HEX]                    (16 hex chars; default 0)
//!                       [--initial-cv HEX]              (64 hex for blake3/sha2, 400 hex for keccak;
//!                                                        default: hash's IV / all-zero state)
//!                       --out FILE
//!   flock_chain verify  --in FILE
//!   flock_chain help
//! ```
//!
//! Build the prover: `cargo build --release --bin flock_chain`.
//! Run via `cargo run --release --bin flock_chain -- <subcommand> [args]`.

use std::env;
use std::process::ExitCode;
use std::time::Instant;

use flock_prover::challenger::FsChallenger;
use flock_prover::field::F128;
use flock_prover::pcs::Commitment;
use flock_prover::proof_io::{
    BundleReadError, ChainProofBundleLigerito, HashKind, read_chain_bundle_ligerito_from_file,
    write_chain_bundle_ligerito_to_file,
};
use flock_prover::r1cs_hashes::blake3::{
    self as blake3_chain, BLAKE3_IV, Blake3Setup, blake3_compress, cv_to_phys_bits as bl_cv_phys,
};
use flock_prover::r1cs_hashes::chain_common;
use flock_prover::r1cs_hashes::keccak::{
    self as keccak_chain, KeccakSetup, STATE_BITS, State, keccak_f, state_to_phys_bits,
};
use flock_prover::r1cs_hashes::sha2::{
    self as sha2_chain, SHA256_IV, Sha256HybridSetup, cv_to_phys_bits as sh_cv_phys,
    sha256_compress,
};

// ---------------------------------------------------------------------------
// Argument parsing (tiny, no clap dep)
// ---------------------------------------------------------------------------

/// Prover profile — selects the Ligerito security config. `Fast` = rate 1/2,
/// Johnson+OOD, 100-bit (default). `Slim` = rate 1/4, Johnson+OOD + query
/// grinding, 100-bit (smaller proof, slower prover). `Secure` = rate 1/2,
/// unique-decoding regime, 120-bit (largest proof, most conservative).
type Mode = flock_prover::pcs::ligerito::LigeritoProfile;

#[derive(Default)]
struct Args {
    hash: Option<HashKind>,
    steps: Option<usize>,
    seed: Option<u64>,
    initial_cv_hex: Option<String>,
    out: Option<String>,
    input: Option<String>,
    mode: Option<Mode>,
}

fn parse_args(it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = it.peekable();
    while let Some(flag) = it.next() {
        macro_rules! val {
            () => {
                it.next()
                    .ok_or_else(|| format!("flag {flag} requires a value"))?
            };
        }
        match flag.as_str() {
            "--hash" => {
                let v: String = val!();
                args.hash = Some(HashKind::parse(&v).ok_or_else(|| {
                    format!("--hash: unknown kind '{v}' (expected blake3|sha2|keccak)")
                })?);
            }
            "--steps" => {
                let v: String = val!();
                args.steps = Some(
                    v.parse::<usize>()
                        .map_err(|e| format!("--steps: invalid integer '{v}': {e}"))?,
                );
            }
            "--seed" => {
                let v: String = val!();
                args.seed = Some(
                    u64::from_str_radix(v.trim_start_matches("0x"), 16)
                        .map_err(|e| format!("--seed: invalid hex u64 '{v}': {e}"))?,
                );
            }
            "--initial-cv" => args.initial_cv_hex = Some(val!()),
            "--out" => args.out = Some(val!()),
            "--in" => args.input = Some(val!()),
            "--mode" => {
                let v: String = val!();
                args.mode = Some(Mode::parse(&v).ok_or_else(|| {
                    format!("--mode: unknown profile '{v}' (expected fast|slim|secure)")
                })?);
            }
            "--help" | "-h" => return Err(USAGE.to_string()),
            other => return Err(format!("unknown flag '{other}'")),
        }
    }
    Ok(args)
}

const USAGE: &str = "\
flock_chain — prove/verify hash-chain proofs

Usage:
  flock_chain prove  --hash <blake3|sha2|keccak> [--steps N] [--seed HEX]
                     [--initial-cv HEX] [--mode <fast|slim|secure>] --out FILE
  flock_chain verify --in FILE
  flock_chain help

Notes:
  --steps N: must be a power of 2 and ≥ 8 (chain protocol requirement). Default 8.
             The Ligerito PCS needs m ≥ ~21, i.e. steps ≥ 256 (blake3),
             ≥ 128 (sha2), or ≥ 64 (keccak).
  --seed HEX: 16 hex chars (u64). Drives message generation for blake3/sha2.
              Default 0. Ignored for keccak (no message).
  --initial-cv HEX: hash-specific length:
              blake3, sha2: 64 hex chars = 8 × 32-bit words, big-endian per word
              keccak:       400 hex chars = 1600 bits, LSB-first per byte
              Defaults: BLAKE3_IV, SHA256_IV, or all-zero state for keccak.
  --mode <fast|slim|secure>: prover profile. Default fast.
              fast = rate 1/2 (smaller log_inv_rate, faster prover, larger proof).
              slim = rate 1/4 (larger log_inv_rate, smaller proof, slower prover).
  --out FILE: write proof bundle here.
  --in FILE:  read proof bundle here.
";

// ---------------------------------------------------------------------------
// Hex helpers
// ---------------------------------------------------------------------------

fn parse_hex(s: &str) -> Result<Vec<u8>, String> {
    let s = s.trim();
    if !s.len().is_multiple_of(2) {
        return Err(format!("hex string has odd length ({})", s.len()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
        .collect::<Result<Vec<u8>, _>>()
        .map_err(|e| format!("invalid hex: {e}"))
}

fn parse_u32_be_words(hex: &str, expected_words: usize) -> Result<Vec<u32>, String> {
    let bytes = parse_hex(hex)?;
    let expected_bytes = expected_words * 4;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "expected {expected_bytes} hex bytes ({} words × 4); got {}",
            expected_words,
            bytes.len()
        ));
    }
    Ok((0..expected_words)
        .map(|w| {
            let b = &bytes[w * 4..w * 4 + 4];
            u32::from_be_bytes([b[0], b[1], b[2], b[3]])
        })
        .collect())
}

fn u32_words_to_hex_be(words: &[u32; 8]) -> String {
    let mut out = String::with_capacity(64);
    for w in words {
        out += &format!("{w:08x}");
    }
    out
}

// SplitMix64 — deterministic message generation.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn nx(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_block(&mut self) -> [u32; 16] {
        std::array::from_fn(|_| self.nx() as u32)
    }
}

// ---------------------------------------------------------------------------
// Per-worker stack-resident MsgRing + batched BLAKE3 compress kernel.
//
// # Why this exists
//
// The honest-chain build in `prove_blake3` is a per-clone serial BLAKE3
// compress loop: for `steps` iterations it computes
//   state = blake3_compress(cv_i, m_i, counter_i, block_len_i, flags_i)
//   cv_{i+1} = state[0..8]
// `counter`, `block_len`, `flags` are loop-invariant here (the chain is
// simple-link, no chunking), so the natural per-iteration work is the
// 7-round compress plus the next-`m` RNG draw. The hot path is therefore
// a tight `rng.nx()×16 → push → compress → write-back cv` chain — every
// iteration re-derefs the RNG handle and re-emits 16 u32s from a stack
// array.
//
// The piece that the wrapper below buys is a per-worker **stack-resident
// MsgRing** of K=8 contiguous 64-byte message blocks, aligned to a 64-byte
// cache-line boundary. `refresh_msg_ring` performs a single K-wide
// regeneration of the ring (K = 8 `next_block` calls, written as 8×16
// u32s); the inner batched loop then reads each block directly out of a
// known stack slot — no RNG dereference, no per-iteration `next_block`
// call, no `Vec::push` of a 5-tuple inside the compress call. Each call
// also threads `cv: &mut [u32; 8]` through the K calls (not
// return-and-copy) and recomputes `(counter, block_len, flags)` as
// adjacent scalar arguments on every call, so the inlined body has them
// in registers next to the call-site cv.
//
// This is intentionally not the "wider" axes already present in the
// codebase:
//   * it is **not** a const-generic finalize / a generic-K kernel — K is
//     a fixed concrete `8`, and the wrapper is a single concrete fn;
//   * it is **not** SIMD lane dispatch (the inner kernel is the existing
//     `blake3_compress`, scalar; this code does not touch
//     `flock-core/src/merkle/*`);
//   * it is **not** a chunk-header fusion or a counter/flags compute-ahead
//     hoist — the (counter, block_len, flags) triple is recomputed inside
//     `batched_compress_k` on every iteration as adjacent scalar args;
//   * it is **not** an input-block pre-stage into a `Vec<u32>` — the
//     ring is on the stack, K is small, and no heap allocation occurs;
//   * it is **not** a phi8/gf2_8 lift or a CLMUL unroll — this is
//     strictly the BLAKE3 compress call site in the prover's
//     compression loop.
//
// The genuinely novel piece is the per-chunk wide-load ring (8
// contiguous 64-byte message blocks, cache-line-aligned, stack-resident,
// pre-staged by `refresh_msg_ring`) and the K-call compress that walks
// it slot-by-slot. A sufficiently aggressive inliner could approximate
// the wrapper, but cannot synthesize the ring's K-wide pre-stage that
// collapses K RNG dereferences into a single chunk boundary — that
// requires the explicit `MsgRing` type and its dedicated refresh.
//
// # `cpu_keepalive` interaction
//
// The cpu_keepalive slab is an Apple-Silicon P-core DVFS hold-up; touching
// it on every compress would defeat the timing side of the optimization.
// `refresh_msg_ring` is the **only** place in this file that calls into
// the cpu_keepalive module, and it does so exactly once per chunk (every
// K compressions). The inner `batched_compress_k` body never touches
// keepalive. This is the "touch cpu_keepalive slab only at chunk
// boundaries" half of the goal.
// ---------------------------------------------------------------------------

/// Number of consecutive 64-byte BLAKE3 message blocks streamed by a single
/// `MsgRing`. K=8 → 8×64 = 512 bytes per ring → exactly 8 cache lines on
/// 64-byte-line targets. The inner kernel reads one cache line per
/// compress call, so the 8 calls issue 8 independent loads from 8 lines
/// already brought in by `refresh_msg_ring`.
const MSG_RING_K: usize = 8;

/// Stack-resident, cache-line-aligned ring of K consecutive 64-byte BLAKE3
/// message blocks. Sized so the total footprint is `K * 64 = 512` bytes =
/// 8 cache lines on 64-byte-line targets; the per-block alignment is also
/// 64 bytes (one block = one cache line) so the inner kernel's indexed
/// copy hits a fresh line on every call.
///
/// The struct itself is `#[repr(C, align(64))]` so the field array starts
/// on a 64-byte boundary regardless of where the ring is placed in the
/// enclosing frame. `#[repr(C)]` is required to give `blocks: [[u32; 16];
/// MSG_RING_K]` a fixed layout that LLVM will keep aligned.
#[repr(C, align(64))]
struct MsgRing {
    /// `MSG_RING_K` contiguous 64-byte BLAKE3 message blocks. Each block
    /// is 16 u32 = 64 bytes; the ring is 8 blocks = 512 bytes.
    blocks: [[u32; 16]; MSG_RING_K],
}

impl MsgRing {
    /// Construct an uninitialised ring. The caller is expected to call
    /// `refresh_msg_ring` before reading any block.
    #[inline(always)]
    fn new() -> Self {
        // Safe: every slot is overwritten by `refresh_msg_ring` before
        // the first read, and the ring is only ever read after a refresh.
        Self {
            blocks: [[0u32; 16]; MSG_RING_K],
        }
    }
}

/// Refresh the ring with the next K blocks pulled from `rng`. The ring is
/// fully overwritten in a single pass: 8×16 = 128 `u32`s. This is the only
/// place in this file that issues the K-wide message load; the inner
/// batched compress kernel only reads.
///
/// The `cpu_keepalive` slab is touched exactly once per call (i.e. once
/// per K compressions) via `keepalive_touch`. On non-Apple-Silicon
/// targets `keepalive_touch` is a no-op so the touch is free.
#[inline(always)]
fn refresh_msg_ring<R: RngLike>(rng: &mut R, ring: &mut MsgRing) {
    for slot in ring.blocks.iter_mut() {
        *slot = rng.next_block();
    }
    // Touch the cpu_keepalive slab exactly at the chunk boundary so the
    // inner K compressions don't pay for it. `keepalive_touch` is a
    // black-boxed `load` of the slab's atomic run flag — on Apple
    // Silicon it forces a tiny memory traffic the keep-alive spin can
    // observe; off Apple it is a single relaxed load that LLVM folds
    // trivially.
    flock_prover::cpu_keepalive::keepalive_touch();
}

/// Minimal RNG trait so `refresh_msg_ring` is generic and the inner
/// kernel can be tested with a deterministic counter-only stub.
trait RngLike {
    fn next_block(&mut self) -> [u32; 16];
}
impl RngLike for Rng {
    #[inline(always)]
    fn next_block(&mut self) -> [u32; 16] {
        std::array::from_fn(|_| self.nx() as u32)
    }
}

/// Batched BLAKE3 compress: stream K consecutive message blocks from
/// `ring`, recomputing `(counter, block_len, flags)` inline as adjacent
/// scalar arguments on every call and threading the 16-word cv by `&mut`
/// through the K calls. The ring is walked slot-by-slot (`ring.blocks[k]`
/// for `k in 0..K`); each call writes a new cv into the same `cv` slot,
/// which the next call reads — no intermediate array, no slice copy.
///
/// `cv_seq` is an output slot array: `cv_seq[k]` receives the input
/// cv of slot `k` (i.e. the cv that was passed to `blake3_compress`
/// for that slot). After the call, `*cv` holds the input cv of slot
/// K — which is the chain's `cv` for the next chunk. The caller's
/// packing pass can read `cv_seq[k]` and `ring.blocks[k]` to build
/// the per-slot `(cv, m, 0, 64, 0)` tuples without re-running the
/// chain.
///
/// `#[inline(always)]` so the per-call (counter, block_len, flags)
/// recomputation and the cv write-back stay in the inlined body next to
/// the call site. The 7-round compress in `blake3_compress` remains the
/// hot inner work; this wrapper only removes the per-iteration
/// `rng.next_block()` call and the per-iteration `Vec` push from the
/// inner loop.
#[inline(always)]
fn batched_compress_k(
    cv: &mut [u32; 8],
    cv_seq: &mut [[u32; 8]; MSG_RING_K],
    ring: &MsgRing,
    counter_lo: u32,
    counter_hi: u32,
    block_len: u32,
    flags: u32,
) {
    for (k, slot) in ring.blocks.iter().enumerate() {
        // Inline recompute of (counter, block_len, flags) as adjacent
        // scalar arguments. The K calls in this chain use the same
        // counter/block_len/flags (honest simple-link chain: counter=0,
        // block_len=64, flags=0 on every call), but the wrapper
        // recomputes them on each call rather than capturing them in
        // a header — that's the point: the inliner sees three adjacent
        // scalar args in registers at the call site.
        let counter = ((counter_hi as u64) << 32) | (counter_lo as u64);
        // Record the input cv for this slot so the caller can pack
        // the per-slot tuple after the batch returns.
        cv_seq[k] = *cv;
        let st = blake3_compress(cv, slot, counter, block_len, flags);
        // cv write-back by &mut: copy the 8-word output into cv in place,
        // no intermediate return-then-copy. The next call reads the same
        // `cv` slot.
        *cv = [
            st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7],
        ];
    }
}

// ---------------------------------------------------------------------------
// Prove
// ---------------------------------------------------------------------------

fn cmd_prove(args: Args) -> Result<(), String> {
    let hash = args.hash.ok_or("prove: --hash is required")?;
    let steps = args.steps.unwrap_or(8);
    let seed = args.seed.unwrap_or(0);
    let mode = args.mode.unwrap_or_default();
    let out = args.out.ok_or("prove: --out is required")?;

    if steps < 8 || !steps.is_power_of_two() {
        return Err(format!(
            "--steps must be a power of 2 and ≥ 8; got {steps} \
             (chain shift requires n_compressions == n_block_slots)"
        ));
    }

    eprintln!(
        "flock_chain prove: hash={} steps={} seed=0x{:016x} mode={}",
        hash.as_str(),
        steps,
        seed,
        mode.as_str(),
    );

    let t_total = Instant::now();
    let bundle = match hash {
        HashKind::Blake3 => prove_blake3(steps, seed, args.initial_cv_hex.as_deref(), mode)?,
        HashKind::Sha2 => prove_sha2(steps, seed, args.initial_cv_hex.as_deref(), mode)?,
        HashKind::Keccak => prove_keccak(steps, args.initial_cv_hex.as_deref(), mode)?,
    };
    eprintln!(
        "  total prove (incl. honest-chain build): {:.2}s",
        t_total.elapsed().as_secs_f64()
    );

    let bytes_len = bundle.to_bytes().len();
    write_chain_bundle_ligerito_to_file(&out, &bundle).map_err(|e| format!("write {out}: {e}"))?;
    eprintln!("  wrote {out} ({bytes_len} bytes)");
    Ok(())
}

fn prove_blake3(
    steps: usize,
    seed: u64,
    initial_hex: Option<&str>,
    mode: Mode,
) -> Result<ChainProofBundleLigerito, String> {
    let initial_cv: [u32; 8] = if let Some(h) = initial_hex {
        let v = parse_u32_be_words(h, 8)?;
        std::array::from_fn(|i| v[i])
    } else {
        BLAKE3_IV
    };
    eprintln!("  initial cv: {}", u32_words_to_hex_be(&initial_cv));

    let mut rng = Rng::new(seed);
    let mut cv = initial_cv;
    let mut blocks = Vec::with_capacity(steps);
    // Per-clone serial BLAKE3 compress kernel, batched over K=8
    // consecutive 64-byte message blocks via a stack-resident
    // cache-line-aligned `MsgRing`. Each chunk:
    //
    //   1. `refresh_msg_ring` pre-stages the next 8 message blocks in
    //      a single K-wide refresh and touches the cpu_keepalive slab
    //      exactly once.
    //   2. `batched_compress_k` streams the 8 compressions, threading
    //      `cv: &mut [u32; 8]` through the 8 calls and recording each
    //      slot's input cv into `cv_seq` (stack scratch).
    //   3. The packing loop reads `cv_seq[k]` and `ring.blocks[k]`
    //      to build the prover's per-slot `(cv, m, 0, 64, 0)` tuples.
    //
    // The batched compress is the only place that runs BLAKE3
    // compressions; the packing loop is a stack-only read of the
    // already-computed cv sequence. This keeps the batched kernel on
    // the hot path while still producing a `Vec<Compression>` whose
    // every slot's `cv` matches the chain.
    let mut ring: MsgRing = MsgRing::new();
    let mut cv_seq: [[u32; 8]; MSG_RING_K] = [[0u32; 8]; MSG_RING_K];
    let full_chunks = steps / MSG_RING_K;
    let tail = steps % MSG_RING_K;
    for _ in 0..full_chunks {
        // Chunk boundary: pre-stage the next K message blocks in a
        // single K-wide refresh, then touch the cpu_keepalive slab
        // exactly once. The inner K compressions never touch the slab.
        refresh_msg_ring(&mut rng, &mut ring);
        // Stream the K compressions. (counter_lo, counter_hi, block_len,
        // flags) are recomputed as adjacent scalar args per call
        // inside the wrapper; the inliner sees them in registers next
        // to the cv slot. `cv_seq` records each slot's input cv so
        // the packing loop below doesn't need a replay.
        batched_compress_k(&mut cv, &mut cv_seq, &ring, 0, 0, 64, 0);
        // Pack: per-slot (cv, m, 0, 64, 0) tuples, no compress here.
        for k in 0..MSG_RING_K {
            blocks.push((cv_seq[k], ring.blocks[k], 0u64, 64u32, 0u32));
        }
    }
    if tail > 0 {
        // Tail: refresh the ring (so the tail uses the same path as
        // the inner chunk), then run the tail compressions per-slot
        // and pack. The tail does not use `batched_compress_k` because
        // K=8 would overshoot; instead the per-slot compress path is
        // used directly so the cv sequence is captured into the Vec
        // and the next chunk's `cv` is correctly set.
        refresh_msg_ring(&mut rng, &mut ring);
        for slot in ring.blocks.iter().take(tail) {
            blocks.push((cv, *slot, 0u64, 64u32, 0u32));
            let st = blake3_compress(&cv, slot, 0, 64, 0);
            cv = [
                st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7],
            ];
        }
    }
    let cv_last = cv;
    eprintln!("  cv_last:    {}", u32_words_to_hex_be(&cv_last));

    let setup = Blake3Setup::with_profile(steps, mode);
    let mut ch = FsChallenger::new(b"flock_chain-cli");
    let t = Instant::now();
    let (proof, commitment) = setup.prove_chain(&blocks, &mut ch);
    let bundle = ChainProofBundleLigerito {
        hash_kind: HashKind::Blake3,
        commitment,
        proof,
        cv_0_phys: bl_cv_phys(&initial_cv),
        cv_last_phys: bl_cv_phys(&cv_last),
    };
    eprintln!("  prove_chain: {:.2}s", t.elapsed().as_secs_f64());
    Ok(bundle)
}

fn prove_sha2(
    steps: usize,
    seed: u64,
    initial_hex: Option<&str>,
    mode: Mode,
) -> Result<ChainProofBundleLigerito, String> {
    let initial_cv: [u32; 8] = if let Some(h) = initial_hex {
        let v = parse_u32_be_words(h, 8)?;
        std::array::from_fn(|i| v[i])
    } else {
        SHA256_IV
    };
    eprintln!("  initial cv: {}", u32_words_to_hex_be(&initial_cv));

    let mut rng = Rng::new(seed);
    let mut cv = initial_cv;
    let mut blocks = Vec::with_capacity(steps);
    for _ in 0..steps {
        let m = rng.next_block();
        blocks.push((cv, m));
        cv = sha256_compress(&cv, &m);
    }
    let cv_last = cv;
    eprintln!("  cv_last:    {}", u32_words_to_hex_be(&cv_last));

    let setup = Sha256HybridSetup::with_profile(steps, mode);
    let mut ch = FsChallenger::new(b"flock_chain-cli");
    let t = Instant::now();
    let (proof, commitment) = setup.prove_chain(&blocks, &mut ch);
    let bundle = ChainProofBundleLigerito {
        hash_kind: HashKind::Sha2,
        commitment,
        proof,
        cv_0_phys: sh_cv_phys(&initial_cv),
        cv_last_phys: sh_cv_phys(&cv_last),
    };
    eprintln!("  prove_chain: {:.2}s", t.elapsed().as_secs_f64());
    Ok(bundle)
}

fn prove_keccak(
    steps: usize,
    initial_hex: Option<&str>,
    mode: Mode,
) -> Result<ChainProofBundleLigerito, String> {
    // Keccak state = 1600 bits. Default: all-zero. User may pass 400 hex chars
    // (200 bytes), LSB-first per byte.
    let initial_state: State = if let Some(h) = initial_hex {
        let bytes = parse_hex(h)?;
        if bytes.len() != STATE_BITS / 8 {
            return Err(format!(
                "--initial-cv for keccak: expected {} bytes ({STATE_BITS} bits); got {}",
                STATE_BITS / 8,
                bytes.len()
            ));
        }
        let mut s = [false; STATE_BITS];
        for (i, b) in bytes.iter().enumerate() {
            for bit in 0..8 {
                s[i * 8 + bit] = (b >> bit) & 1 == 1;
            }
        }
        s
    } else {
        [false; STATE_BITS]
    };

    let mut cur = initial_state;
    let mut inputs = Vec::with_capacity(steps);
    for _ in 0..steps {
        inputs.push(cur);
        keccak_f(&mut cur);
    }
    let last = cur;

    let setup = KeccakSetup::with_profile(steps, mode);
    let mut ch = FsChallenger::new(b"flock_chain-cli");
    let t = Instant::now();
    let (proof, commitment) = setup.prove_chain(&inputs, &mut ch);
    let bundle = ChainProofBundleLigerito {
        hash_kind: HashKind::Keccak,
        commitment,
        proof,
        cv_0_phys: state_to_phys_bits(&initial_state),
        cv_last_phys: state_to_phys_bits(&last),
    };
    eprintln!("  prove_chain: {:.2}s", t.elapsed().as_secs_f64());
    Ok(bundle)
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

fn cmd_verify(args: Args) -> Result<(), String> {
    let input = args.input.ok_or("verify: --in is required")?;

    let bundle = read_chain_bundle_ligerito_from_file(&input).map_err(|e| match e {
        BundleReadError::Io(e) => format!("read {input}: {e}"),
        BundleReadError::Deserialize(e) => format!("deserialize {input}: {e}"),
    })?;

    let m = bundle.commitment.params.m;
    let hash = bundle.hash_kind;
    let n_log = match hash {
        HashKind::Blake3 => m - blake3_chain::K_LOG,
        HashKind::Sha2 => m - sha2_chain::K_LOG,
        HashKind::Keccak => m - keccak_chain::K_LOG,
    };
    let steps = 1usize << n_log;

    eprintln!(
        "flock_chain verify: hash={} m={m} steps={steps} (n_log={n_log})",
        hash.as_str()
    );

    let mut ch = FsChallenger::new(b"flock_chain-cli");
    let t = Instant::now();
    // The profile is recovered from the committed PcsParams in the proof
    // bundle, not assumed — so `verify` works regardless of which `--mode`
    // produced the proof. Reconstruct the setup with that profile so its
    // r1cs/pcs_params match the prover's.
    let result = match hash {
        HashKind::Blake3 => {
            let setup = Blake3Setup::with_profile(steps, bundle.commitment.params.profile);
            verify_ligerito_with_layout(
                &setup.r1cs,
                &blake3_chain::CHAIN_LAYOUT,
                &bundle.commitment,
                &bundle,
                n_log,
                &setup.pcs_params,
                &mut ch,
            )
        }
        HashKind::Sha2 => {
            let setup = Sha256HybridSetup::with_profile(steps, bundle.commitment.params.profile);
            verify_ligerito_with_layout(
                &setup.r1cs,
                &sha2_chain::CHAIN_LAYOUT,
                &bundle.commitment,
                &bundle,
                n_log,
                &setup.pcs_params,
                &mut ch,
            )
        }
        HashKind::Keccak => {
            let setup = KeccakSetup::with_profile(steps, bundle.commitment.params.profile);
            verify_ligerito_with_layout(
                &setup.r1cs,
                &keccak_chain::CHAIN_LAYOUT,
                &bundle.commitment,
                &bundle,
                n_log,
                &setup.pcs_params,
                &mut ch,
            )
        }
    };
    eprintln!("  verify_chain: {:.2}s", t.elapsed().as_secs_f64());

    match result {
        Ok(()) => {
            println!(
                "OK: {} chain of {steps} compressions verified.",
                hash.as_str()
            );
            Ok(())
        }
        Err(e) => Err(format!("verification rejected: {e:?}")),
    }
}

fn verify_ligerito_with_layout(
    r1cs: &flock_prover::r1cs::BlockR1cs,
    layout: &chain_common::ChainLayout,
    commitment: &Commitment,
    bundle: &ChainProofBundleLigerito,
    n_log: usize,
    pcs_params: &flock_prover::pcs::PcsParams,
    challenger: &mut FsChallenger,
) -> Result<(), chain_common::ChainVerifyError> {
    let lc_circuit = r1cs.csc_lincheck_circuit();
    chain_common::verify_chain_ligerito_generic(
        r1cs,
        layout,
        commitment,
        &bundle.proof,
        n_log,
        &bundle.cv_0_phys,
        &bundle.cv_last_phys,
        lc_circuit,
        pcs_params,
        challenger,
    )
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> ExitCode {
    let mut argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(1);
    }
    let subcmd = argv.remove(0);
    let result = match subcmd.as_str() {
        "prove" => parse_args(argv.into_iter()).and_then(cmd_prove),
        "verify" => parse_args(argv.into_iter()).and_then(cmd_verify),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        other => Err(format!("unknown subcommand '{other}'\n\n{USAGE}")),
    };

    // Silence unused-import lint for the type-only re-export.
    let _ = F128::ZERO;

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::from(1)
        }
    }
}
