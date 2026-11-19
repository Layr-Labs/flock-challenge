//! Counter-based compression fixtures and lane-layout helpers. The prover
//! consumes caller-supplied blocks; this module performs no seed or proof I/O.

use rayon::prelude::*;

use crate::r1cs_hashes::blake3::Compression;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
/// `cv[8] + message[16] + counter[1]` draws per generated compression.
const DRAWS_PER_BLOCK: usize = 25;
const ZERO_COMPRESSION: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

// ---------------------------------------------------------------------------
// Counter-based reproduction of the protected generator
// ---------------------------------------------------------------------------

#[inline(always)]
fn mix(mut z: u64) -> u32 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) as u32
}

/// Bit-exact parallel reproduction of
/// `flock_benchmark_common::generate_compressions`.
///
/// The reference walks one `Rng` sequentially; because its state recurrence is
/// `s += GOLDEN` (the mixing function is *not* fed back), the state before
/// block `i`'s first draw is `init + 25·i·GOLDEN` and blocks are independent.
/// `seed_pipe_matches_reference_generator` checks the full ranked-size output
/// against a literal transcription of the reference.
pub fn generate_compressions_par(log2_size: u32, seed: u64) -> Vec<Compression> {
    let mut out = zero_blocks(1usize << log2_size);
    fill_compressions_par(&mut out, log2_size, seed);
    out
}

/// Allocate `count` valid sentinel blocks.
///
/// A `Vec<Compression>` may expose only initialized elements, even when the
/// element type is `Copy` and has no drop glue. Lazy witgen identifies this
/// allocation by address and synthesizes the real blocks without reading these
/// sentinels as inputs.
fn zero_blocks(count: usize) -> Vec<Compression> {
    vec![ZERO_COMPRESSION; count]
}

/// The reference generator's initial state for `(log2_size, seed)` —
/// `flock_benchmark_common::generate_compressions` seeds its `Rng` with
/// exactly this value.
#[inline(always)]
fn generator_init(log2_size: u32, seed: u64) -> u64 {
    seed ^ u64::from(log2_size).rotate_left(29)
}

/// One block of the protected generator's output, from the closed form.
///
/// The reference RNG's state recurrence is `s += GOLDEN` (the mixing function
/// is not fed back), so the state before block `i`'s first draw is
/// `init + 25·i·GOLDEN` and every block is computable in isolation. Both the
/// eager parallel fill and the lazy per-quad AoS fallback write through this
/// definition. `seed_pipe_matches_reference_generator`
/// pins it against a literal transcription of the reference, and
/// `gen_quad_soa_matches_four_reference_blocks` independently pins the ranked
/// word-major quad form to four results from this scalar recurrence.
#[inline(always)]
pub(crate) fn gen_block(init: u64, block: usize) -> Compression {
    gen_block_with(init, block, !gen_block_ilp_killed())
}

/// Exact-`1` kill for the ILP-unrolled [`gen_block`]; anything else leaves
/// it on. Deliberately UNCACHED (one getenv per call) so same-process A/B
/// tests can toggle it: the hot paths (the lazy witgen quad synth and the
/// eager parallel fill) resolve it once per prove/fill and pass the bool
/// down via [`gen_block_with`]; only cold callers pay the per-call read.
pub(crate) fn gen_block_ilp_killed() -> bool {
    std::env::var("FLOCK_NO_GEN_BLOCK_SIMD").is_ok_and(|v| v == "1")
}

/// [`gen_block`] with the ILP/scalar choice resolved by the caller (hot
/// paths hoist the env read out of their per-block loops).
#[inline(always)]
pub(crate) fn gen_block_with(init: u64, block: usize, ilp: bool) -> Compression {
    if ilp {
        gen_block_ilp(init, block)
    } else {
        gen_block_scalar(init, block)
    }
}

/// Incumbent scalar body: one running state, 25 sequential `+= GOLDEN`
/// steps. Kept verbatim as the `FLOCK_NO_GEN_BLOCK_SIMD=1` control and the
/// equality oracle for [`gen_block_ilp`].
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

/// ILP form of [`gen_block_scalar`] (witgen-stack item C): the state before
/// draw `d` is the closed form `base + (d+1)·GOLDEN` (the recurrence is a
/// pure counter — the mix is not fed back), so all 25 draws are
/// independent. Computing states by closed-form multiply in a 4-wide
/// unroll removes the serial add chain and lets the core run four
/// independent `mix` pipelines (2 multiplies each) abreast instead of
/// issuing them behind a 25-deep dependency-ordered loop.
/// `wrapping_mul`/`wrapping_add` reassociation is exact: everything is
/// mod-2^64 ring arithmetic, so `(init + 25·b·G) + (d+1)·G` equals the
/// scalar's stepped state bit-for-bit.
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
        // Four fully independent state computations + mixes per step.
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

/// Four consecutive protected-generator inputs in the word-major layout the
/// ranked NEON witness builder consumes. Every field is initialized normally;
/// this small value carries generated inputs rather than the lazy backing
/// vector's valid sentinel values.
#[repr(align(16))]
pub(crate) struct CompressionQuadSoa {
    pub(crate) cv: [[u32; 4]; 8],
    pub(crate) message: [[u32; 4]; 16],
    pub(crate) counter_lo: [u32; 4],
    pub(crate) counter_hi: [u32; 4],
    pub(crate) block_len: [u32; 4],
    pub(crate) flags: [u32; 4],
}

/// Generate blocks `first_block..first_block + 4` directly in word-major
/// form. The four RNG states are independent counter streams separated by
/// exactly `25 * GOLDEN`; stepping them together exposes the same four-way ILP
/// as [`gen_block_ilp`] without first materializing four AoS `Compression`s or
/// transposing their words in the witness builder.
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

/// [`gen_quad_soa`] written in place: the chaining, message and counter rows
/// are drawn straight into their fields of `out` and the three constant rows
/// are stored as values, so there is no out-of-line row helper, no
/// whole-struct copy and no pattern fill call. Same states, same draw order,
/// same 448 bytes.
#[inline(always)]
pub(crate) fn gen_quad_soa_into(
    init: u64,
    first_block: usize,
    out: &mut std::mem::MaybeUninit<CompressionQuadSoa>,
) -> &CompressionQuadSoa {
    let first_state =
        init.wrapping_add(((DRAWS_PER_BLOCK * first_block) as u64).wrapping_mul(GOLDEN));
    let block_stride = (DRAWS_PER_BLOCK as u64).wrapping_mul(GOLDEN);
    let mut s0 = first_state;
    let mut s1 = first_state.wrapping_add(block_stride);
    let mut s2 = first_state.wrapping_add(block_stride.wrapping_mul(2));
    let mut s3 = first_state.wrapping_add(block_stride.wrapping_mul(3));
    let mut draw_row = || {
        s0 = s0.wrapping_add(GOLDEN);
        s1 = s1.wrapping_add(GOLDEN);
        s2 = s2.wrapping_add(GOLDEN);
        s3 = s3.wrapping_add(GOLDEN);
        [mix(s0), mix(s1), mix(s2), mix(s3)]
    };
    let p = out.as_mut_ptr();
    // The count is 8; read through `black_box` so the loop stays rolled like
    // the message loop (fully unrolled, its state steps turn into one 64-bit
    // constant per draw).
    let cv_rows = std::hint::black_box(8usize);
    debug_assert_eq!(cv_rows, 8);
    // SAFETY: `black_box` returns its argument, so the first loop writes the
    // 8 rows of `cv`; the other fields are written by name, so all 448 bytes
    // are written before the shared reference is formed.
    unsafe {
        let cv = std::ptr::addr_of_mut!((*p).cv).cast::<[u32; 4]>();
        for w in 0..cv_rows {
            cv.add(w).write(draw_row());
        }
        for w in 0..16 {
            std::ptr::addr_of_mut!((*p).message[w]).write(draw_row());
        }
        std::ptr::addr_of_mut!((*p).counter_lo).write(draw_row());
        std::ptr::addr_of_mut!((*p).counter_hi).write([0, 0, 0, 0]);
        std::ptr::addr_of_mut!((*p).block_len).write([64, 64, 64, 64]);
        std::ptr::addr_of_mut!((*p).flags).write([11, 11, 11, 11]);
        out.assume_init_ref()
    }
}


/// Full fail-closed oracle for the ranked word-major generator. This is called
/// only during the protected worker's untimed warm-up, after `blocks` has
/// already been byte-matched to the protected generator.
#[cfg(test)]
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

/// Fill `out` with the blocks the protected generator would produce.
fn fill_compressions_par(out: &mut [Compression], log2_size: u32, seed: u64) {
    fill_compressions_from_init(out, generator_init(log2_size, seed));
}

fn fill_compressions_from_init(out: &mut [Compression], init: u64) {
    // Resolve the ILP/scalar choice once for the whole fill (getenv is far
    // too slow for the per-block loop).
    let ilp = !gen_block_ilp_killed();
    // 4096 blocks ≈ 448 KiB per task: large enough that the RNG chain
    // dominates task overhead, small enough to keep all workers fed.
    out.par_chunks_mut(4096)
        .enumerate()
        .for_each(|(chunk_index, dst)| {
            let base = chunk_index * 4096;
            for (offset, slot) in dst.iter_mut().enumerate() {
                *slot = gen_block_with(init, base + offset, ilp);
            }
        });
}

/// Select the existing machine warm-up policy without reading any input.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Literal transcription of `flock_benchmark_common::generate_compressions`
    /// and its `Rng`, so the parallel form is checked against the protected
    /// definition rather than against itself.
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

    /// The ranked size is the one that actually ships; check it exactly once.
    #[test]
    fn seed_pipe_matches_reference_at_ranked_size() {
        let seed = 0x1234_5678_9ABC_DEF0;
        assert_eq!(generate_compressions_par(18, seed), reference(18, seed));
    }

    /// The lazy witgen fallback regenerates single blocks via `gen_block`; pin
    /// it against the parallel fill (itself pinned against the reference), at
    /// every position class the quad loop can ask for.
    #[test]
    fn gen_block_matches_the_parallel_fill_at_every_index() {
        let (log2, seed) = (13u32, 0xDEAD_BEEF_1234_5678u64);
        let all = generate_compressions_par(log2, seed);
        let init = generator_init(log2, seed);
        // Cross a 4096-block task boundary (the eager fill's chunk size) and
        // both endpoints — for the dispatching entry AND both explicit
        // variants (item C: the ILP form must be draw-exact with the scalar
        // stepped form everywhere).
        for &i in &[0usize, 1, 7, 8, 4095, 4096, all.len() - 2, all.len() - 1] {
            assert_eq!(gen_block(init, i), all[i], "block {i}");
            assert_eq!(gen_block_scalar(init, i), all[i], "scalar block {i}");
            assert_eq!(gen_block_ilp(init, i), all[i], "ilp block {i}");
        }
        // Dense scalar-vs-ILP sweep across many seeds and every index class
        // mod the 4-wide unroll, plus large indices (closed-form multiply
        // overflow territory).
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

    /// The in-place word-major generator against the by-value one: every quad
    /// of the ranked 2^18-block input for several inits, plus quads whose
    /// state offset wraps. The destination is filled with a stale pattern
    /// first, so a field the in-place form failed to write would show.
    #[test]
    fn gen_quad_soa_into_matches_gen_quad_soa() {
        let check = |init: u64, first: usize| {
            let want = gen_quad_soa(init, first);
            let mut slot = std::mem::MaybeUninit::<CompressionQuadSoa>::uninit();
            // SAFETY: plain u32 rows, every byte pattern is a valid value.
            unsafe { std::ptr::write_bytes(slot.as_mut_ptr(), 0xA5, 1) };
            let got = gen_quad_soa_into(init, first, &mut slot);
            assert!(
                got.cv == want.cv
                    && got.message == want.message
                    && got.counter_lo == want.counter_lo
                    && got.counter_hi == want.counter_hi
                    && got.block_len == want.block_len
                    && got.flags == want.flags,
                "init={init:#x} first={first}"
            );
        };
        for &(log2, seed) in &[
            (18u32, 0u64),
            (18, u64::MAX),
            (18, 0x0123_4567_89AB_CDEF),
            (32, 0x9E37_79B9_7F4A_7C15),
        ] {
            let init = generator_init(log2, seed);
            for quad in 0..(1usize << 16) {
                check(init, 4 * quad);
            }
        }
        let top = (usize::MAX / DRAWS_PER_BLOCK) & !3;
        for init in [0u64, 1, u64::MAX, GOLDEN.wrapping_neg()] {
            for first in [0usize, 4, 1 << 18, (1 << 18) - 4, top - 4, top] {
                check(init, first);
            }
        }
    }

}
