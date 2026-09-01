pub type RawCompression = ([u32; 8], [u32; 16], u64, u32, u32);
pub const DOMAIN: &[u8] = b"flock-bench-v0";

struct Rng(u64);

impl Rng {
    /// Additive step of the state walk. It is odd, so the walk is a bijection
    /// on u64 and jump-ahead is exact: the state before draw k is s0 + k*STEP.
    const STEP: u64 = 0x9E37_79B9_7F4A_7C15;

    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(Self::STEP);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
}

/// Rng draws per compression: 8 cv + 16 message + 1 counter.
const DRAWS_PER_COMPRESSION: u64 = 25;
/// Below this many compressions the parallel path is not worth its overhead.
const PAR_MIN_CHUNK: usize = 4096;

/// Fill `out` with compressions starting at compression index `first_compression`.
///
/// The sequential generator's state before draw k is exactly
/// `start_state + k * STEP` (wrapping, STEP odd), because `next_u32` only
/// ever advances the state by a wrapping add. Chunks therefore reproduce the
/// original stream bit-for-bit while being independent of each other.
#[inline]
fn fill_slice(out: &mut [RawCompression], first_compression: usize, start_state: u64) {
    let jumps = (first_compression as u64).wrapping_mul(DRAWS_PER_COMPRESSION);
    let mut rng = Rng(start_state.wrapping_add(jumps.wrapping_mul(Rng::STEP)));
    for slot in out.iter_mut() {
        *slot = (
            std::array::from_fn(|_| rng.next_u32()),
            std::array::from_fn(|_| rng.next_u32()),
            u64::from(rng.next_u32()),
            64,
            11,
        );
    }
}

pub fn generate_compressions(log2_size: u32, seed: u64) -> Vec<RawCompression> {
    let count = 1usize
        .checked_shl(log2_size)
        .expect("log2_size exceeds usize width");
    let start_state = seed ^ u64::from(log2_size).rotate_left(29);

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let n_chunks = (count / PAR_MIN_CHUNK).max(1).min(threads);
    if n_chunks <= 1 {
        // Serial path, byte-identical to the original implementation.
        let mut rng = Rng::new(start_state);
        return (0..count)
            .map(|_| {
                let cv = std::array::from_fn(|_| rng.next_u32());
                let message = std::array::from_fn(|_| rng.next_u32());
                let counter = u64::from(rng.next_u32());
                (cv, message, counter, 64, 11)
            })
            .collect();
    }

    let per_chunk = count.div_ceil(n_chunks);
    let mut out = Vec::with_capacity(count);
    out.resize_with(count, || ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32));
    std::thread::scope(|s| {
        let mut rest: &mut [RawCompression] = out.as_mut_slice();
        for t in 0..n_chunks {
            let first = t * per_chunk;
            let len = per_chunk.min(count - first);
            let (mine, tail) = rest.split_at_mut(len);
            rest = tail;
            s.spawn(move || fill_slice(mine, first, start_state));
        }
    });
    out
}
