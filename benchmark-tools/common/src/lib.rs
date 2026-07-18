pub type RawCompression = ([u32; 8], [u32; 16], u64, u32, u32);
pub const DOMAIN: &[u8] = b"flock-bench-v0";

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
}

pub fn generate_compressions(log2_size: u32, seed: u64) -> Vec<RawCompression> {
    let count = 1usize
        .checked_shl(log2_size)
        .expect("log2_size exceeds usize width");
    let mut rng = Rng::new(seed ^ u64::from(log2_size).rotate_left(29));
    (0..count)
        .map(|_| {
            let cv = std::array::from_fn(|_| rng.next_u32());
            let message = std::array::from_fn(|_| rng.next_u32());
            let counter = u64::from(rng.next_u32());
            (cv, message, counter, 64, 11)
        })
        .collect()
}
