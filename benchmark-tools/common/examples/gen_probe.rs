//! Micro-bench: reference-serial vs parallel generate_compressions at a log2.
//! Run: cargo run --profile challenge --example gen_probe -- [LOG2] [REPS]
//! The serial arm replicates the pre-edit loop bit-for-bit so both arms are
//! measured in one process (same host, same scheduler state).

use std::time::Instant;

use flock_benchmark_common::{generate_compressions, RawCompression};

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

fn serial_gen(log2_size: u32, seed: u64) -> Vec<RawCompression> {
    let count = 1usize.checked_shl(log2_size).expect("log2 too wide");
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

fn main() {
    let log2: u32 = std::env::args().nth(1).unwrap_or("18".into()).parse().unwrap();
    let reps: u32 = std::env::args().nth(2).unwrap_or("5".into()).parse().unwrap();
    // Verify both arms agree bit-for-bit.
    let a = serial_gen(log2, 0xDEADBEEF);
    let b = generate_compressions(log2, 0xDEADBEEF);
    assert_eq!(a, b, "parallel arm diverges from serial reference");
    println!("equality check ok: {} compressions", a.len());
    for rep in 0..reps {
        let t = Instant::now();
        let v = serial_gen(log2, 0x424242 + u64::from(rep));
        let ser = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let w = generate_compressions(log2, 0x424242 + u64::from(rep));
        let par = t.elapsed().as_secs_f64();
        assert_eq!(v, w);
        println!(
            "rep{rep} serial={ser:.4}s parallel={par:.4}s speedup={:.2}x",
            ser / par
        );
    }
}
