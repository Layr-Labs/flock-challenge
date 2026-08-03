//! Property probe: the subset-sum nibble-table construction is bit-exact
//! against the incumbent Horner loop for every nibble n in 0..16 and random
//! bases — the same identity the MSL `DEF_BUILD_NIBBLE_BLOCK` relies on.
//! Run: cargo run --release --bin subset_sum_probe

fn gf_mulx(v: [u32; 4]) -> [u32; 4] {
    // Mirrors the MSL gf_mulx: multiply by x in the additive field, limb-shift
    // with the two-bit fold-back on the high limb.
    let h = v[3] >> 30;
    let r = [
        (v[0] << 2) ^ ((h << 7) ^ (h << 2) ^ (h << 1) ^ h),
        (v[1] << 2) | (v[0] >> 30),
        (v[2] << 2) | (v[1] >> 30),
        (v[3] << 2) | (v[2] >> 30),
    ];
    r
}

fn xor4(a: [u32; 4], b: [u32; 4]) -> [u32; 4] {
    [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
}

fn horner(base: [u32; 4], n: usize) -> [u32; 4] {
    let mut p = base;
    let mut val = [0u32; 4];
    for k in 0..4 {
        if (n >> k) & 1 == 1 {
            val = xor4(val, p);
        }
        p = gf_mulx(p);
    }
    val
}

fn subset(base: [u32; 4], n: usize) -> [u32; 4] {
    let b0 = base;
    let b1 = gf_mulx(b0);
    let b2 = gf_mulx(b1);
    let b3 = gf_mulx(b2);
    let t = [
        [0u32; 4],
        b0,
        b1,
        xor4(b0, b1),
        b2,
        xor4(b0, b2),
        xor4(b1, b2),
        xor4(xor4(b0, b1), b2),
        b3,
        xor4(b0, b3),
        xor4(b1, b3),
        xor4(xor4(b0, b1), b3),
        xor4(b2, b3),
        xor4(xor4(b0, b2), b3),
        xor4(xor4(b1, b2), b3),
        xor4(xor4(xor4(b0, b1), b2), b3),
    ];
    t[n]
}

fn main() {
    // xorshift64* rng
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        s.wrapping_mul(0x2545F4914F6CDD1D)
    };
    let mut mismatches = 0usize;
    let mut checked = 0usize;
    for _ in 0..200_000 {
        let base = [next() as u32, (next() >> 32) as u32, next() as u32, (next() >> 32) as u32];
        for n in 0..16usize {
            let h = horner(base, n);
            let t = subset(base, n);
            checked += 1;
            if h != t {
                mismatches += 1;
                if mismatches <= 3 {
                    println!("MISMATCH base={base:?} n={n} horner={h:?} subset={t:?}");
                }
            }
        }
    }
    println!("checked={checked} mismatches={mismatches}");
    if mismatches == 0 {
        println!("SUBSET-SUM == HORNER: bit-exact");
    } else {
        std::process::exit(1);
    }
}
