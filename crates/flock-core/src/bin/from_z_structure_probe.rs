//! From-z structure probe (portable, x86-runnable).
//!
//! Question: the GPU ranked from-z first pass computes layers 0..3 of the
//! additive NTT of the rate-1/2 coefficient vector [z, 0, ..., 0]. The
//! incumbent from-z kernels read z (512 MiB) and write the full post-layer-3
//! codeword (1 GiB); the layer-4 pass then re-reads that 1 GiB.
//!
//! If the post-layer-3 output at element e of a tile equals the raw z value
//! at that tile position (i.e. the 4-layer network degenerates to a copy for
//! the element the layer-4 pass consumes), the whole 1 GiB intermediate
//! write+read (2 GiB of DRAM traffic) can be eliminated by fusing the from-z
//! pass into the layer-4 pass.
//!
//! This probe applies the CPU pipeline's real layer transforms to a
//! rate-1/2-from-message buffer at small scale and reports the per-element
//! relation between the layer-3 output and the input message. Runs on x86;
//! no Metal involved.

use flock_core::field::F128;
use flock_core::ntt::additive_ntt_f128::AdditiveNttF128;

fn main() {
    // Small ranked-like geometry: num_ntts = 8 lanes, log_d = 4 (16 elements
    // per lane), rate 1/2 (log_inv_rate = 1).
    let log_d = 4usize;
    let num_ntts = 8usize;
    let n = 1usize << log_d; // 16 elements per lane
    let n_total = n * num_ntts;

    // Deterministic pseudo-random message z: 8 elements per lane (rate-1/2:
    // the message occupies positions 0..7 of each 16-element block).
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let z: Vec<F128> = (0..num_ntts * (n / 2))
        .map(|_| F128::new(next(), next()))
        .collect();

    // Codeword buffer: [z, z] (layer 0 is the degenerate copy: message
    // replicated into the zero-padded upper half), then layers 1..3.
    let mut data: Vec<F128> = vec![F128::ZERO; n_total];
    for lane in 0..num_ntts {
        for e in 0..n / 2 {
            data[e * num_ntts + lane] = z[e * num_ntts + lane];
            data[(e + n / 2) * num_ntts + lane] = z[e * num_ntts + lane];
        }
    }

    let ntt = AdditiveNttF128::standard(log_d);
    // Apply layers 1..3 (layer 0 is the copy already materialized above).
    ntt.forward_transform_interleaved_from_layer(&mut data, num_ntts, 1);

    // Report: is post-layer-3 element 0 at each lane equal to z[lane]?
    // Layout: element e of lane l lives at data[e * num_ntts + l].
    println!("=== from-z structure probe (log_d={log_d}, num_ntts={num_ntts}) ===");
    let mut any_identity = true;
    for lane in 0..num_ntts {
        let e0 = data[0 * num_ntts + lane];
        let zl = z[lane];
        let same = e0 == zl;
        any_identity &= same;
        println!(
            "lane {lane}: layer3[e=0] == z? {same}  (layer3[e=0]={e0:?}, z={zl:?})"
        );
    }
    println!("ALL elements e=0 equal z: {any_identity}");

    // Also check every element e in 0..16: is layer3[e] a fixed simple
    // function? Report the full layer-3 row for lane 0.
    let lane = 0usize;
    print!("lane {lane} layer-3 row: ");
    for e in 0..n {
        if e > 0 {
            print!(", ");
        }
        let v = data[e * num_ntts + lane];
        // Compact hex print (first 4 bytes).
        let bytes = v.hi.to_le_bytes();
        print!("e{e}={:02x}{:02x}{:02x}{:02x}", bytes[0], bytes[1], bytes[2], bytes[3]);
    }
    println!();
}
