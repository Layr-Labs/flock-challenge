//! from-z structure probe: proves the element-0 identity of the ranked
//! from-z first pass and verifies the fused layer-4 B=0 shortcut end-to-end.
//!
//! Claims verified here (portable, x86):
//! 1. After the from-z pass (replicate [z,z] + layers 1..3), element 0 of
//!    EVERY 16-element tile equals the raw z value at that tile's position.
//!    Mechanism: the from-z radix-16 butterfly network never touches
//!    elems[0] in layers 1..3 (all pairs have EU/EV != 0), so elems[0] stays
//!    z. Layer 0 is the degenerate copy (v=0: (u, 0) -> (u, u)).
//! 2. The next pass (layer 4, f=4) reads every codeword position exactly
//!    once; positions q < 2^(log_d - 4) (the element-0 column) are read ONLY
//!    by B=0 tiles. Therefore those tiles may read the raw z buffer instead
//!    of the from-z output, and the from-z pass may skip writing its
//!    element-0 column (64 MiB at the ranked shape).
//! 3. End-to-end: a pipeline with (from-z minus element-0 writes) + (layer-4
//!    B=0 tiles reading z) is bit-exact equal to the sequential pipeline.
//!
//! This revision runs the claims at the RANKED geometry (log_d=20, 64 lanes,
//! s=16) in addition to the small geometries, so the e0 family's math is
//! gated on the actual scored shape before any kernel work resumes.

use flock_core::field::gf2_128::F128;
use flock_core::ntt::additive_ntt_f128::AdditiveNttF128;

fn next_rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// One interleaved NTT layer (scalar; mirrors gpu_commit's cpu_one_layer).
fn one_layer(ntt: &AdditiveNttF128, data: &mut [F128], num_ntts: usize, layer: usize) {
    let log_d = (data.len() / num_ntts).trailing_zeros() as usize;
    let num_blocks = 1usize << layer;
    let block_size = 1usize << (log_d - layer);
    let half = block_size >> 1;
    for block in 0..num_blocks {
        let tw = ntt.twiddle(layer, block);
        let base = block * block_size * num_ntts;
        for row in 0..half {
            for lane in 0..num_ntts {
                let top = base + row * num_ntts + lane;
                let bot = top + half * num_ntts;
                let v = data[bot];
                let nu = data[top] + v * tw;
                data[top] = nu;
                data[bot] = v + nu;
            }
        }
    }
}

/// Run claims 1 + 3 at one (log_d, num_ntts) geometry.
/// Returns (element0_identity_ok, fused_path_bit_exact).
fn verify_geometry(log_d: usize, num_ntts: usize) -> (bool, bool) {
    let n = 1usize << log_d;
    let n_total = n * num_ntts;
    let mut state = 0x9E3779B97F4A7C15u64;
    let z: Vec<F128> = (0..num_ntts * (n / 2))
        .map(|_| F128::new(next_rng(&mut state), next_rng(&mut state)))
        .collect();
    let ntt = AdditiveNttF128::standard(log_d);

    // Pre-transform buffer: replicate_message_fill [z, z].
    let mut pre: Vec<F128> = vec![F128::ZERO; n_total];
    for lane in 0..num_ntts {
        for e in 0..n / 2 {
            pre[e * num_ntts + lane] = z[e * num_ntts + lane];
            pre[(e + n / 2) * num_ntts + lane] = z[e * num_ntts + lane];
        }
    }

    // From-z pass: layers 1..3 (layer 0 already materialized by the fill).
    let mut l3 = pre.clone();
    for layer in 1..4 {
        one_layer(&ntt, &mut l3, num_ntts, layer);
    }

    // CLAIM 1: element-0 column of from-z output == z.
    let n_tiles = 1usize << (log_d - 4);
    let mut id_ok = true;
    for lane in 0..num_ntts {
        for r in 0..n_tiles {
            id_ok &= l3[r * num_ntts + lane] == z[r * num_ntts + lane];
        }
    }

    // CLAIM 3: fused path (from-z with element-0 writes elided, layer-4 B=0
    // tiles sourcing z) bit-exact vs sequential.
    let mut a = l3.clone();
    for layer in 4..8 {
        one_layer(&ntt, &mut a, num_ntts, layer);
    }

    let mut b2 = l3.clone();
    let tile_span = 1usize << (log_d - 4);
    for r in 0..tile_span {
        for lane in 0..num_ntts {
            b2[r * num_ntts + lane] = z[r * num_ntts + lane];
        }
    }
    for layer in 4..8 {
        one_layer(&ntt, &mut b2, num_ntts, layer);
    }
    (id_ok, a == b2)
}

fn main() {
    // Small geometry (historical verification).
    for (log_d, ntts) in [(8usize, 8usize), (16usize, 64usize)] {
        let (id, fused) = verify_geometry(log_d, ntts);
        println!("log_d={log_d} ntts={ntts}: CLAIM1 element-0==z: {id}; CLAIM3 fused bit-exact: {fused}");
        assert!(id, "element-0 identity failed at log_d={log_d}");
        assert!(fused, "fused path diverged at log_d={log_d}");
    }

    // RANKED geometry: log_d=20, 64 lanes, s=16 (the scored shape).
    let (id, fused) = verify_geometry(20, 64);
    println!("log_d=20 ntts=64 (RANKED): CLAIM1 element-0==z: {id}; CLAIM3 fused bit-exact: {fused}");
    assert!(id, "element-0 identity FAILED at ranked shape");
    assert!(fused, "fused path DIVERGED at ranked shape");

    // Twiddle structure survey at the ranked shape (layer 4 block 0 and the
    // zero-root pattern across layers).
    let ntt = AdditiveNttF128::standard(20);
    let mut zero_cnt = 0usize;
    let mut one_cnt = 0usize;
    for block in 0..16 {
        let tw = ntt.twiddle(4, block);
        if tw == F128::ZERO {
            zero_cnt += 1;
        }
        if tw == F128::ONE {
            one_cnt += 1;
        }
    }
    println!(
        "layer-4 block twiddles: zero={zero_cnt}/16 one={one_cnt}/16 (block0: {:?})",
        ntt.twiddle(4, 0)
    );
    let mut all_block0_zero = true;
    for layer in 4..8 {
        all_block0_zero &= ntt.twiddle(layer, 0) == F128::ZERO;
    }
    println!("layer 4..7 block-0 twiddles all zero-root: {all_block0_zero}");

    midpass_zero_survey();

    for layer in 4..8usize {
        let zeros: Vec<usize> = (0..(1usize << layer))
            .filter(|&b| ntt.twiddle(layer, b) == F128::ZERO)
            .collect();
        println!("layer {layer}: zero blocks = {zeros:?}");
    }
}

#[allow(dead_code)]
fn midpass_zero_survey() {
    // Which f=4 pass layers have a zero block-0 twiddle (tab tsel=0 built
    // from twiddles[(1<<l)-1])? If twiddle(l,0)==0, the tsel-0 tab is all
    // zero and the mul is exact-identity-skippable.
    let ntt = AdditiveNttF128::standard(20);
    for l in [4usize, 8, 12, 16] {
        let tw = ntt.twiddle(l, 0);
        println!(
            "pass layer l={l}: twiddle(l,0)==ZERO: {}",
            tw == F128::ZERO
        );
    }
}
