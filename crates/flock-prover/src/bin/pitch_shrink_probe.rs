//! Pitch-ledger probe: how much of the lincheck z_lincheck stripe is dead
//! weight on the padded fold path?
//!
//! Evidence chain (all paths read, this iteration):
//! - `r1cs_hashes/common.rs:371` allocates `(n_total/8)*k` bytes for the
//!   stripe buffer — full 16 KiB pitch per stripe.
//! - the transpose (`common.rs:446-458`) writes only `useful_words*64` B
//!   per stripe (15424 for Blake3), never the tail.
//! - every padded fold reader bounds the useful region:
//!   portable `lincheck.rs:706` reads `stripe[..useful_bits]`, the NEON
//!   kernels (`lincheck/kernels/aarch64.rs:88`) and x86 tiled kernel
//!   (`kernels/x86_64.rs:99`) round up to 8-row BLOCK_K blocks.
//!
//! This probe measures (a) byte-identity of the fold across honest-padded
//! vs garbage-tail stripes — proving the tail is never read — and (b) the
//! wall-time ratio dense-useful vs padded-useful across a SIZE SWEEP
//! (m = 20 → 128 KiB buffer, L2-resident; m = 24 → 2 MiB; m = 28 → 32 MiB,
//! past the last-level cache on most hosts). The sweep answers whether the
//! small-scale savings ratio survives into the DRAM-bound regime the ranked
//! m=32 prove lives in, instead of assuming the L2-resident point transfers.

use flock_prover::field::F128;
use flock_prover::lincheck::partial_fold_packed_z_fast_padded;
use flock_prover::r1cs_hashes::blake3::USEFUL_BITS;
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// Build a deterministic (dense, padded) buffer pair plus eq_outer for one
/// size. The two buffers share byte-for-byte content on `[0, useful_bits)`;
/// the dense tail is random, the padded tail is zero.
fn make_buffers(m: usize, k_log: usize, useful_bits: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<F128>) {
    let k = 1usize << k_log;
    let n_outer = 1usize << (m - k_log);
    let n_stripes = n_outer / 8;
    let z_len = n_stripes * k;
    let mut rng = Rng(seed);
    let eq_outer: Vec<F128> = (0..n_outer)
        .map(|_| {
            let hi = rng.next();
            let lo = rng.next();
            F128::new(hi, lo)
        })
        .collect();
    let mut dense = vec![0u8; z_len];
    let mut padded = vec![0u8; z_len];
    for (s_d, s_p) in dense.chunks_mut(k).zip(padded.chunks_mut(k)) {
        for i in 0..k {
            let v = (rng.next() & 0xFF) as u8;
            if i < useful_bits {
                s_d[i] = v;
                s_p[i] = v;
            } else {
                s_d[i] = v; // dense path legitimately reads the tail
                s_p[i] = 0; // padded reader must never see it
            }
        }
    }
    (dense, padded, eq_outer)
}

/// Min wall time over `reps` runs of one fold configuration.
fn min_fold_time(
    z: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq: &[F128],
    reps: usize,
) -> f64 {
    let mut tmin = f64::MAX;
    for _ in 0..reps {
        let t0 = Instant::now();
        let _ = partial_fold_packed_z_fast_padded(z, m, k_log, useful_bits, eq);
        tmin = tmin.min(t0.elapsed().as_secs_f64());
    }
    tmin
}

fn main() {
    const K_LOG: usize = 14; // k = 16384, matches Blake3 ranked k_log
    const M: usize = 20; // identity/timing anchor size (128 KiB buffer)
    let k = 1usize << K_LOG;

    let pitch64 = USEFUL_BITS.div_ceil(64) * 64;
    let pitch8 = USEFUL_BITS.div_ceil(8) * 8;
    assert!(USEFUL_BITS <= pitch8 && pitch8 <= pitch64 && pitch64 <= k);
    let tail_alloc = k - pitch64;
    let tail_xpose = pitch64 - pitch8;
    eprintln!(
        "[pitch] k={k} useful_bits={USEFUL_BITS} pitch64={pitch64} pitch8={pitch8} \
         tail_alloc/stripe={tail_alloc} tail_xpose/stripe={tail_xpose}"
    );

    // Hypothetical ranked-domain totals (m ∈ {26, 29, 32} → stripe counts).
    for m in [26usize, 29, 32] {
        let n_rows = (1usize << m) / 8 / k;
        let full_surface_mib = (n_rows * k) as f64 / (1024.0 * 1024.0);
        let useful_surface_mib = (n_rows * pitch8) as f64 / (1024.0 * 1024.0);
        let elide_alloc_mib = tail_alloc as f64 * n_rows as f64 / (1024.0 * 1024.0);
        let elide_xpose_mib = tail_xpose as f64 * n_rows as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[ledger] m={m} n_rows(stripes)={n_rows} full_surface_MiB={full_surface_mib:.3} \
             useful_surface_MiB={useful_surface_mib:.3} elidable_alloc_MiB={elide_alloc_mib:.3} \
             elidable_xpose_MiB={elide_xpose_mib:.3} total={:.3}",
            elide_alloc_mib + elide_xpose_mib,
        );
    }

    // Production reality (common.rs:446-462): the transpose in the live
    // witness builder already writes only `pitch64` per stripe in release
    // builds (`cfg(test)`-gated tail memset), so `elidable_xpose` is zero on
    // the timed path, and the scratch pool (`take_u8`/`give_u8`, common.rs:368-371)
    // makes the `elidable_alloc` tail page-resident rather than DRAM traffic.
    // What remains is the DOUBLE TOUCH: transpose writes pitch64·n_rows, then
    // the padded fold re-reads pitch8·n_rows in a later pass.
    eprintln!("[fusion-ledger] remaining stripe cost is the double touch (transpose-write + fold-read):");
    for m in [26usize, 29, 32] {
        let n_rows = (1usize << m) / 8 / k;
        let write_mib = (n_rows * pitch64) as f64 / (1024.0 * 1024.0);
        let read_mib = (n_rows * pitch8) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[fusion-ledger] m={m} transpose_write_MiB={write_mib:.3} fold_read_MiB={read_mib:.3} \
             fusion_upper_bound_MiB={read_mib:.3} (fold-side re-read, only if the fold's \
             outer-dim parallelization can ride the transpose's byte-position pass)"
        );
    }

    // Identity at the anchor size: padded fold must equal garbage-tail fold
    // byte-for-byte (tail never read), and both must differ from the dense
    // fold (which reads the tail).
    {
        let n_outer = 1usize << (M - K_LOG);
        let n_stripes = n_outer / 8;
        let z_len = n_stripes * k;
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let eq_outer: Vec<F128> = (0..n_outer)
            .map(|_| {
                let hi = rng.next();
                let lo = rng.next();
                F128::new(hi, lo)
            })
            .collect();
        let mut padded = vec![0u8; z_len];
        let mut garb_tail = vec![0u8; z_len];
        let mut dense = vec![0u8; z_len];
        for (s_p, (s_g, s_d)) in padded
            .chunks_mut(k)
            .zip(garb_tail.chunks_mut(k).zip(dense.chunks_mut(k)))
        {
            for (i, ((bp, bg), bd)) in s_p
                .iter_mut()
                .zip(s_g.iter_mut())
                .zip(s_d.iter_mut())
                .enumerate()
            {
                let v = (rng.next() & 0xFF) as u8;
                if i < USEFUL_BITS {
                    *bp = v;
                    *bg = v;
                    *bd = v;
                } else if i >= pitch64 {
                    *bp = 0;
                    *bg = 0xFF; // sentinel: any reader of the alloc tail differs
                    *bd = v;
                } else {
                    *bp = 0;
                    *bg = 0xFF; // transpose tail [pitch8, pitch64)
                    *bd = v;
                }
            }
        }
        let fp = partial_fold_packed_z_fast_padded(&padded, M, K_LOG, USEFUL_BITS, &eq_outer);
        let fg = partial_fold_packed_z_fast_padded(&garb_tail, M, K_LOG, USEFUL_BITS, &eq_outer);
        let fd = partial_fold_packed_z_fast_padded(&dense, M, K_LOG, k, &eq_outer);
        let same_pg = fp == fg;
        let diff_pd = fp != fd;
        eprintln!(
            "[identity] padded==garbage_tail: {same_pg} (proves alloc tail never read) \
             padded!=dense: {diff_pd} (proves dense/padded differ, reader bound real)"
        );
        assert!(same_pg, "padded fold changed with garbage tail — tail IS read!");
        assert!(diff_pd, "padded and dense folds identical — padded bound inactive!");
        println!(
            "PITCH_PROBE_RESULT k={k} useful_bits={USEFUL_BITS} pitch64={pitch64} \
             pitch8={pitch8} same_pg={same_pg} diff_pd={diff_pd}"
        );
    }

    // Scaling sweep: does the dense-vs-padded savings ratio hold as the
    // buffer leaves L2 and becomes DRAM traffic?
    println!("PITCH_SCALE_RESULT");
    // m=30 (128 MiB) and m=32 (512 MiB, the ranked fold size) extend the
    // sweep into the DRAM regime the ranked prove lives in.
    for (m, reps) in [(20usize, 200usize), (24, 60), (28, 15), (30, 6), (32, 3)] {
        let (dense, padded, eq) = make_buffers(m, K_LOG, USEFUL_BITS, 0x9E37_79B9_7F4A_7C15);
        let t_dense = min_fold_time(&dense, m, K_LOG, k, &eq, reps);
        let t_pad = min_fold_time(&padded, m, K_LOG, USEFUL_BITS, &eq, reps);
        let n_stripes = (1usize << (m - K_LOG)) / 8;
        let bytes_dense = (n_stripes * k) as f64;
        let bytes_pad = (n_stripes * pitch8) as f64;
        println!(
            "PITCH_SCALE m={m} z_MiB={:.1} dense_min_s={t_dense:.6} padded_min_s={t_pad:.6} \
             time_ratio={:.4} byte_ratio={:.4}",
            bytes_dense / (1024.0 * 1024.0),
            t_dense / t_pad,
            bytes_dense / bytes_pad,
        );
    }
}
