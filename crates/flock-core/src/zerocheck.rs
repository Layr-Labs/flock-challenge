use crate::challenger::Challenger;
use crate::field::{F8, F128};
use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use serde::{Deserialize, Serialize};

pub mod multilinear;
pub mod univariate_skip;
pub mod univariate_skip_deg4;
pub mod univariate_skip_deg4_optimized;
pub mod univariate_skip_optimized;

use multilinear::{
    UniSkipFoldTable, fold_2vars_and_compute_round_lookahead_into,
    fold_2vars_and_compute_round_pair_into, fold_and_compute_round_pair_into, fold_in_place_pair,
    interpolate_at_z_combined, interpolate_at_z_on_lambda, round_pair_naive,
    uni_skip_fold_2vars_into_and_round45_lookahead_packed_padded,
    uni_skip_fold_and_round_pair_optimized_packed_padded,
    uni_skip_round2_and_round3_lookahead_packed_padded,
};
use univariate_skip_optimized::{
    c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed_padded,
    small_challenges_ghash,
};

pub const K_SKIP: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct PaddingSpec {
    pub k_log: usize,
    pub useful_bits_per_block: usize,
}

impl PaddingSpec {
    pub fn dense(m: usize) -> Self {
        Self {
            k_log: m,
            useful_bits_per_block: 1usize << m,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZerocheckClaim {
    pub z: F128,
    pub mlv_challenges: Vec<F128>,
    pub r_rest: Vec<F128>,
    pub a_eval: F128,
    pub b_eval: F128,
    pub c_eval: F128,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZerocheckProof {
    pub round1_ab: Vec<F128>,
    pub round1_c: Vec<F128>,
    pub multilinear_rounds: Vec<(F128, F128)>,
    pub final_a_eval: F128,
    pub final_b_eval: F128,
    pub final_c_eval: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    LogNTooSmall { log_n: usize, k_skip: usize },
    BadRound1Length { expected: usize, got: usize },
    BadMultilinearRoundsLength { expected: usize, got: usize },
    CEvalMismatch,
    SumcheckFinalFailed,
}

pub fn prove_packed<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    prove_packed_padded(
        a_packed,
        b_packed,
        c_packed,
        m,
        &PaddingSpec::dense(m),
        challenger,
    )
}

pub fn prove_packed_padded<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim) {
    let (proof, claim, _) = prove_packed_padded_inner(
        PackedAb::Slice(a_packed, b_packed),
        c_packed,
        None,
        m,
        padding,
        false,
        challenger,
    );
    (proof, claim)
}

pub fn prove_packed_padded_capture_s_hat_v_c<C: Challenger>(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Vec<F128>) {
    let (proof, claim, captured) = prove_packed_padded_inner(
        PackedAb::Slice(a_packed, b_packed),
        c_packed,
        None,
        m,
        padding,
        true,
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

pub fn prove_f128_packed_padded_capture_s_hat_v_c<C: Challenger>(
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    c_packed: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Vec<F128>) {
    let (proof, claim, captured) = prove_packed_padded_inner(
        PackedAb::OwnedF128(a_packed_f128, b_packed_f128),
        c_packed,
        None,
        m,
        padding,
        true,
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

pub fn prove_f128_packed_padded_capture_fold8_c<C: Challenger>(
    a_packed_f128: Vec<F128>,
    b_packed_f128: Vec<F128>,
    c_packed: &[u8],
    z_packed_lincheck: &[u8],
    m: usize,
    padding: &PaddingSpec,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Vec<F128>) {
    let (proof, claim, captured) = prove_packed_padded_inner(
        PackedAb::OwnedF128(a_packed_f128, b_packed_f128),
        c_packed,
        Some(z_packed_lincheck),
        m,
        padding,
        true,
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce s_hat_v_c"),
    )
}

enum PackedAb<'a> {
    Slice(&'a [u8], &'a [u8]),
    OwnedF128(Vec<F128>, Vec<F128>),
}

impl<'a> PackedAb<'a> {
    #[inline]
    fn as_byte_slices(&self) -> (&[u8], &[u8]) {
        match self {
            PackedAb::Slice(a, b) => (*a, *b),
            PackedAb::OwnedF128(a, b) => unsafe {
                (
                    std::slice::from_raw_parts(
                        a.as_ptr() as *const u8,
                        a.len() * core::mem::size_of::<F128>(),
                    ),
                    std::slice::from_raw_parts(
                        b.as_ptr() as *const u8,
                        b.len() * core::mem::size_of::<F128>(),
                    ),
                )
            },
        }
    }

    #[inline]
    fn recycle_if_owned(self) {
        if let PackedAb::OwnedF128(a, b) = self {
            crate::scratch::give_f128(a);
            crate::scratch::give_f128(b);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_packed_padded_inner<C: Challenger>(
    packed_ab: PackedAb<'_>,
    c_packed: &[u8],
    z_packed_lincheck: Option<&[u8]>,
    m: usize,
    padding: &PaddingSpec,
    capture_s_hat_v_c: bool,
    challenger: &mut C,
) -> (ZerocheckProof, ZerocheckClaim, Option<Vec<F128>>) {
    let (a_packed, b_packed) = packed_ab.as_byte_slices();
    let k_skip = K_SKIP;
    const N_INNER: usize = 7;
    assert!(
        m >= k_skip + N_INNER,
        "prove requires m >= k_skip + N_INNER (= {})",
        k_skip + N_INNER
    );
    let expected_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), expected_bytes);
    assert_eq!(b_packed.len(), expected_bytes);
    assert_eq!(c_packed.len(), expected_bytes);
    let n_mlv = m - k_skip;

    challenger.observe_label(b"flock-zerocheck-v0");

    let r_skip = challenger.sample_f128_vec(k_skip);
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    let zc_timing = std::env::var_os("FLOCK_ZC_TIMING").is_some();
    let t_round1 = std::time::Instant::now();
    static INV_TABLE_K6: std::sync::OnceLock<InvNttTableByteSingleGf8> = std::sync::OnceLock::new();
    let inv_table_owned;
    let inv_table: &InvNttTableByteSingleGf8 = if k_skip == 6 {
        INV_TABLE_K6.get_or_init(|| {
            let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(6, F8(1u8 << 6));
            InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
        })
    } else {
        let ntt_s = AdditiveNttGf8::new(k_skip, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(k_skip, F8(1u8 << k_skip));
        inv_table_owned = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        &inv_table_owned
    };
    let c_s = c_s_f128();
    let (round1_ab, round1_c, s_hat_v_c) = if let Some(z_lc) =
        z_packed_lincheck.filter(|_| capture_s_hat_v_c && padding.k_log >= k_skip + 7)
    {
        let round1_ab_opt =
            crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_ab_only_packed_padded(
                a_packed, b_packed, m, k_skip, &r, &inv_table, padding,
            );
        let round1_ab: Vec<F128> = round1_ab_opt.iter().map(|x| c_s * *x).collect();

        let eq_outer_c = crate::lincheck::build_eq_table(&r[padding.k_log..]);
        let c_vec = crate::lincheck::partial_fold_packed_z_best(
            z_lc,
            m,
            padding.k_log,
            padding.useful_bits_per_block,
            &eq_outer_c,
        );
        crate::scratch::give_f128(eq_outer_c);
        let s_hat_v_c_fold8 = crate::pcs::ring_switch::s_hat_v_fold8_from_z_vec(
            &c_vec,
            &r[k_skip + 1..padding.k_log],
        );
        crate::scratch::give_f128(c_vec);
        let s_hat_v_c_128 = crate::pcs::ring_switch::collapse_s_hat_v_fold8(
            &s_hat_v_c_fold8,
            &r[k_skip + 1..k_skip + 7],
        );
        let prefix = r[k_skip];
        let one_plus_prefix = F128::ONE + prefix;
        let ell = 1usize << k_skip;
        let mut res_c_s = vec![F128::ZERO; ell];
        for lane in 0..ell {
            res_c_s[lane] =
                one_plus_prefix * s_hat_v_c_128[lane] + prefix * s_hat_v_c_128[ell + lane];
        }
        let round1_c =
            crate::zerocheck::univariate_skip::ntt_extend_f128_vec_ghash(&res_c_s, &inv_table);
        (round1_ab, round1_c, Some(s_hat_v_c_fold8))
    } else if capture_s_hat_v_c {
        let (ab, c, s) =
            crate::zerocheck::univariate_skip_optimized::round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                a_packed,
                b_packed,
                c_packed,
                m,
                k_skip,
                &r,
                &inv_table,
                padding,
            );
        (
            ab.iter().map(|x| c_s * *x).collect(),
            c.iter().map(|x| c_s * *x).collect(),
            Some(s),
        )
    } else {
        let (ab, c) = round1_shift_reduce_extract_c_packed_padded(
            a_packed, b_packed, c_packed, m, k_skip, &r, &inv_table, padding,
        );
        (
            ab.iter().map(|x| c_s * *x).collect(),
            c.iter().map(|x| c_s * *x).collect(),
            None,
        )
    };
    if zc_timing {
        eprintln!(
            "[zc-timing] round1 URM: {:.2} ms",
            t_round1.elapsed().as_secs_f64() * 1e3
        );
    }

    challenger.observe_f128_slice(&round1_ab);
    challenger.observe_f128_slice(&round1_c);
    let z = challenger.sample_f128();

    let final_c_eval = interpolate_at_z_on_lambda(&round1_c, k_skip, z);

    let t_round2 = std::time::Instant::now();
    let fold_table = UniSkipFoldTable::new(k_skip, z);
    let mut mlv_arg = vec![F128::ONE; n_mlv];
    mlv_arg[1..].copy_from_slice(&r[k_skip + 1..]);
    let mut multilinear_msgs = Vec::with_capacity(n_mlv);
    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);

    let (mut a_mlv, mut b_mlv, mut a_nxt, mut b_nxt, start_i, t_tail) = match packed_ab {
        PackedAb::OwnedF128(a_vec, b_vec) if n_mlv >= 10 => {
            let n_quarter = a_vec.len() / 2;
            let mut a_out = crate::scratch::take_f128(n_quarter);
            let mut b_out = crate::scratch::take_f128(n_quarter);
            let (lookahead45, t_tail) = {
                let a_bytes = unsafe {
                    core::slice::from_raw_parts(
                        a_vec.as_ptr() as *const u8,
                        a_vec.len() * core::mem::size_of::<F128>(),
                    )
                };
                let b_bytes = unsafe {
                    core::slice::from_raw_parts(
                        b_vec.as_ptr() as *const u8,
                        b_vec.len() * core::mem::size_of::<F128>(),
                    )
                };
                let lookahead = uni_skip_round2_and_round3_lookahead_packed_padded(
                    a_bytes,
                    b_bytes,
                    m,
                    k_skip,
                    &fold_table,
                    &mlv_arg,
                    padding,
                );
                if zc_timing {
                    eprintln!(
                        "[zc-timing] round2+3 lookahead msg-only: {:.2} ms",
                        t_round2.elapsed().as_secs_f64() * 1e3
                    );
                }
                let t_tail = std::time::Instant::now();
                multilinear_msgs.push((lookahead.msg2_1, lookahead.msg2_inf));
                challenger.observe_f128(lookahead.msg2_1);
                challenger.observe_f128(lookahead.msg2_inf);
                let rho_0 = challenger.sample_f128();
                mlv_rhos.push(rho_0);

                let (msg3_1, msg3_inf) = lookahead.eval_round3(rho_0, F128::ONE);
                multilinear_msgs.push((msg3_1, msg3_inf));
                challenger.observe_f128(msg3_1);
                challenger.observe_f128(msg3_inf);
                let rho_1 = challenger.sample_f128();
                mlv_rhos.push(rho_1);

                let mut r_next4 = vec![F128::ONE; n_mlv - 2];
                r_next4[1..].copy_from_slice(&r[k_skip + 3..]);
                let lookahead45 = uni_skip_fold_2vars_into_and_round45_lookahead_packed_padded(
                    a_bytes,
                    b_bytes,
                    &mut a_out,
                    &mut b_out,
                    m,
                    k_skip,
                    &fold_table,
                    rho_0,
                    rho_1,
                    &r_next4,
                    padding,
                );
                if zc_timing {
                    eprintln!(
                        "[zc-timing] round4+5 fold_2vars+lookahead: {:.2} ms",
                        t_tail.elapsed().as_secs_f64() * 1e3
                    );
                }
                let t_tail2 = std::time::Instant::now();
                (lookahead45, t_tail2)
            };
            multilinear_msgs.push((lookahead45.msg2_1, lookahead45.msg2_inf));
            challenger.observe_f128(lookahead45.msg2_1);
            challenger.observe_f128(lookahead45.msg2_inf);
            let rho_2 = challenger.sample_f128();
            mlv_rhos.push(rho_2);

            let (msg5_1, msg5_inf) = lookahead45.eval_round3(rho_2, F128::ONE);
            multilinear_msgs.push((msg5_1, msg5_inf));
            challenger.observe_f128(msg5_1);
            challenger.observe_f128(msg5_inf);
            let _rho_3 = challenger.sample_f128();
            mlv_rhos.push(_rho_3);
            (a_out, b_out, a_vec, b_vec, 2usize, t_tail)
        }
        other => {
            let (a_bytes, b_bytes) = other.as_byte_slices();
            let (a_folded, b_folded, msg_1, msg_inf) =
                uni_skip_fold_and_round_pair_optimized_packed_padded(
                    a_bytes,
                    b_bytes,
                    m,
                    k_skip,
                    &fold_table,
                    &mlv_arg,
                    padding,
                );
            other.recycle_if_owned();
            if zc_timing {
                eprintln!(
                    "[zc-timing] round2 fused fold: {:.2} ms",
                    t_round2.elapsed().as_secs_f64() * 1e3
                );
            }
            let t_tail = std::time::Instant::now();
            multilinear_msgs.push((msg_1, msg_inf));
            challenger.observe_f128(msg_1);
            challenger.observe_f128(msg_inf);
            mlv_rhos.push(challenger.sample_f128());
            let n_in = a_folded.len();
            let (a_nxt, b_nxt) = if n_in >= 1024 {
                (
                    crate::scratch::take_f128(n_in / 2),
                    crate::scratch::take_f128(n_in / 2),
                )
            } else {
                (Vec::new(), Vec::new())
            };
            (a_folded, b_folded, a_nxt, b_nxt, 0usize, t_tail)
        }
    };

    let mut i = start_i;
    while i < (n_mlv - 1) {
        let log_n_before = a_mlv.len().trailing_zeros() as usize;
        if mlv_rhos.len() >= i + 2 && log_n_before >= 11 && (i + 2) < n_mlv {
            let rho_a = mlv_rhos[i];
            let rho_b = mlv_rhos[i + 1];
            let quarter = a_mlv.len() / 4;
            let mut r_next = vec![F128::ONE; log_n_before - 2];
            r_next[1..].copy_from_slice(&r[k_skip + i + 3..]);
            if (i + 3) < n_mlv && log_n_before >= 12 {
                let look = fold_2vars_and_compute_round_lookahead_into(
                    &a_mlv,
                    &b_mlv,
                    &mut a_nxt[..quarter],
                    &mut b_nxt[..quarter],
                    rho_a,
                    rho_b,
                    &r_next,
                );
                std::mem::swap(&mut a_mlv, &mut a_nxt);
                std::mem::swap(&mut b_mlv, &mut b_nxt);
                a_mlv.truncate(quarter);
                b_mlv.truncate(quarter);

                multilinear_msgs.push((look.msg2_1, look.msg2_inf));
                challenger.observe_f128(look.msg2_1);
                challenger.observe_f128(look.msg2_inf);
                let rho_c = challenger.sample_f128();
                mlv_rhos.push(rho_c);

                let (m_next1, m_nexti) = look.eval_round3(rho_c, F128::ONE);
                multilinear_msgs.push((m_next1, m_nexti));
                challenger.observe_f128(m_next1);
                challenger.observe_f128(m_nexti);
                mlv_rhos.push(challenger.sample_f128());
                i += 2;
                continue;
            } else {
                let (m1, mi) = fold_2vars_and_compute_round_pair_into(
                    &a_mlv,
                    &b_mlv,
                    &mut a_nxt[..quarter],
                    &mut b_nxt[..quarter],
                    rho_a,
                    rho_b,
                    &r_next,
                );
                std::mem::swap(&mut a_mlv, &mut a_nxt);
                std::mem::swap(&mut b_mlv, &mut b_nxt);
                a_mlv.truncate(quarter);
                b_mlv.truncate(quarter);

                multilinear_msgs.push((m1, mi));
                challenger.observe_f128(m1);
                challenger.observe_f128(mi);
                mlv_rhos.push(challenger.sample_f128());
                i += 2;
                continue;
            }
        }
        if mlv_rhos.len() > i + 1 {
            let rho_a = mlv_rhos[i];
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_a);
            i += 1;
            if i >= n_mlv - 1 {
                break;
            }
        }

        let rho_prev = mlv_rhos[i];
        let log_n_before = a_mlv.len().trailing_zeros() as usize;

        let mut r_next = vec![F128::ONE; log_n_before - 1];
        r_next[1..].copy_from_slice(&r[k_skip + i + 2..]);

        let (m1, mi) = if log_n_before >= 10 {
            let half = a_mlv.len() / 2;
            let (m1, mi) = fold_and_compute_round_pair_into(
                &a_mlv,
                &b_mlv,
                &mut a_nxt[..half],
                &mut b_nxt[..half],
                rho_prev,
                &r_next,
            );
            std::mem::swap(&mut a_mlv, &mut a_nxt);
            std::mem::swap(&mut b_mlv, &mut b_nxt);
            a_mlv.truncate(half);
            b_mlv.truncate(half);
            (m1, mi)
        } else {
            fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_prev);
            round_pair_naive(&a_mlv, &b_mlv, &r_next)
        };

        multilinear_msgs.push((m1, mi));
        challenger.observe_f128(m1);
        challenger.observe_f128(mi);
        mlv_rhos.push(challenger.sample_f128());
        i += 1;
    }

    let rho_last = *mlv_rhos.last().expect("at least one ρ sampled");
    fold_in_place_pair(&mut a_mlv, &mut b_mlv, rho_last);
    debug_assert_eq!(a_mlv.len(), 1);
    debug_assert_eq!(b_mlv.len(), 1);

    let final_a_eval = a_mlv[0];
    let final_b_eval = b_mlv[0];

    challenger.observe_f128(final_a_eval);
    challenger.observe_f128(final_b_eval);

    crate::scratch::give_f128(a_mlv);
    crate::scratch::give_f128(b_mlv);
    crate::scratch::give_f128(a_nxt);
    crate::scratch::give_f128(b_nxt);

    if zc_timing {
        eprintln!(
            "[zc-timing] rounds 3+ tail: {:.2} ms",
            t_tail.elapsed().as_secs_f64() * 1e3
        );
    }

    let r_rest: Vec<F128> = r[k_skip..].to_vec();

    let proof = ZerocheckProof {
        round1_ab,
        round1_c,
        multilinear_rounds: multilinear_msgs,
        final_a_eval,
        final_b_eval,
        final_c_eval,
    };
    let claim = ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: final_a_eval,
        b_eval: final_b_eval,
        c_eval: final_c_eval,
    };
    (proof, claim, s_hat_v_c)
}

pub fn verify<C: Challenger>(
    log_n: usize,
    proof: &ZerocheckProof,
    challenger: &mut C,
) -> Result<ZerocheckClaim, VerifyError> {
    let m = log_n;
    let k_skip = K_SKIP;
    const N_INNER: usize = 7;

    if m < k_skip + N_INNER {
        return Err(VerifyError::LogNTooSmall { log_n: m, k_skip });
    }
    let n_mlv = m - k_skip;
    let ell = 1usize << k_skip;

    if proof.round1_ab.len() != ell {
        return Err(VerifyError::BadRound1Length {
            expected: ell,
            got: proof.round1_ab.len(),
        });
    }
    if proof.round1_c.len() != ell {
        return Err(VerifyError::BadRound1Length {
            expected: ell,
            got: proof.round1_c.len(),
        });
    }
    if proof.multilinear_rounds.len() != n_mlv {
        return Err(VerifyError::BadMultilinearRoundsLength {
            expected: n_mlv,
            got: proof.multilinear_rounds.len(),
        });
    }

    challenger.observe_label(b"flock-zerocheck-v0");

    let r_skip = challenger.sample_f128_vec(k_skip);
    let r_outer = challenger.sample_f128_vec(m - k_skip - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..k_skip].copy_from_slice(&r_skip);
    for (i, val) in small_challenges_ghash().iter().enumerate() {
        r[k_skip + i] = *val;
    }
    for (i, val) in medium_challenges_ghash().iter().enumerate() {
        r[k_skip + 3 + i] = *val;
    }
    r[k_skip + N_INNER..].copy_from_slice(&r_outer);

    challenger.observe_f128_slice(&proof.round1_ab);
    challenger.observe_f128_slice(&proof.round1_c);
    let z = challenger.sample_f128();

    let computed_c_eval = interpolate_at_z_on_lambda(&proof.round1_c, k_skip, z);
    if computed_c_eval != proof.final_c_eval {
        return Err(VerifyError::CEvalMismatch);
    }

    let combined_at_lambda: Vec<F128> = proof
        .round1_ab
        .iter()
        .zip(&proof.round1_c)
        .map(|(x, y)| *x + *y)
        .collect();
    let combined_at_z = interpolate_at_z_combined(&combined_at_lambda, k_skip, z);
    let p_c_at_z = interpolate_at_z_on_lambda(&proof.round1_c, k_skip, z);
    let mut c_running = combined_at_z + p_c_at_z;

    let mut mlv_rhos: Vec<F128> = Vec::with_capacity(n_mlv);
    for (i, &(msg_1, msg_inf)) in proof.multilinear_rounds.iter().enumerate() {
        let r_eq = r[k_skip + i];
        let one_plus_r_eq = F128::ONE + r_eq;

        let g1 = msg_1;
        let g_inf = msg_inf;
        let g0 = (c_running + r_eq * g1) * one_plus_r_eq.inv();

        challenger.observe_f128(msg_1);
        challenger.observe_f128(msg_inf);
        let rho = challenger.sample_f128();
        mlv_rhos.push(rho);

        let one_plus_rho = F128::ONE + rho;
        c_running = g0 * one_plus_rho + g1 * rho + g_inf * rho * one_plus_rho;
    }

    let r_rest: Vec<F128> = r[k_skip..].to_vec();
    let expected_final = proof.final_a_eval * proof.final_b_eval;
    if c_running != expected_final {
        return Err(VerifyError::SumcheckFinalFailed);
    }

    challenger.observe_f128(proof.final_a_eval);
    challenger.observe_f128(proof.final_b_eval);

    Ok(ZerocheckClaim {
        z,
        mlv_challenges: mlv_rhos,
        r_rest,
        a_eval: proof.final_a_eval,
        b_eval: proof.final_b_eval,
        c_eval: proof.final_c_eval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
    }

    fn pack_abc(a: &[bool], b: &[bool], c: &[bool]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        use univariate_skip::pack_bits;
        (pack_bits(a), pack_bits(b), pack_bits(c))
    }

    #[test]
    fn prove_runs_end_to_end() {
        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Rng::new(m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut challenger = FsChallenger::new(b"flock-test-v0");
            let (proof, claim) = prove_packed(&a_p, &b_p, &c_p, m, &mut challenger);

            assert_eq!(proof.round1_ab.len(), 1usize << K_SKIP, "m={m}");
            assert_eq!(proof.round1_c.len(), 1usize << K_SKIP, "m={m}");
            assert_eq!(proof.multilinear_rounds.len(), m - K_SKIP, "m={m}");
            assert_eq!(claim.mlv_challenges.len(), m - K_SKIP, "m={m}");

            assert_eq!(claim.a_eval, proof.final_a_eval, "m={m}");
            assert_eq!(claim.b_eval, proof.final_b_eval, "m={m}");
            assert_eq!(claim.c_eval, proof.final_c_eval, "m={m}");
        }
    }

    #[test]
    fn prove_verify_roundtrip_honest() {
        for &m in &[13usize, 14, 15, 16] {
            let mut rng = Rng::new(1000 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let result = verify(m, &proof, &mut ch_verify);
            let claim_v = result.unwrap_or_else(|e| panic!("verify rejected at m={m}: {e:?}"));

            assert_eq!(claim_p, claim_v, "claim mismatch at m={m}");
        }
    }

    #[test]
    fn verify_rejects_mutations() {
        let m = 14;
        let mut rng = Rng::new(5050);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let _seed: u64 = 0xDEAD_BEEF;
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        let mutations: Vec<(&str, Box<dyn Fn(&ZerocheckProof) -> ZerocheckProof>)> = vec![
            (
                "round1_ab[0] bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.round1_ab[0].lo ^= 1;
                    q
                }),
            ),
            (
                "round1_c[5] bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.round1_c[5].lo ^= 1;
                    q
                }),
            ),
            (
                "multilinear_rounds[0].0 bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.multilinear_rounds[0].0.lo ^= 1;
                    q
                }),
            ),
            (
                "multilinear_rounds[2].1 bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    let last = q.multilinear_rounds.len() / 2;
                    q.multilinear_rounds[last].1.hi ^= 1;
                    q
                }),
            ),
            (
                "final_a_eval bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.final_a_eval.lo ^= 1;
                    q
                }),
            ),
            (
                "final_c_eval bit-flip",
                Box::new(|p| {
                    let mut q = p.clone();
                    q.final_c_eval.hi ^= 1;
                    q
                }),
            ),
        ];

        for (label, mutate) in mutations {
            let bad = mutate(&proof);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let result = verify(m, &bad, &mut ch);
            assert!(
                result.is_err(),
                "verify accepted mutated proof ({label}) — should have rejected"
            );
        }
    }

    #[test]
    fn verify_rejects_shape_errors() {
        let m = 14;
        let mut rng = Rng::new(606);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        let mut bad = proof.clone();
        bad.round1_ab.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, &bad, &mut ch),
            Err(VerifyError::BadRound1Length { .. })
        ));

        let mut bad = proof.clone();
        bad.multilinear_rounds.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, &bad, &mut ch),
            Err(VerifyError::BadMultilinearRoundsLength { .. })
        ));

        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(K_SKIP + 6, &proof, &mut ch),
            Err(VerifyError::LogNTooSmall { .. })
        ));
    }

    #[test]
    fn audit_false_statement_rejected() {
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(7777 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            c[3] = !c[3];

            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &proof, &mut ch_verify);
            assert!(
                res.is_err(),
                "verify ACCEPTED a false statement at m={m}: {res:?}"
            );
        }
    }

    #[test]
    fn audit_round_msg_inf_tamper_rejected() {
        let m = 14;
        let mut rng = Rng::new(424242);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        for idx in 0..proof.multilinear_rounds.len() {
            let mut bad = proof.clone();
            bad.multilinear_rounds[idx].1 += F128::ONE;
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &bad, &mut ch);
            assert!(res.is_err(), "msg_inf tamper at round {idx} ACCEPTED");
        }
    }

    #[test]
    fn audit_last_round_inf_constrained() {
        let m = 13;
        let mut rng = Rng::new(98765);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        let last = proof.multilinear_rounds.len() - 1;
        let mut bad = proof.clone();
        bad.multilinear_rounds[last].1 += F128::ONE;
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &bad, &mut ch).is_err(),
            "last-round msg_inf unconstrained"
        );
    }

    #[test]
    fn audit_final_ab_claims_bound_to_transcript() {
        let m = 14;
        let mut rng = Rng::new(0xF1A7_5A11);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);

        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);

        let mut ch_honest = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &proof, &mut ch_honest).is_ok(),
            "honest verify rejected"
        );
        let alpha_honest = ch_honest.sample_f128();

        let t = F128 {
            lo: 0x0123_4567_89ab_cdef,
            hi: 0xfedc_ba98_7654_3210,
        };
        assert!(t != F128::ZERO && t != F128::ONE, "t must be nontrivial");
        let mut bad = proof.clone();
        bad.final_a_eval *= t;
        bad.final_b_eval *= t.inv();
        assert_ne!(bad.final_a_eval, proof.final_a_eval, "tamper must change â");
        assert_ne!(bad.final_b_eval, proof.final_b_eval, "tamper must change b̂");
        assert_eq!(
            bad.final_a_eval * bad.final_b_eval,
            proof.final_a_eval * proof.final_b_eval,
            "tamper must preserve the product",
        );

        let mut ch_tampered = FsChallenger::new(b"flock-test-v0");
        assert!(
            verify(m, &bad, &mut ch_tampered).is_ok(),
            "product-preserving tamper rejected by zerocheck's own checks (unexpected)",
        );
        let alpha_tampered = ch_tampered.sample_f128();

        assert_ne!(
            alpha_honest, alpha_tampered,
            "final â/b̂ claims are NOT bound into the transcript: a product-preserving \
             tamper leaves the downstream challenge unchanged, breaking lincheck's \
             α-batched reduction of (v_a, v_b)",
        );
    }

    #[test]
    fn audit_many_false_statements_rejected() {
        let m = 13;
        for seed in 0..20u64 {
            let mut rng = Rng::new(0xBADC0DE ^ seed);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let mut c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
            let nflip = 1 + (rng.next_u64() as usize % 4);
            for _ in 0..nflip {
                let idx = rng.next_u64() as usize % c.len();
                c[idx] = !c[idx];
            }
            let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
            let mut ch_prove = FsChallenger::new(b"flock-test-v0");
            let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);
            let mut ch_verify = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, &proof, &mut ch_verify);
            assert!(
                res.is_err(),
                "false statement (seed={seed}) ACCEPTED: {res:?}"
            );
        }
    }

    #[test]
    fn audit_round_msg_1_tamper_rejected() {
        let m = 14;
        let mut rng = Rng::new(31415);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch_prove = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch_prove);
        for idx in 0..proof.multilinear_rounds.len() {
            let mut bad = proof.clone();
            bad.multilinear_rounds[idx].0 += F128::ONE;
            let mut ch = FsChallenger::new(b"flock-test-v0");
            assert!(
                verify(m, &bad, &mut ch).is_err(),
                "msg_1 tamper round {idx} ACCEPTED"
            );
        }
    }

    #[test]
    fn prove_deterministic() {
        let m = 14;
        let mut rng = Rng::new(99);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();

        let (a_p, b_p, c_p) = pack_abc(&a, &b, &c);
        let mut ch1 = FsChallenger::new(b"flock-test-v0");
        let mut ch2 = FsChallenger::new(b"flock-test-v0");
        let (proof1, claim1) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch1);
        let (proof2, claim2) = prove_packed(&a_p, &b_p, &c_p, m, &mut ch2);

        assert_eq!(proof1.round1_ab, proof2.round1_ab);
        assert_eq!(proof1.round1_c, proof2.round1_c);
        assert_eq!(proof1.multilinear_rounds, proof2.multilinear_rounds);
        assert_eq!(proof1.final_a_eval, proof2.final_a_eval);
        assert_eq!(proof1.final_b_eval, proof2.final_b_eval);
        assert_eq!(proof1.final_c_eval, proof2.final_c_eval);
        assert_eq!(claim1.z, claim2.z);
        assert_eq!(claim1.mlv_challenges, claim2.mlv_challenges);
    }
}
