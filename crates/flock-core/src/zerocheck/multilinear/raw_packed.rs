//! First-cascade reconstruction from packed A/B instead of compact state.
//!
//! R2 already computes its messages from packed rows. Keeping those inputs
//! for this consumer deletes the ranked 1 GiB anchor and 512 MiB delta writes,
//! and reads 1 GiB of packed rows instead of that 1.5 GiB intermediate.
//! Four scaled byte tables replace the stored anchors; all challenges and
//! round-four/five messages remain in the incumbent transcript order.

#[cfg(not(target_arch = "aarch64"))]
use super::F256Unreduced;
use super::{
    F128, Round3Lookahead, SplitEqGhash, UniSkipFoldTable, cascade_k_pass_n_hi,
    lookahead_inverse_reuse_enabled, lookahead_kappa_and_retained_inv, round2_pair_skip,
};
use crate::zerocheck::PaddingSpec;

/// Consume the original bit rows after rho1/rho2 are sampled and emit the
/// same post-two-fold tables and next messages as the compact first cascade.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_packed_and_round45_into(
    a_packed: &[u8],
    b_packed: &[u8],
    table: &UniSkipFoldTable,
    rho1: F128,
    rho2: F128,
    padding: &PaddingSpec,
    r_next4: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
) -> (F128, F128, Round3Lookahead) {
    assert_eq!(table.n_chunks, 8);
    assert_eq!(a_packed.len(), b_packed.len());
    assert_eq!(a_packed.len() % 32, 0);
    let n_groups = a_packed.len() / 32;
    assert!(n_groups >= 4 && n_groups.is_power_of_two());
    assert_eq!(a_out.len(), n_groups);
    assert_eq!(b_out.len(), n_groups);
    assert_eq!(r_next4.len(), n_groups.trailing_zeros() as usize);
    let r_par = r_next4[1];
    assert_ne!(r_par, F128::ZERO, "cascade requires non-zero r_next4[1]");

    // Indices are the original four adjacent rows: rho1 binds their low bit,
    // rho2 the next. Linearity of T_z makes each product a pre-scaled table.
    let lambda1 = rho1 * (F128::ONE + rho2);
    let lambda3 = rho1 * rho2;
    let lambda0 = F128::ONE + rho2 + lambda1;
    let lambda2 = rho2 + lambda3;
    let tables = [
        table.scaled_linear(lambda0),
        table.scaled_linear(lambda1),
        table.scaled_linear(lambda2),
        table.scaled_linear(lambda3),
    ];
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, 6);
    let n_vars = r_next4.len() - 1;
    let eq = SplitEqGhash::with_n_hi(&r_next4[1..], cascade_k_pass_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_groups);
    assert!(lo_size >= 2);
    let (kappa, retained_r_inv) =
        lookahead_kappa_and_retained_inv(r_par, lookahead_inverse_reuse_enabled());
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let out_chunk = 2 * lo_size;
    #[cfg(target_arch = "aarch64")]
    let degen = super::r2_degen_enabled();
    #[cfg(target_arch = "aarch64")]
    let folded_ones = table.fold_one_row(&[0xff; 8]);

    let mut partials = vec![(F128::ZERO, F128::ZERO); hi_size];
    let mut la_partials = vec![[F128::ZERO; 6]; hi_size];
    let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
    let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    let la_base = crate::epool::SyncPtr(la_partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(hi_size, |x_hi| {
        // SAFETY: this queue job owns one disjoint output chunk and its
        // partial slots. Inputs and the four tables are read-only.
        let (a_ptr, b_ptr) = unsafe {
            (
                a_base.ptr().add(x_hi * out_chunk),
                b_base.ptr().add(x_hi * out_chunk),
            )
        };
        let group_base = x_hi * out_chunk;
        #[cfg(target_arch = "aarch64")]
        let outv = {
            let mut outv = [F128::ZERO; 8];
            unsafe {
                super::kernels::aarch64::fold2_packed_and_round45_chunk_neon_8(
                    std::array::from_fn(|j| tables[j].as_ptr().cast::<u8>()),
                    folded_ones,
                    a_packed.as_ptr().add(group_base * 32),
                    b_packed.as_ptr().add(group_base * 32),
                    2 * group_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    a_ptr,
                    b_ptr,
                    eq_lo.as_ptr(),
                    lo_size,
                    degen,
                    outv.as_mut_ptr(),
                );
            }
            outv
        };
        #[cfg(not(target_arch = "aarch64"))]
        let outv = {
            let a_chunk = unsafe { std::slice::from_raw_parts_mut(a_ptr, out_chunk) };
            let b_chunk = unsafe { std::slice::from_raw_parts_mut(b_ptr, out_chunk) };
            fold2_packed_round45_chunk_scalar(
                a_packed,
                b_packed,
                &tables,
                pair_in_block_mask,
                useful_pairs_inclusive,
                group_base,
                a_chunk,
                b_chunk,
                eq_lo,
                lo_size,
            )
        };

        let eq_h = eq_hi[x_hi];
        unsafe {
            *partials_base.ptr().add(x_hi) = (
                eq_h * (kappa * outv[0] + outv[2]),
                eq_h * (kappa * outv[1] + outv[3]),
            );
            *la_base.ptr().add(x_hi) = [
                eq_h * outv[2],
                eq_h * outv[3],
                eq_h * outv[4],
                eq_h * outv[5],
                eq_h * outv[6],
                eq_h * outv[7],
            ];
        }
    });

    let (sum1, sum_inf) = partials
        .iter()
        .fold((F128::ZERO, F128::ZERO), |(sum1, sum_inf), &(p1, pinf)| {
            (sum1 + p1, sum_inf + pinf)
        });
    let mut agg = [F128::ZERO; 6];
    for slot in &la_partials {
        for (a, v) in agg.iter_mut().zip(slot) {
            *a += *v;
        }
    }
    let r_inv = retained_r_inv.unwrap_or_else(|| r_par.inv());
    let [w1, w2, w0, w3, w4, w5] = agg.map(|v| r_inv * v);
    let lookahead = Round3Lookahead {
        c: [w0, w0 + w1 + w2, w2, w3, w3 + w4 + w5, w5],
    };
    (r_next4[0] * sum1, sum_inf, lookahead)
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn fold2_packed_round45_chunk_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    tables: &[Vec<F128>; 4],
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    group_base: usize,
    a_out: &mut [F128],
    b_out: &mut [F128],
    eq_lo: &[F128],
    lo_size: usize,
) -> [F128; 8] {
    for local in 0..a_out.len() {
        let group = group_base + local;
        let mut a = F128::ZERO;
        let mut b = F128::ZERO;
        for row in 0..4 {
            let pair = 2 * group + row / 2;
            if (pair & pair_in_block_mask) >= useful_pairs_inclusive {
                continue;
            }
            let offset = group * 32 + row * 8;
            for byte in 0..8 {
                a += tables[row][byte * 256 + a_packed[offset + byte] as usize];
                b += tables[row][byte * 256 + b_packed[offset + byte] as usize];
            }
        }
        a_out[local] = a;
        b_out[local] = b;
    }

    let mut products = [F256Unreduced::ZERO; 8];
    for t in 0..lo_size / 2 {
        let offset = 4 * t;
        let a = &a_out[offset..offset + 4];
        let b = &b_out[offset..offset + 4];
        let w = eq_lo[2 * t + 1];
        let [a0w, a1w, a2w, a3w] = [a[0], a[1], a[2], a[3]].map(|v| w * v);
        products[0] ^= a1w.mul_unreduced(b[1]);
        products[1] ^= (a0w + a1w).mul_unreduced(b[0] + b[1]);
        products[2] ^= a3w.mul_unreduced(b[3]);
        products[3] ^= (a2w + a3w).mul_unreduced(b[2] + b[3]);
        products[4] ^= a2w.mul_unreduced(b[2]);
        let (e_aw, e_b) = (a0w + a2w, b[0] + b[2]);
        let (o_aw, o_b) = (a1w + a3w, b[1] + b[3]);
        products[5] ^= e_aw.mul_unreduced(e_b);
        products[6] ^= o_aw.mul_unreduced(o_b);
        products[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
    }
    products.map(F256Unreduced::reduce)
}

#[cfg(test)]
mod tests {
    use super::super::{
        fold2_compact_and_round45_into,
        uni_skip_fold_and_round_pair_compact_padded_lookahead,
        uni_skip_fold_and_round_pair_packed_padded_lookahead,
    };
    use super::*;

    #[test]
    fn packed_messages_and_first_cascade_match_compact_with_poison_padding() {
        const M: usize = 16;
        const K_SKIP: usize = 6;
        let a: Vec<u8> = (0..1usize << (M - 3))
            .map(|i| ((i.wrapping_mul(73) ^ (i >> 3) ^ (i >> 9)) & 255) as u8)
            .collect();
        let random_b: Vec<u8> = (0..a.len())
            .map(|i| ((i.wrapping_mul(151) ^ (i >> 2) ^ 0x5a) & 255) as u8)
            .collect();
        let ones_b = vec![0xff; a.len()];
        let zero_b = vec![0; a.len()];
        let table = UniSkipFoldTable::new(K_SKIP, F128::new(0x1234, 0x9876));
        let mut mlv = vec![F128::ONE; M - K_SKIP];
        for (i, r) in mlv.iter_mut().enumerate().skip(1) {
            *r = F128::new((i * 19 + 7) as u64, (i * 31 + 11) as u64);
        }
        let mut r_next4 = vec![F128::ONE; M - K_SKIP - 2];
        r_next4[1..].copy_from_slice(&mlv[3..]);
        let n_groups = a.len() / 32;

        // Ranked padding gives pair120 live and pair121 dead, including
        // mixed four-row groups. Input dead-pair bytes stay deliberately
        // nonzero: both implementations must enforce the same pair mask.
        for padding in [
            PaddingSpec {
                k_log: 14,
                useful_bits_per_block: 15_409,
            },
            PaddingSpec::dense(M),
            PaddingSpec {
                k_log: 14,
                useful_bits_per_block: 0,
            },
        ] {
            for b in [&random_b, &ones_b, &zero_b] {
                let (compact, m1, mi, lookahead) =
                    uni_skip_fold_and_round_pair_compact_padded_lookahead(
                        &a,
                        b,
                        M,
                        K_SKIP,
                        &table,
                        &mlv,
                        &padding,
                        None,
                    );
                let raw_messages = uni_skip_fold_and_round_pair_packed_padded_lookahead(
                    &a, b, M, K_SKIP, &table, &mlv, &padding,
                );
                assert_eq!(raw_messages, (m1, mi, lookahead));

                for (rho1, rho2) in [
                    (F128::ZERO, F128::ZERO),
                    (F128::ONE, F128::ZERO),
                    (F128::ZERO, F128::ONE),
                    (F128::ONE, F128::ONE),
                    (F128::new(0x4d, 0x81), F128::new(0x29, 0x73)),
                ] {
                    let mut expected_a = vec![F128::ZERO; n_groups];
                    let mut expected_b = vec![F128::ZERO; n_groups];
                    let expected = fold2_compact_and_round45_into(
                        &compact,
                        &table,
                        rho1,
                        rho2,
                        &r_next4,
                        &mut expected_a,
                        &mut expected_b,
                    );
                    let mut actual_a = vec![F128::ZERO; n_groups];
                    let mut actual_b = vec![F128::ZERO; n_groups];
                    let actual = fold2_packed_and_round45_into(
                        &a,
                        b,
                        &table,
                        rho1,
                        rho2,
                        &padding,
                        &r_next4,
                        &mut actual_a,
                        &mut actual_b,
                    );
                    assert_eq!(actual_a, expected_a);
                    assert_eq!(actual_b, expected_b);
                    assert_eq!(actual, expected);
                }
                compact.recycle();
            }
        }
    }
}
