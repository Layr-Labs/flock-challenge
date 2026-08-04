//! Subset-sum construction for the GPU NTT nibble tables
//! (`DEF_BUILD_NIBBLE_BLOCK` in `gpu_commit.rs`).
//!
//! Each 16-entry nibble table stores the scalar multiples `base · k` for a
//! 4-bit scalar `k` (`k` embedded as the polynomial `Σ k_j·x^j`), i.e.
//!
//! ```text
//! table[k] = ⊕ over set bits j of k of (base · x^j)
//! ```
//!
//! Because XOR distributes over the field product, the four seed terms
//! `base·x^0 … base·x^3` cost only **3 mulx** (the base itself is free), and
//! all 16 entries follow by XOR — 3 gf_mulx + 15 XORs per base, versus 64 mulx
//! for a naive per-entry chain. This is the ALU-side sibling of the byte-ledger
//! thesis: replace an instruction-heavy build with constant-time XOR fan-out
//! and let the table size (memory) stay identical.
//!
//! This module is the *portable* reference for the MSL kernel
//! (`gpu_commit.rs::gf_mulx`, `gf_mul_tab4`); the test suite cross-checks it
//! against the crate's full carry-less multiply (`F128::mul`) so the two
//! transcriptions cannot silently diverge (AGENTS.md §4).

/// `v · x mod P` in the GHASH bit-reflected layout `[u64; 2]` (lo = bits 0..64,
/// hi = bits 64..128). Mirrors `gpu_commit.rs::gf_mulx` on `[u32; 4]`.
#[inline(always)]
pub fn mulx(v: [u64; 2]) -> [u64; 2] {
    let carry = v[1] >> 63;
    [v[0] << 1 ^ (carry * 0x87), (v[1] << 1) | (v[0] >> 63)]
}

/// Build the 16-entry nibble table for `base` with 3 mulx + 15 XORs.
///
/// `table[k] == base · k` where `k` is the 4-bit scalar embedded as a field
/// element (bit `j` ↔ `x^j`). `table[0]` is the zero element.
pub fn build_nibble_block(base: [u64; 2]) -> [[u64; 2]; 16] {
    // seeds[j] = base · x^j, j = 0..4 — 3 mulx total (seed[0] = base is free).
    let mut seeds = [[0u64; 2]; 4];
    seeds[0] = base;
    for j in 1..4 {
        seeds[j] = mulx(seeds[j - 1]);
    }
    let mut table = [[0u64; 2]; 16];
    for k in 1..16u32 {
        let mut acc = [0u64; 2];
        for j in 0..4 {
            if (k >> j) & 1 != 0 {
                acc[0] ^= seeds[j as usize][0];
                acc[1] ^= seeds[j as usize][1];
            }
        }
        table[k as usize] = acc;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::F128;

    /// mulx agrees with the crate's canonical `field::mul_by_x` on known
    /// vectors: `x·x = x²` and the reduction constant fires exactly at the top
    /// bit (x^127 → x^127·x = x^128 ≡ x^7+x^2+x+1 → low byte 0x87).
    #[test]
    fn mulx_known_vectors() {
        let x: [u64; 2] = [2, 0];
        assert_eq!(mulx(x), [4, 0], "x·x = x²");
        assert_eq!(mulx([0x8000_0000_0000_0000, 0]), [0, 1], "x^127·x = x^128");
        assert_eq!(
            mulx([0, 0x8000_0000_0000_0000]),
            [0x87, 0],
            "x^127 (hi limb) folds to 0x87 in the low byte"
        );
    }

    /// The subset-sum table equals the true scalar multiples, computed with the
    /// crate's full carry-less multiply + reduce (independent oracle, and the
    /// same arithmetic the NEON/portable field kernels are tested against).
    #[test]
    fn subset_sum_matches_full_field_mul() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..8 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let base = [rng, rng.wrapping_mul(0x2545_f491_4f6c_dd1d)];
            let base_f = F128::new(base[0], base[1]);
            let table = build_nibble_block(base);
            for k in 0..16u64 {
                // embed scalar k as the polynomial Σ k_j·x^j
                let want = base_f * F128::new(k, 0);
                let got = F128::new(table[k as usize][0], table[k as usize][1]);
                assert_eq!(got, want, "table[{k}] = base·{k}");
            }
        }
    }

    /// Table entries are linear: table[a XOR b] == table[a] XOR table[b]
    /// (the property `gf_mul_tab4` relies on when XORing nibble contributions).
    #[test]
    fn nibble_table_is_linear() {
        let base = [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210];
        let table = build_nibble_block(base);
        for a in 0..16u32 {
            for b in 0..16u32 {
                let got = table[(a ^ b) as usize];
                let want = [
                    table[a as usize][0] ^ table[b as usize][0],
                    table[a as usize][1] ^ table[b as usize][1],
                ];
                assert_eq!(got, want, "table[{a} ^ {b}] == table[{a}] ^ table[{b}]");
            }
        }
    }
}
