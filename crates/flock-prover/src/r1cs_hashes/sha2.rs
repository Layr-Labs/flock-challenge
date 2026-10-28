use super::common::{BitRecord, add_carry_parts, or_bit_at, or_u32_at_bit};
use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};

pub const K_LOG: usize = 15;
pub const K: usize = 1 << K_LOG;
pub const K_SKIP: usize = 6;

pub const N_ROUNDS: usize = 64;
pub const N_SCHED: usize = 48;
pub const WORD_BITS: usize = 32;
pub const H_WORDS: usize = 8;
pub const M_WORDS: usize = 16;
pub const N_OUT_WORDS: usize = 8;
pub const ADDS_PER_ROUND: usize = 7;
pub const ADDS_PER_SCHED: usize = 3;
pub const CARRIES_PER_ADD: usize = WORD_BITS - 1;

pub const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
pub const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub const SLOT_BITS: usize = 256;
pub const H_BASE: usize = 0;
pub const H_OUT_BASE: usize = SLOT_BITS;
pub const M_BASE: usize = 2 * SLOT_BITS;
pub const CH_AND_BASE: usize = M_BASE + M_WORDS * WORD_BITS;
pub const MAJ_AND_BASE: usize = CH_AND_BASE + N_ROUNDS * WORD_BITS;
pub const ROUND_CARRY_BASE: usize = MAJ_AND_BASE + N_ROUNDS * WORD_BITS;
pub const W_BASE: usize = ROUND_CARRY_BASE + N_ROUNDS * ADDS_PER_ROUND * CARRIES_PER_ADD;
pub const SCHED_CARRY_BASE: usize = W_BASE + N_SCHED * WORD_BITS;
pub const T1_BASE: usize = SCHED_CARRY_BASE + N_SCHED * ADDS_PER_SCHED * CARRIES_PER_ADD;
pub const E_NEW_BASE: usize = T1_BASE + N_ROUNDS * WORD_BITS;
pub const A_NEW_BASE: usize = E_NEW_BASE + N_ROUNDS * WORD_BITS;
pub const OUT_CARRY_BASE: usize = A_NEW_BASE + N_ROUNDS * WORD_BITS;
pub const Z_CONST_POS: usize = OUT_CARRY_BASE + N_OUT_WORDS * CARRIES_PER_ADD;
pub const USEFUL_BITS: usize = Z_CONST_POS + 1;

#[inline]
pub fn h_bit(w: usize, b: usize) -> usize {
    H_BASE + WORD_BITS * w + b
}
#[inline]
pub fn m_bit(i: usize, b: usize) -> usize {
    M_BASE + WORD_BITS * i + b
}
#[inline]
pub fn ch_and_bit(r: usize, b: usize) -> usize {
    CH_AND_BASE + WORD_BITS * r + b
}
#[inline]
pub fn maj_and_bit(r: usize, b: usize) -> usize {
    MAJ_AND_BASE + WORD_BITS * r + b
}
#[inline]
pub fn round_carry_bit(r: usize, add: usize, b: usize) -> usize {
    ROUND_CARRY_BASE + r * ADDS_PER_ROUND * CARRIES_PER_ADD + add * CARRIES_PER_ADD + b
}
#[inline]
pub fn w_bit(t: usize, b: usize) -> usize {
    debug_assert!(t < N_SCHED + 16);
    if t < 16 {
        m_bit(t, b)
    } else {
        W_BASE + (t - 16) * WORD_BITS + b
    }
}
#[inline]
pub fn sched_carry_bit(t: usize, add: usize, b: usize) -> usize {
    debug_assert!((16..16 + N_SCHED).contains(&t));
    SCHED_CARRY_BASE + (t - 16) * ADDS_PER_SCHED * CARRIES_PER_ADD + add * CARRIES_PER_ADD + b
}
#[inline]
pub fn t1_bit(r: usize, b: usize) -> usize {
    T1_BASE + WORD_BITS * r + b
}
#[inline]
pub fn e_new_bit(r: usize, b: usize) -> usize {
    E_NEW_BASE + WORD_BITS * r + b
}
#[inline]
pub fn a_new_bit(r: usize, b: usize) -> usize {
    A_NEW_BASE + WORD_BITS * r + b
}
#[inline]
pub fn out_carry_bit(w: usize, b: usize) -> usize {
    OUT_CARRY_BASE + w * CARRIES_PER_ADD + b
}
#[inline]
pub fn h_out_bit(w: usize, b: usize) -> usize {
    H_OUT_BASE + w * WORD_BITS + b
}

type Sup = Vec<usize>;
type Word = Vec<Sup>;

fn zero_word() -> Word {
    (0..WORD_BITS).map(|_| Sup::new()).collect()
}

fn wire_word<F: Fn(usize) -> usize>(slot: F) -> Word {
    (0..WORD_BITS).map(|b| vec![slot(b)]).collect()
}

fn xor_sup(a: &Sup, b: &Sup) -> Sup {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn xor3(a: &Sup, b: &Sup, c: &Sup) -> Sup {
    xor_sup(&xor_sup(a, b), c)
}

fn xor_words(x: &Word, y: &Word) -> Word {
    (0..WORD_BITS).map(|i| xor_sup(&x[i], &y[i])).collect()
}

fn rotr(w: &Word, n: usize) -> Word {
    (0..WORD_BITS)
        .map(|i| w[(i + n) % WORD_BITS].clone())
        .collect()
}

fn shr(w: &Word, n: usize) -> Word {
    (0..WORD_BITS)
        .map(|i| {
            if i + n < WORD_BITS {
                w[i + n].clone()
            } else {
                Sup::new()
            }
        })
        .collect()
}

fn rot_xor3(w: &Word, r1: usize, r2: usize, r3: usize) -> Word {
    let a = rotr(w, r1);
    let b = rotr(w, r2);
    let c = rotr(w, r3);
    (0..WORD_BITS).map(|i| xor3(&a[i], &b[i], &c[i])).collect()
}

fn sigma_xor(w: &Word, r1: usize, r2: usize, sh: usize) -> Word {
    let a = rotr(w, r1);
    let b = rotr(w, r2);
    let s = shr(w, sh);
    (0..WORD_BITS).map(|i| xor3(&a[i], &b[i], &s[i])).collect()
}

#[inline]
fn sigma_0(w: &Word) -> Word {
    sigma_xor(w, 7, 18, 3)
}
#[inline]
fn sigma_1(w: &Word) -> Word {
    sigma_xor(w, 17, 19, 10)
}
#[inline]
fn big_sigma_0(w: &Word) -> Word {
    rot_xor3(w, 2, 13, 22)
}
#[inline]
fn big_sigma_1(w: &Word) -> Word {
    rot_xor3(w, 6, 11, 25)
}

fn add32_inline<F: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F,
    a_rows: &mut [Sup],
    b_rows: &mut [Sup],
) -> Word {
    let mut sum = zero_word();
    let mut cin: Sup = Sup::new();
    for i in 0..WORD_BITS {
        sum[i] = xor3(&x[i], &y[i], &cin);
        if i < CARRIES_PER_ADD {
            let slot = carry_slot(i);
            a_rows[slot] = xor_sup(&x[i], &cin);
            b_rows[slot] = xor_sup(&y[i], &cin);
            cin = xor_sup(&cin, &vec![slot]);
        }
    }
    sum
}

fn materialize<F: Fn(usize) -> usize>(
    raw: &Word,
    slot_fn: F,
    a_rows: &mut [Sup],
    b_rows: &mut [Sup],
) -> Word {
    let mut out = zero_word();
    for b in 0..WORD_BITS {
        let s = slot_fn(b);
        a_rows[s] = raw[b].clone();
        b_rows[s] = vec![Z_CONST_POS];
        out[b] = vec![s];
    }
    out
}

fn add32_alloc<F1: Fn(usize) -> usize, F2: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F1,
    sum_slot: F2,
    a_rows: &mut [Sup],
    b_rows: &mut [Sup],
) -> Word {
    let raw = add32_inline(x, y, carry_slot, a_rows, b_rows);
    materialize(&raw, sum_slot, a_rows, b_rows)
}

pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut a_rows: Vec<Sup> = vec![Sup::new(); K];
    let mut b_rows: Vec<Sup> = vec![Sup::new(); K];

    a_rows[Z_CONST_POS] = vec![Z_CONST_POS];
    b_rows[Z_CONST_POS] = vec![Z_CONST_POS];

    for w in 0..H_WORDS {
        for b in 0..WORD_BITS {
            let s = h_bit(w, b);
            a_rows[s] = vec![s];
            b_rows[s] = vec![Z_CONST_POS];
        }
    }
    for i in 0..M_WORDS {
        for b in 0..WORD_BITS {
            let s = m_bit(i, b);
            a_rows[s] = vec![s];
            b_rows[s] = vec![Z_CONST_POS];
        }
    }

    let h_in: Vec<Word> = (0..H_WORDS).map(|w| wire_word(|b| h_bit(w, b))).collect();
    let mut w_arr: Vec<Word> = (0..M_WORDS).map(|i| wire_word(|b| m_bit(i, b))).collect();

    for t in 16..(16 + N_SCHED) {
        let s1 = sigma_1(&w_arr[t - 2]);
        let s0 = sigma_0(&w_arr[t - 15]);
        let w_m7 = w_arr[t - 7].clone();
        let w_m16 = w_arr[t - 16].clone();
        let sched_0 = add32_inline(
            &s1,
            &w_m7,
            |i| sched_carry_bit(t, 0, i),
            &mut a_rows,
            &mut b_rows,
        );
        let sched_1 = add32_inline(
            &sched_0,
            &s0,
            |i| sched_carry_bit(t, 1, i),
            &mut a_rows,
            &mut b_rows,
        );
        let w_t = add32_alloc(
            &sched_1,
            &w_m16,
            |i| sched_carry_bit(t, 2, i),
            |b| w_bit(t, b),
            &mut a_rows,
            &mut b_rows,
        );
        w_arr.push(w_t);
    }

    let mut state: [Word; 8] = [
        h_in[0].clone(),
        h_in[1].clone(),
        h_in[2].clone(),
        h_in[3].clone(),
        h_in[4].clone(),
        h_in[5].clone(),
        h_in[6].clone(),
        h_in[7].clone(),
    ];

    for r in 0..N_ROUNDS {
        let a = state[0].clone();
        let bb = state[1].clone();
        let c = state[2].clone();
        let d = state[3].clone();
        let e = state[4].clone();
        let f = state[5].clone();
        let g = state[6].clone();
        let h_var = state[7].clone();

        let mut ch_and = zero_word();
        for bit in 0..WORD_BITS {
            let s = ch_and_bit(r, bit);
            a_rows[s] = e[bit].clone();
            b_rows[s] = xor_sup(&f[bit], &g[bit]);
            ch_and[bit] = vec![s];
        }
        let mut maj_and = zero_word();
        for bit in 0..WORD_BITS {
            let s = maj_and_bit(r, bit);
            a_rows[s] = xor_sup(&a[bit], &bb[bit]);
            b_rows[s] = xor_sup(&a[bit], &c[bit]);
            maj_and[bit] = vec![s];
        }
        let ch_out = xor_words(&ch_and, &g);
        let maj_out = xor_words(&maj_and, &a);

        let t1_0 = add32_inline(
            &h_var,
            &big_sigma_1(&e),
            |i| round_carry_bit(r, 0, i),
            &mut a_rows,
            &mut b_rows,
        );
        let t1_1 = add32_inline(
            &t1_0,
            &ch_out,
            |i| round_carry_bit(r, 1, i),
            &mut a_rows,
            &mut b_rows,
        );
        let k_word: Word = (0..WORD_BITS)
            .map(|i| {
                if (SHA256_K[r] >> i) & 1 == 1 {
                    vec![Z_CONST_POS]
                } else {
                    Sup::new()
                }
            })
            .collect();
        let t1_2 = add32_inline(
            &t1_1,
            &k_word,
            |i| round_carry_bit(r, 2, i),
            &mut a_rows,
            &mut b_rows,
        );
        let t1 = add32_alloc(
            &t1_2,
            &w_arr[r],
            |i| round_carry_bit(r, 3, i),
            |b| t1_bit(r, b),
            &mut a_rows,
            &mut b_rows,
        );
        let t2 = add32_inline(
            &big_sigma_0(&a),
            &maj_out,
            |i| round_carry_bit(r, 4, i),
            &mut a_rows,
            &mut b_rows,
        );
        let e_new = add32_alloc(
            &d,
            &t1,
            |i| round_carry_bit(r, 5, i),
            |b| e_new_bit(r, b),
            &mut a_rows,
            &mut b_rows,
        );
        let a_new = add32_alloc(
            &t1,
            &t2,
            |i| round_carry_bit(r, 6, i),
            |b| a_new_bit(r, b),
            &mut a_rows,
            &mut b_rows,
        );

        state = [a_new, a, bb, c, e_new, e, f, g];
    }

    for w in 0..N_OUT_WORDS {
        let _ = add32_alloc(
            &state[w],
            &h_in[w],
            |i| out_carry_bit(w, i),
            |b| h_out_bit(w, b),
            &mut a_rows,
            &mut b_rows,
        );
    }

    let to_mat = |rows| SparseBinaryMatrix {
        num_rows: K,
        num_cols: K,
        rows,
    };
    (to_mat(a_rows), to_mat(b_rows))
}

fn write_word(z: &mut [bool], base: usize, v: u32) {
    for b in 0..WORD_BITS {
        z[base + b] = (v >> b) & 1 == 1;
    }
}

fn add32_w(x: u32, y: u32, carry_base: usize, z: &mut [bool]) -> u32 {
    let mut cin: bool = false;
    for i in 0..CARRIES_PER_ADD {
        let xi = ((x >> i) & 1) == 1;
        let yi = ((y >> i) & 1) == 1;
        let aux = (xi ^ cin) && (yi ^ cin);
        z[carry_base + i] = aux;
        cin ^= aux;
    }
    x.wrapping_add(y)
}

pub fn build_block_witness(h_in: &[u32; 8], m: &[u32; 16]) -> Vec<bool> {
    let mut z = vec![false; K];
    z[Z_CONST_POS] = true;

    for w in 0..H_WORDS {
        write_word(&mut z, h_bit(w, 0), h_in[w]);
    }
    for i in 0..M_WORDS {
        write_word(&mut z, m_bit(i, 0), m[i]);
    }

    let mut w_arr = [0u32; 64];
    w_arr[..16].copy_from_slice(m);
    for t in 16..64 {
        let s0 =
            w_arr[t - 15].rotate_right(7) ^ w_arr[t - 15].rotate_right(18) ^ (w_arr[t - 15] >> 3);
        let s1 =
            w_arr[t - 2].rotate_right(17) ^ w_arr[t - 2].rotate_right(19) ^ (w_arr[t - 2] >> 10);
        let sched_0 = add32_w(s1, w_arr[t - 7], sched_carry_bit(t, 0, 0), &mut z);
        let sched_1 = add32_w(sched_0, s0, sched_carry_bit(t, 1, 0), &mut z);
        let w_t = add32_w(sched_1, w_arr[t - 16], sched_carry_bit(t, 2, 0), &mut z);
        write_word(&mut z, w_bit(t, 0), w_t);
        w_arr[t] = w_t;
    }

    let mut state = *h_in;
    for r in 0..N_ROUNDS {
        let (a, b, c, d, e, f, g, h_var) = (
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        );
        let ch_and = e & (f ^ g);
        write_word(&mut z, ch_and_bit(r, 0), ch_and);
        let maj_and = (a ^ b) & (a ^ c);
        write_word(&mut z, maj_and_bit(r, 0), maj_and);

        let ch_out = ch_and ^ g;
        let maj_out = maj_and ^ a;
        let s1e = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let s0a = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);

        let t1_0 = add32_w(h_var, s1e, round_carry_bit(r, 0, 0), &mut z);
        let t1_1 = add32_w(t1_0, ch_out, round_carry_bit(r, 1, 0), &mut z);
        let t1_2 = add32_w(t1_1, SHA256_K[r], round_carry_bit(r, 2, 0), &mut z);
        let t1 = add32_w(t1_2, w_arr[r], round_carry_bit(r, 3, 0), &mut z);
        write_word(&mut z, t1_bit(r, 0), t1);

        let t2 = add32_w(s0a, maj_out, round_carry_bit(r, 4, 0), &mut z);
        let e_new = add32_w(d, t1, round_carry_bit(r, 5, 0), &mut z);
        write_word(&mut z, e_new_bit(r, 0), e_new);
        let a_new = add32_w(t1, t2, round_carry_bit(r, 6, 0), &mut z);
        write_word(&mut z, a_new_bit(r, 0), a_new);

        state = [a_new, a, b, c, e_new, e, f, g];
    }

    for w in 0..N_OUT_WORDS {
        let h_out = add32_w(state[w], h_in[w], out_carry_bit(w, 0), &mut z);
        write_word(&mut z, h_out_bit(w, 0), h_out);
    }
    z
}

pub fn read_h_out(z: &[bool]) -> [u32; 8] {
    std::array::from_fn(|w| {
        (0..WORD_BITS).fold(0u32, |acc, b| acc | ((z[h_out_bit(w, b)] as u32) << b))
    })
}

pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
    let (a_0, b_0) = build_matrices();
    super::common::build_block_r1cs_with_matrices(
        n_blocks_log,
        K_LOG,
        K_SKIP,
        USEFUL_BITS,
        a_0,
        b_0,
        Some(Z_CONST_POS),
    )
}

fn scatter_add32_inline<F: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F,
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
) -> Word {
    let mut sum = zero_word();
    let mut cin: Sup = Sup::new();
    for i in 0..WORD_BITS {
        sum[i] = xor3(&x[i], &y[i], &cin);
        if i < CARRIES_PER_ADD {
            let slot = carry_slot(i);
            let row = slot;
            let e = eq_inner[row];
            let ea = alpha * e;
            for &c in xor_sup(&x[i], &cin).iter() {
                comb[c] += ea;
            }
            for &c in xor_sup(&y[i], &cin).iter() {
                comb[c] += e;
            }
            cin = xor_sup(&cin, &vec![slot]);
        }
    }
    sum
}

fn scatter_materialize<F: Fn(usize) -> usize>(
    raw: &Word,
    slot_fn: F,
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
) -> Word {
    let mut out = zero_word();
    for b in 0..WORD_BITS {
        let s = slot_fn(b);
        let e = eq_inner[s];
        let ea = alpha * e;
        for &c in raw[b].iter() {
            comb[c] += ea;
        }
        comb[Z_CONST_POS] += e;
        out[b] = vec![s];
    }
    out
}

fn scatter_add32_alloc<F1: Fn(usize) -> usize, F2: Fn(usize) -> usize>(
    x: &Word,
    y: &Word,
    carry_slot: F1,
    sum_slot: F2,
    comb: &mut [F128],
    alpha: F128,
    eq_inner: &[F128],
) -> Word {
    let raw = scatter_add32_inline(x, y, carry_slot, comb, alpha, eq_inner);
    scatter_materialize(&raw, sum_slot, comb, alpha, eq_inner)
}

pub struct Sha2LincheckCircuit;

impl flock_core::lincheck::LincheckCircuit for Sha2LincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");
        let mut comb = vec![F128::ZERO; K];

        let e0 = eq_inner[Z_CONST_POS];
        comb[Z_CONST_POS] = alpha.fma(e0, comb[Z_CONST_POS] + e0);

        for w in 0..H_WORDS {
            for b in 0..WORD_BITS {
                let s = h_bit(w, b);
                let e = eq_inner[s];
                comb[s] = alpha.fma(e, comb[s]);
                comb[Z_CONST_POS] += e;
            }
        }
        for i in 0..M_WORDS {
            for b in 0..WORD_BITS {
                let s = m_bit(i, b);
                let e = eq_inner[s];
                comb[s] = alpha.fma(e, comb[s]);
                comb[Z_CONST_POS] += e;
            }
        }

        let h_in: Vec<Word> = (0..H_WORDS).map(|w| wire_word(|b| h_bit(w, b))).collect();
        let mut w_arr: Vec<Word> = (0..M_WORDS).map(|i| wire_word(|b| m_bit(i, b))).collect();

        for t in 16..(16 + N_SCHED) {
            let s1 = sigma_1(&w_arr[t - 2]);
            let s0 = sigma_0(&w_arr[t - 15]);
            let w_m7 = w_arr[t - 7].clone();
            let w_m16 = w_arr[t - 16].clone();
            let sched_0 = scatter_add32_inline(
                &s1,
                &w_m7,
                |i| sched_carry_bit(t, 0, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let sched_1 = scatter_add32_inline(
                &sched_0,
                &s0,
                |i| sched_carry_bit(t, 1, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let w_t = scatter_add32_alloc(
                &sched_1,
                &w_m16,
                |i| sched_carry_bit(t, 2, i),
                |b| w_bit(t, b),
                &mut comb,
                alpha,
                eq_inner,
            );
            w_arr.push(w_t);
        }

        let mut state: [Word; 8] = [
            h_in[0].clone(),
            h_in[1].clone(),
            h_in[2].clone(),
            h_in[3].clone(),
            h_in[4].clone(),
            h_in[5].clone(),
            h_in[6].clone(),
            h_in[7].clone(),
        ];

        for r in 0..N_ROUNDS {
            let a = state[0].clone();
            let bb = state[1].clone();
            let c = state[2].clone();
            let d = state[3].clone();
            let e = state[4].clone();
            let f = state[5].clone();
            let g = state[6].clone();
            let h_var = state[7].clone();

            let mut ch_and = zero_word();
            for bit in 0..WORD_BITS {
                let s = ch_and_bit(r, bit);
                let eq = eq_inner[s];
                let ea = alpha * eq;
                for &c2 in e[bit].iter() {
                    comb[c2] += ea;
                }
                for &c2 in xor_sup(&f[bit], &g[bit]).iter() {
                    comb[c2] += eq;
                }
                ch_and[bit] = vec![s];
            }
            let mut maj_and = zero_word();
            for bit in 0..WORD_BITS {
                let s = maj_and_bit(r, bit);
                let eq = eq_inner[s];
                let ea = alpha * eq;
                for &c2 in xor_sup(&a[bit], &bb[bit]).iter() {
                    comb[c2] += ea;
                }
                for &c2 in xor_sup(&a[bit], &c[bit]).iter() {
                    comb[c2] += eq;
                }
                maj_and[bit] = vec![s];
            }
            let ch_out = xor_words(&ch_and, &g);
            let maj_out = xor_words(&maj_and, &a);

            let t1_0 = scatter_add32_inline(
                &h_var,
                &big_sigma_1(&e),
                |i| round_carry_bit(r, 0, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let t1_1 = scatter_add32_inline(
                &t1_0,
                &ch_out,
                |i| round_carry_bit(r, 1, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let k_word: Word = (0..WORD_BITS)
                .map(|i| {
                    if (SHA256_K[r] >> i) & 1 == 1 {
                        vec![Z_CONST_POS]
                    } else {
                        Sup::new()
                    }
                })
                .collect();
            let t1_2 = scatter_add32_inline(
                &t1_1,
                &k_word,
                |i| round_carry_bit(r, 2, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let t1 = scatter_add32_alloc(
                &t1_2,
                &w_arr[r],
                |i| round_carry_bit(r, 3, i),
                |b| t1_bit(r, b),
                &mut comb,
                alpha,
                eq_inner,
            );
            let t2 = scatter_add32_inline(
                &big_sigma_0(&a),
                &maj_out,
                |i| round_carry_bit(r, 4, i),
                &mut comb,
                alpha,
                eq_inner,
            );
            let e_new = scatter_add32_alloc(
                &d,
                &t1,
                |i| round_carry_bit(r, 5, i),
                |b| e_new_bit(r, b),
                &mut comb,
                alpha,
                eq_inner,
            );
            let a_new = scatter_add32_alloc(
                &t1,
                &t2,
                |i| round_carry_bit(r, 6, i),
                |b| a_new_bit(r, b),
                &mut comb,
                alpha,
                eq_inner,
            );

            state = [a_new, a, bb, c, e_new, e, f, g];
        }

        for w in 0..N_OUT_WORDS {
            let _ = scatter_add32_alloc(
                &state[w],
                &h_in[w],
                |i| out_carry_bit(w, i),
                |b| h_out_bit(w, b),
                &mut comb,
                alpha,
                eq_inner,
            );
        }

        comb
    }
}

#[inline]
pub(crate) fn big_sigma0(x: u32) -> u32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return flock_core::gf2_kernels::gf2_kern_sha256_big_sigma0(x as u64) as u32;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
    }
}
#[inline]
pub(crate) fn big_sigma1(x: u32) -> u32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return flock_core::gf2_kernels::gf2_kern_sha256_big_sigma1(x as u64) as u32;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
    }
}
#[inline]
pub(crate) fn small_sigma0(x: u32) -> u32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return flock_core::gf2_kernels::gf2_kern_sha256_small_sigma0(x as u64) as u32;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
    }
}
#[inline]
pub(crate) fn small_sigma1(x: u32) -> u32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return flock_core::gf2_kernels::gf2_kern_sha256_small_sigma1(x as u64) as u32;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
    }
}
#[inline(always)]
fn sha256_ch_and_out(e: u32, f: u32, g: u32) -> (u32, u32) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let p = flock_core::gf2_kernels::gf2_kern_sha256_ch_and_out(e as u64, f as u64, g as u64);
        return (p as u32, (p >> 32) as u32);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let ch_and = e & (f ^ g);
        (ch_and, ch_and ^ g)
    }
}
#[inline(always)]
fn sha256_maj_and_out(a: u32, b: u32, c: u32) -> (u32, u32) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let p = flock_core::gf2_kernels::gf2_kern_sha256_maj_and_out(a as u64, b as u64, c as u64);
        return (p as u32, (p >> 32) as u32);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let maj_and = (a ^ b) & (a ^ c);
        (maj_and, maj_and ^ a)
    }
}

#[inline(always)]
fn add_inline_ab(
    x: u32,
    y: u32,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
    carry_base: usize,
) -> u32 {
    let (sum_word, left, right, carry_aux) = add_carry_parts(x, y);
    or_u32_at_bit(z, carry_base, carry_aux);
    or_u32_at_bit(a, carry_base, left);
    or_u32_at_bit(b, carry_base, right);
    sum_word
}

#[inline(always)]
fn add_alloc_ab(
    x: u32,
    y: u32,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
    sum_base: usize,
    carry_base: usize,
) -> u32 {
    let sum = add_inline_ab(x, y, z, a, b, carry_base);
    or_u32_at_bit(z, sum_base, sum);
    or_u32_at_bit(a, sum_base, sum);
    or_u32_at_bit(b, sum_base, 0xFFFF_FFFF);
    sum
}

fn build_block_ab_packed_into(
    h_in: &[u32; 8],
    m: &[u32; 16],
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    const U64_PER_BLOCK: usize = K / 64;
    debug_assert_eq!(z.len(), U64_PER_BLOCK);
    debug_assert_eq!(a.len(), U64_PER_BLOCK);
    debug_assert_eq!(b.len(), U64_PER_BLOCK);

    or_bit_at(z, Z_CONST_POS);
    or_bit_at(a, Z_CONST_POS);
    or_bit_at(b, Z_CONST_POS);

    for w in 0..H_WORDS {
        let off = h_bit(w, 0);
        let v = h_in[w];
        or_u32_at_bit(z, off, v);
        or_u32_at_bit(a, off, v);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
    }
    for i in 0..M_WORDS {
        let off = m_bit(i, 0);
        let v = m[i];
        or_u32_at_bit(z, off, v);
        or_u32_at_bit(a, off, v);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
    }

    let mut w_sched = [0u32; 64];
    w_sched[..16].copy_from_slice(m);
    const SC0: usize = 0;
    const SC1: usize = CARRIES_PER_ADD;
    const SC2: usize = 2 * CARRIES_PER_ADD;
    for t in 16..64 {
        let mut rz = BitRecord::<2>::new();
        let mut ra = BitRecord::<2>::new();
        let mut rb = BitRecord::<2>::new();

        macro_rules! add_into {
            ($pos:ident, $x:expr, $y:expr) => {{
                let (sum, left, right, carry) = add_carry_parts($x, $y);
                rz.push::<$pos>(carry);
                ra.push::<$pos>(left);
                rb.push::<$pos>(right);
                sum
            }};
        }

        let s_0 = add_into!(SC0, small_sigma1(w_sched[t - 2]), w_sched[t - 7]);
        let s_1 = add_into!(SC1, s_0, small_sigma0(w_sched[t - 15]));
        let w_t = add_into!(SC2, s_1, w_sched[t - 16]);

        let sched_base = sched_carry_bit(t, 0, 0);
        rz.flush(z, sched_base);
        ra.flush(a, sched_base);
        rb.flush(b, sched_base);

        let off = w_bit(t, 0);
        or_u32_at_bit(z, off, w_t);
        or_u32_at_bit(a, off, w_t);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        w_sched[t] = w_t;
    }

    let [
        mut aa,
        mut bb,
        mut cc,
        mut dd,
        mut ee,
        mut ff,
        mut gg,
        mut hh,
    ] = *h_in;
    for r in 0..N_ROUNDS {
        let f_xor_g = ff ^ gg;
        let (ch_and_v, ch_out) = sha256_ch_and_out(ee, ff, gg);
        let off = ch_and_bit(r, 0);
        or_u32_at_bit(z, off, ch_and_v);
        or_u32_at_bit(a, off, ee);
        or_u32_at_bit(b, off, f_xor_g);

        let b_xor_a = bb ^ aa;
        let c_xor_a = cc ^ aa;
        let (maj_and_v, maj_out) = sha256_maj_and_out(aa, bb, cc);
        let off = maj_and_bit(r, 0);
        or_u32_at_bit(z, off, maj_and_v);
        or_u32_at_bit(a, off, b_xor_a);
        or_u32_at_bit(b, off, c_xor_a);

        const RC0: usize = 0;
        const RC1: usize = CARRIES_PER_ADD;
        const RC2: usize = 2 * CARRIES_PER_ADD;
        const RC3: usize = 3 * CARRIES_PER_ADD;
        const RC4: usize = 4 * CARRIES_PER_ADD;
        const RC5: usize = 5 * CARRIES_PER_ADD;
        const RC6: usize = 6 * CARRIES_PER_ADD;
        let mut rz = BitRecord::<4>::new();
        let mut ra = BitRecord::<4>::new();
        let mut rb = BitRecord::<4>::new();

        macro_rules! add_into {
            ($pos:ident, $x:expr, $y:expr) => {{
                let (sum, left, right, carry) = add_carry_parts($x, $y);
                rz.push::<$pos>(carry);
                ra.push::<$pos>(left);
                rb.push::<$pos>(right);
                sum
            }};
        }

        let t1_0 = add_into!(RC0, hh, big_sigma1(ee));
        let t1_1 = add_into!(RC1, t1_0, ch_out);
        let t1_2 = add_into!(RC2, t1_1, SHA256_K[r]);
        let t1 = add_into!(RC3, t1_2, w_sched[r]);
        let off = t1_bit(r, 0);
        or_u32_at_bit(z, off, t1);
        or_u32_at_bit(a, off, t1);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        let t2 = add_into!(RC4, big_sigma0(aa), maj_out);
        let e_new = add_into!(RC5, dd, t1);
        let off = e_new_bit(r, 0);
        or_u32_at_bit(z, off, e_new);
        or_u32_at_bit(a, off, e_new);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);
        let a_new = add_into!(RC6, t1, t2);
        let off = a_new_bit(r, 0);
        or_u32_at_bit(z, off, a_new);
        or_u32_at_bit(a, off, a_new);
        or_u32_at_bit(b, off, 0xFFFF_FFFF);

        let round_base = round_carry_bit(r, 0, 0);
        rz.flush(z, round_base);
        ra.flush(a, round_base);
        rb.flush(b, round_base);

        hh = gg;
        gg = ff;
        ff = ee;
        ee = e_new;
        dd = cc;
        cc = bb;
        bb = aa;
        aa = a_new;
    }

    let final_state = [aa, bb, cc, dd, ee, ff, gg, hh];
    for w in 0..N_OUT_WORDS {
        add_alloc_ab(
            final_state[w],
            h_in[w],
            z,
            a,
            b,
            h_out_bit(w, 0),
            out_carry_bit(w, 0),
        );
    }
}

pub fn generate_witness_with_ab_packed_and_lincheck(
    compressions: &[([u32; 8], [u32; 16])],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    let padding: ([u32; 8], [u32; 16]) = ([0u32; 8], [0u32; 16]);
    super::common::drive_witness_packed_and_lincheck(
        compressions,
        Some(&padding),
        n_blocks_log,
        K_LOG,
        |comp: &([u32; 8], [u32; 16]), z_u64, a_u64, b_u64| {
            let (h_in, m) = comp;
            build_block_ab_packed_into(h_in, m, z_u64, a_u64, b_u64);
        },
    )
}

pub fn min_n_blocks_log(n_compressions: usize) -> usize {
    assert!(n_compressions >= 1);
    let n = n_compressions.max(8);
    n.next_power_of_two().trailing_zeros() as usize
}

pub fn generate_witness(compressions: &[([u32; 8], [u32; 16])], n_blocks_log: usize) -> Vec<bool> {
    use rayon::prelude::*;
    let n_total_blocks = 1usize << n_blocks_log;
    assert!(compressions.len() <= n_total_blocks);

    let mut z = vec![false; n_total_blocks * K];
    z.par_chunks_mut(K)
        .enumerate()
        .for_each(|(block_idx, chunk)| {
            if block_idx >= compressions.len() {
                return;
            }
            let (h_in, m) = &compressions[block_idx];
            let block_witness = build_block_witness(h_in, m);
            chunk.copy_from_slice(&block_witness);
        });
    z
}

#[derive(Clone, Debug)]
pub struct Sha256HybridSetup {
    pub n_compressions: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: flock_core::pcs::PcsParams,
}

impl Sha256HybridSetup {
    pub fn new(n_compressions: usize) -> Self {
        Self::with_log_inv_rate(n_compressions, 1)
    }

    pub fn new_batch_major(n_compressions: usize) -> Self {
        let mut s = Self::new(n_compressions);
        s.r1cs.layout = flock_core::r1cs::WitnessLayout::BatchMajor;
        s
    }

    fn generate_witness_ab(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
    ) -> (
        Vec<flock_core::field::F128>,
        Vec<flock_core::field::F128>,
        Vec<flock_core::field::F128>,
        Vec<u8>,
    ) {
        match self.r1cs.layout {
            flock_core::r1cs::WitnessLayout::RowMajor => {
                generate_witness_with_ab_packed_and_lincheck(compressions, self.n_blocks_log())
            }
            flock_core::r1cs::WitnessLayout::BatchMajor => {
                generate_witness_batch_major(compressions, self.n_blocks_log())
            }
        }
    }

    pub fn with_log_inv_rate(n_compressions: usize, log_inv_rate: usize) -> Self {
        let profile = match log_inv_rate {
            1 => flock_core::pcs::ligerito::LigeritoProfile::Fast,
            2 => flock_core::pcs::ligerito::LigeritoProfile::Slim,
            _ => flock_core::pcs::ligerito::LigeritoProfile::Fast,
        };
        Self::with_profile_and_rate(n_compressions, profile, log_inv_rate)
    }

    pub fn with_profile(
        n_compressions: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        Self::with_profile_and_rate(n_compressions, profile, profile.log_inv_rate())
    }

    fn with_profile_and_rate(
        n_compressions: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
        log_inv_rate: usize,
    ) -> Self {
        assert!(n_compressions >= 1, "n_compressions must be ≥ 1");
        let n_log = min_n_blocks_log(n_compressions);
        let r1cs = build_block_r1cs(n_log);
        r1cs.csc_lincheck_circuit();
        flock_core::scratch::prewarm_prover(r1cs.m);
        let pcs_params = flock_core::pcs::PcsParams {
            m: r1cs.m,
            log_inv_rate,
            log_batch_size: 6,
            profile,
            merkle_hash: Default::default(),
        };
        Self {
            n_compressions,
            r1cs,
            pcs_params,
        }
    }

    pub fn m(&self) -> usize {
        self.r1cs.m
    }
    pub fn n_blocks_log(&self) -> usize {
        self.r1cs.m - self.r1cs.k_log
    }
    pub fn n_block_slots(&self) -> usize {
        1usize << self.n_blocks_log()
    }

    pub fn generate_witness(&self, compressions: &[([u32; 8], [u32; 16])]) -> Vec<bool> {
        assert_eq!(compressions.len(), self.n_compressions);
        generate_witness(compressions, self.n_blocks_log())
    }

    pub fn generate_witness_packed(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
    ) -> Vec<flock_core::field::F128> {
        let (z_packed, _a, _b, _stripe) = self.generate_witness_ab(compressions);
        z_packed
    }

    pub fn prove_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        let z_packed = self.generate_witness_packed(compressions);
        crate::prover::prove_ligerito(&self.r1cs, z_packed, &self.pcs_params, challenger)
    }

    pub fn prove_fast<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
    ) {
        assert_eq!(compressions.len(), self.n_compressions);
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                self.generate_witness_ab(compressions)
            });
        crate::prover::prove_fast_ligerito_from_witness(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            self.r1cs.csc_lincheck_circuit(),
            codeword,
            challenger,
        )
    }

    pub fn prove_fast_timed<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[([u32; 8], [u32; 16])],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        flock_core::pcs::Commitment,
        flock_core::proof::R1csClaim,
        crate::prover::ProvePhaseTimings,
    ) {
        assert_eq!(compressions.len(), self.n_compressions);
        let t0 = std::time::Instant::now();
        let (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck) =
            self.generate_witness_ab(compressions);
        let witness_s = t0.elapsed().as_secs_f64();
        let lc_circuit = self.r1cs.csc_lincheck_circuit();
        let (proof, commitment, claim, mut timings) = crate::prover::prove_fast_ligerito_timed(
            &self.r1cs,
            &self.pcs_params,
            z_packed,
            a_packed_f128,
            b_packed_f128,
            z_packed_lincheck,
            lc_circuit,
            None,
            challenger,
        );
        timings.witness_s = witness_s;
        (proof, commitment, claim, timings)
    }

    pub fn verify<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<flock_core::proof::R1csClaim, flock_core::verifier::VerifyError> {
        flock_core::verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

pub use super::chain_common::{ChainFold, ChainVerifyError};

pub type Compression = ([u32; 8], [u32; 16]);

pub const CHAIN_LAYOUT: super::chain_common::ChainLayout = super::chain_common::ChainLayout {
    k_log: K_LOG,
    k_skip: K_SKIP,
    region_log: 8,
    region_bits: 256,
    input_byte_off: H_BASE / 8,
    output_byte_off: H_OUT_BASE / 8,
};

pub fn cv_to_phys_bits(cv: &[u32; 8]) -> Vec<bool> {
    let mut phys = vec![false; 256];
    for w in 0..8 {
        for b in 0..WORD_BITS {
            phys[WORD_BITS * w + b] = (cv[w] >> b) & 1 == 1;
        }
    }
    phys
}

pub fn sha256_compress(h_in: &[u32; 8], m: &[u32; 16]) -> [u32; 8] {
    let mut w = [0u32; 64];
    w[..16].copy_from_slice(m);
    for t in 16..64 {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = s1
            .wrapping_add(w[t - 7])
            .wrapping_add(s0)
            .wrapping_add(w[t - 16]);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = [
        h_in[0], h_in[1], h_in[2], h_in[3], h_in[4], h_in[5], h_in[6], h_in[7],
    ];
    for r in 0..N_ROUNDS {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[r])
            .wrapping_add(w[r]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    [
        h_in[0].wrapping_add(a),
        h_in[1].wrapping_add(b),
        h_in[2].wrapping_add(c),
        h_in[3].wrapping_add(d),
        h_in[4].wrapping_add(e),
        h_in[5].wrapping_add(f),
        h_in[6].wrapping_add(g),
        h_in[7].wrapping_add(h),
    ]
}

impl Sha256HybridSetup {
    pub fn prove_chain<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[Compression],
        challenger: &mut Ch,
    ) -> (
        super::chain_common::ChainProofLigerito,
        flock_core::pcs::Commitment,
    ) {
        assert_eq!(compressions.len(), self.n_compressions);
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "prove_chain requires n_compressions to exactly fill n_block_slots \
             (no padding); got n_compressions={}, n_block_slots={}. Use a \
             power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        let (z_packed, a_packed, b_packed, z_lincheck) = self.generate_witness_ab(compressions);
        super::chain_common::prove_chain_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &CHAIN_LAYOUT,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    pub fn verify_chain<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &super::chain_common::ChainProofLigerito,
        cv_0: &[u32; 8],
        cv_last: &[u32; 8],
        challenger: &mut Ch,
    ) -> Result<(), ChainVerifyError> {
        assert_eq!(self.n_compressions, self.n_block_slots());
        let n_log = self.n_blocks_log();
        let cv0_phys = cv_to_phys_bits(cv_0);
        let cvlast_phys = cv_to_phys_bits(cv_last);
        super::chain_common::verify_chain_ligerito_generic(
            &self.r1cs,
            &CHAIN_LAYOUT,
            commitment,
            proof,
            n_log,
            &cv0_phys,
            &cvlast_phys,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

pub use super::merkle_path_common::{MerklePathProofLigerito, MerklePathVerifyError};

pub const MERKLE_LAYOUT: super::merkle_path_common::MerkleLayout =
    super::merkle_path_common::MerkleLayout {
        k_log: K_LOG,
        k_skip: K_SKIP,
        region_log: 8,
        region_bits: 256,
        slot_base_byte_off: 0,
        z_slot: 1,
        x_l_slot: 2,
        x_r_slot: 3,
    };

pub fn hash_to_phys_bits(h: &[u32; 8]) -> Vec<bool> {
    cv_to_phys_bits(h)
}

impl Sha256HybridSetup {
    pub fn prove_merkle_path_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        compressions: &[Compression],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> (MerklePathProofLigerito, flock_core::pcs::Commitment) {
        assert_eq!(compressions.len(), self.n_compressions);
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "prove_merkle_path_ligerito requires n_compressions to exactly fill \
             n_block_slots (no padding); got n_compressions={}, \
             n_block_slots={}. Use a power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let (z_packed, a_packed, b_packed, z_lincheck) = self.generate_witness_ab(compressions);
        super::merkle_path_common::prove_merkle_paths_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &MERKLE_LAYOUT,
            0,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    pub fn verify_merkle_path_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        commitment: &flock_core::pcs::Commitment,
        proof: &MerklePathProofLigerito,
        leaf: &[u32; 8],
        root: &[u32; 8],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> Result<(), MerklePathVerifyError> {
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "verify_merkle_path_ligerito requires n_compressions to exactly fill \
             n_block_slots (no padding)",
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let n_log = self.n_blocks_log();
        let leaf_phys = hash_to_phys_bits(leaf);
        let root_phys = hash_to_phys_bits(root);
        super::merkle_path_common::verify_merkle_paths_ligerito_generic(
            &self.r1cs,
            &MERKLE_LAYOUT,
            0,
            commitment,
            proof,
            n_log,
            &[leaf_phys.as_slice()],
            &root_phys,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }

    pub fn prove_merkle_paths_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        path_log: usize,
        compressions: &[Compression],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> (MerklePathProofLigerito, flock_core::pcs::Commitment) {
        assert_eq!(compressions.len(), self.n_compressions);
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "prove_merkle_paths_ligerito requires n_compressions to exactly fill \
             n_block_slots (no padding); got n_compressions={}, \
             n_block_slots={}. Use a power-of-2 ≥ 8.",
            self.n_compressions,
            self.n_block_slots(),
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        assert!(
            path_log <= self.n_blocks_log(),
            "path_log {} > n_blocks_log {}",
            path_log,
            self.n_blocks_log(),
        );
        let (z_packed, a_packed, b_packed, z_lincheck) = self.generate_witness_ab(compressions);
        super::merkle_path_common::prove_merkle_paths_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &MERKLE_LAYOUT,
            path_log,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            challenger,
        )
    }

    pub fn verify_merkle_paths_ligerito<Ch: flock_core::challenger::Challenger>(
        &self,
        path_log: usize,
        commitment: &flock_core::pcs::Commitment,
        proof: &MerklePathProofLigerito,
        leaves: &[[u32; 8]],
        root: &[u32; 8],
        b_bits: &[bool],
        challenger: &mut Ch,
    ) -> Result<(), MerklePathVerifyError> {
        assert_eq!(
            self.n_compressions,
            self.n_block_slots(),
            "verify_merkle_paths_ligerito requires n_compressions to exactly fill \
             n_block_slots (no padding)",
        );
        assert_eq!(
            b_bits.len(),
            self.n_block_slots(),
            "bit vector length mismatch"
        );
        let n_paths = 1usize << path_log;
        assert_eq!(leaves.len(), n_paths, "leaves must have length 2^path_log");
        let n_log = self.n_blocks_log();
        let leaves_phys: Vec<Vec<bool>> = leaves.iter().map(hash_to_phys_bits).collect();
        let leaves_phys_refs: Vec<&[bool]> = leaves_phys.iter().map(|v| v.as_slice()).collect();
        let root_phys = hash_to_phys_bits(root);
        super::merkle_path_common::verify_merkle_paths_ligerito_generic(
            &self.r1cs,
            &MERKLE_LAYOUT,
            path_log,
            commitment,
            proof,
            n_log,
            &leaves_phys_refs,
            &root_phys,
            b_bits,
            self.r1cs.csc_lincheck_circuit(),
            &self.pcs_params,
            challenger,
        )
    }
}

use super::common::{BM_V, BmRow, add_carry_parts_v, or_bit_row, or_u32_row};

#[inline(always)]
fn map_v(x: &[u32; BM_V], f: impl Fn(u32) -> u32) -> [u32; BM_V] {
    std::array::from_fn(|j| f(x[j]))
}
#[inline(always)]
fn xor_v(x: &[u32; BM_V], y: &[u32; BM_V]) -> [u32; BM_V] {
    std::array::from_fn(|j| x[j] ^ y[j])
}
#[inline(always)]
fn and_v(x: &[u32; BM_V], y: &[u32; BM_V]) -> [u32; BM_V] {
    std::array::from_fn(|j| x[j] & y[j])
}

struct BmRows<'a> {
    z: &'a mut [BmRow],
    a: &'a mut [BmRow],
    b: &'a mut [BmRow],
}

#[inline(always)]
fn bm_write_lin(rows: &mut BmRows<'_>, bit: usize, vals: &[u32; BM_V]) {
    or_u32_row(rows.z, bit, vals);
    or_u32_row(rows.a, bit, vals);
    or_u32_row(rows.b, bit, &[0xFFFF_FFFF; BM_V]);
}

#[inline(always)]
fn bm_add_inline(
    rows: &mut BmRows<'_>,
    x: &[u32; BM_V],
    y: &[u32; BM_V],
    carry_bit: usize,
) -> [u32; BM_V] {
    let (sum, left, right, carry) = add_carry_parts_v(x, y);
    or_u32_row(rows.z, carry_bit, &carry);
    or_u32_row(rows.a, carry_bit, &left);
    or_u32_row(rows.b, carry_bit, &right);
    sum
}

fn build_group_batch_major(
    inputs: [&([u32; 8], [u32; 16]); BM_V],
    rz: &mut [BmRow],
    ra: &mut [BmRow],
    rb: &mut [BmRow],
) {
    let mut rows = BmRows {
        z: rz,
        a: ra,
        b: rb,
    };
    let h_in: [[u32; BM_V]; 8] = std::array::from_fn(|w| std::array::from_fn(|j| inputs[j].0[w]));
    let m: [[u32; BM_V]; 16] = std::array::from_fn(|i| std::array::from_fn(|j| inputs[j].1[i]));

    or_bit_row(rows.z, Z_CONST_POS);
    or_bit_row(rows.a, Z_CONST_POS);
    or_bit_row(rows.b, Z_CONST_POS);

    for w in 0..H_WORDS {
        bm_write_lin(&mut rows, h_bit(w, 0), &h_in[w]);
    }
    for i in 0..M_WORDS {
        bm_write_lin(&mut rows, m_bit(i, 0), &m[i]);
    }

    let mut w_sched: Vec<[u32; BM_V]> = Vec::with_capacity(64);
    w_sched.extend_from_slice(&m);
    for t in 16..64 {
        let s_0 = bm_add_inline(
            &mut rows,
            &map_v(&w_sched[t - 2], small_sigma1),
            &w_sched[t - 7],
            sched_carry_bit(t, 0, 0),
        );
        let s_1 = bm_add_inline(
            &mut rows,
            &s_0,
            &map_v(&w_sched[t - 15], small_sigma0),
            sched_carry_bit(t, 1, 0),
        );
        let w_t = bm_add_inline(&mut rows, &s_1, &w_sched[t - 16], sched_carry_bit(t, 2, 0));
        bm_write_lin(&mut rows, w_bit(t, 0), &w_t);
        w_sched.push(w_t);
    }

    let mut aa = h_in[0];
    let mut bb = h_in[1];
    let mut cc = h_in[2];
    let mut dd = h_in[3];
    let mut ee = h_in[4];
    let mut ff = h_in[5];
    let mut gg = h_in[6];
    let mut hh = h_in[7];
    for r in 0..N_ROUNDS {
        let f_xor_g = xor_v(&ff, &gg);
        let ch_and_v = and_v(&ee, &f_xor_g);
        or_u32_row(rows.z, ch_and_bit(r, 0), &ch_and_v);
        or_u32_row(rows.a, ch_and_bit(r, 0), &ee);
        or_u32_row(rows.b, ch_and_bit(r, 0), &f_xor_g);
        let ch_out = xor_v(&ch_and_v, &gg);

        let b_xor_a = xor_v(&bb, &aa);
        let c_xor_a = xor_v(&cc, &aa);
        let maj_and_v = and_v(&b_xor_a, &c_xor_a);
        or_u32_row(rows.z, maj_and_bit(r, 0), &maj_and_v);
        or_u32_row(rows.a, maj_and_bit(r, 0), &b_xor_a);
        or_u32_row(rows.b, maj_and_bit(r, 0), &c_xor_a);
        let maj_out = xor_v(&maj_and_v, &aa);

        let k_r = [SHA256_K[r]; BM_V];
        let t1_0 = bm_add_inline(
            &mut rows,
            &hh,
            &map_v(&ee, big_sigma1),
            round_carry_bit(r, 0, 0),
        );
        let t1_1 = bm_add_inline(&mut rows, &t1_0, &ch_out, round_carry_bit(r, 1, 0));
        let t1_2 = bm_add_inline(&mut rows, &t1_1, &k_r, round_carry_bit(r, 2, 0));
        let t1 = bm_add_inline(&mut rows, &t1_2, &w_sched[r], round_carry_bit(r, 3, 0));
        bm_write_lin(&mut rows, t1_bit(r, 0), &t1);

        let t2 = bm_add_inline(
            &mut rows,
            &map_v(&aa, big_sigma0),
            &maj_out,
            round_carry_bit(r, 4, 0),
        );
        let e_new = bm_add_inline(&mut rows, &dd, &t1, round_carry_bit(r, 5, 0));
        bm_write_lin(&mut rows, e_new_bit(r, 0), &e_new);
        let a_new = bm_add_inline(&mut rows, &t1, &t2, round_carry_bit(r, 6, 0));
        bm_write_lin(&mut rows, a_new_bit(r, 0), &a_new);

        hh = gg;
        gg = ff;
        ff = ee;
        ee = e_new;
        dd = cc;
        cc = bb;
        bb = aa;
        aa = a_new;
    }

    let final_state = [aa, bb, cc, dd, ee, ff, gg, hh];
    for w in 0..N_OUT_WORDS {
        let sum = bm_add_inline(&mut rows, &final_state[w], &h_in[w], out_carry_bit(w, 0));
        bm_write_lin(&mut rows, h_out_bit(w, 0), &sum);
    }
}

pub fn generate_witness_batch_major(
    compressions: &[([u32; 8], [u32; 16])],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    let padding: ([u32; 8], [u32; 16]) = ([0u32; 8], [0u32; 16]);
    super::common::drive_witness_batch_major(
        compressions,
        &padding,
        n_blocks_log,
        K_LOG,
        USEFUL_BITS,
        build_group_batch_major,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (z ^ (z >> 31)) as u32
        }
        fn next_block(&mut self) -> [u32; 16] {
            std::array::from_fn(|_| self.next_u32())
        }
    }

    #[test]
    fn batch_major_witness_matches_row_major_transposed() {
        for (n_inputs, n_log) in [(8usize, 3usize), (11, 4)] {
            let mut rng = Rng::new(0xBA7C_5A + n_log as u64);
            let inputs: Vec<([u32; 8], [u32; 16])> = (0..n_inputs)
                .map(|_| (std::array::from_fn(|_| rng.next_u32()), rng.next_block()))
                .collect();

            let (z_r, a_r, b_r, stripe_r) =
                generate_witness_with_ab_packed_and_lincheck(&inputs, n_log);
            let (z_b, a_b, b_b, stripe_b) = generate_witness_batch_major(&inputs, n_log);

            assert_eq!(stripe_b, stripe_r, "stripe diverged (n_log={n_log})");

            let chunks_per_block = K / 128;
            let transpose = |row: &[flock_core::field::F128]| {
                let mut out = vec![flock_core::field::F128::ZERO; row.len()];
                for o in 0..1usize << n_log {
                    for c in 0..chunks_per_block {
                        out[(c << n_log) + o] = row[o * chunks_per_block + c];
                    }
                }
                out
            };
            assert_eq!(z_b, transpose(&z_r), "z diverged (n_log={n_log})");
            assert_eq!(a_b, transpose(&a_r), "a diverged (n_log={n_log})");
            assert_eq!(b_b, transpose(&b_r), "b diverged (n_log={n_log})");
        }
    }

    #[test]
    #[ignore]
    fn batch_major_prove_fast_roundtrip() {
        use flock_core::challenger::FsChallenger;

        let setup = Sha256HybridSetup::new_batch_major(128);
        let mut rng = Rng::new(0xBA7C_F012);
        let inputs: Vec<([u32; 8], [u32; 16])> = (0..128)
            .map(|_| (std::array::from_fn(|_| rng.next_u32()), rng.next_block()))
            .collect();

        let mut ch_p = FsChallenger::new(b"flock-lig-batch-major-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&inputs, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-lig-batch-major-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("batch-major verifier rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        let mut bad = proof.clone();
        bad.zerocheck.final_a_eval.lo ^= 1;
        let mut ch = FsChallenger::new(b"flock-lig-batch-major-v0");
        assert!(
            setup.verify(&commitment, &bad, &mut ch).is_err(),
            "tampered batch-major proof accepted"
        );
    }

    fn satisfies_singleblock(
        a: &SparseBinaryMatrix,
        b: &SparseBinaryMatrix,
        z: &[bool],
    ) -> Result<(), usize> {
        for i in 0..a.rows.len() {
            let av = a.rows[i].iter().fold(false, |acc, &s| acc ^ z[s]);
            let bv = b.rows[i].iter().fold(false, |acc, &s| acc ^ z[s]);
            if (av && bv) != z[i] {
                return Err(i);
            }
        }
        Ok(())
    }

    #[test]
    fn useful_bits_matches_constants() {
        assert_eq!(H_BASE, 0);
        assert_eq!(H_OUT_BASE, 256);
        assert_eq!(M_BASE, 512);
        assert_eq!(CH_AND_BASE, 1024);
        assert_eq!(MAJ_AND_BASE, 3072);
        assert_eq!(ROUND_CARRY_BASE, 5120);
        assert_eq!(W_BASE, 19008);
        assert_eq!(SCHED_CARRY_BASE, 20544);
        assert_eq!(T1_BASE, 25008);
        assert_eq!(E_NEW_BASE, 27056);
        assert_eq!(A_NEW_BASE, 29104);
        assert_eq!(OUT_CARRY_BASE, 31152);
        assert_eq!(Z_CONST_POS, 31400);
        assert_eq!(USEFUL_BITS, 31401);
        assert!(USEFUL_BITS <= K);
    }

    #[test]
    fn block_witness_satisfies_matrix_and_matches_reference() {
        let (a, b) = build_matrices();
        let mut rng = Rng::new(0xC0FFEE_5A55);
        let cases: [([u32; 8], [u32; 16]); 4] = [
            (SHA256_IV, [0u32; 16]),
            (
                SHA256_IV,
                [
                    0x6162_6380,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0x0000_0018,
                ],
            ),
            (SHA256_IV, rng.next_block()),
            (std::array::from_fn(|_| rng.next_u32()), rng.next_block()),
        ];
        for (h_in, m) in cases {
            let z = build_block_witness(&h_in, &m);
            assert!(
                satisfies_singleblock(&a, &b, &z).is_ok(),
                "R1CS not satisfied for h_in={:08x?}, m[0]={:08x}",
                h_in,
                m[0]
            );
            assert_eq!(
                read_h_out(&z),
                sha256_compress(&h_in, &m),
                "H_out mismatch for h_in={:08x?}, m[0]={:08x}",
                h_in,
                m[0]
            );
        }
    }

    #[test]
    fn lincheck_circuit_matches_sparse() {
        use flock_core::lincheck::{LincheckCircuit, SparseMatrixCircuit};

        let mut rng = Rng::new(0x5_4A2_CCA1);
        let (a_0, b_0) = build_matrices();
        let sparse = SparseMatrixCircuit::new(&a_0, &b_0);
        let walker = Sha2LincheckCircuit;
        assert_eq!(sparse.n_cols(), walker.n_cols());

        let n_cols = walker.n_cols();
        let alpha = F128 {
            lo: ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
            hi: ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
        };
        let eq_inner: Vec<F128> = (0..n_cols)
            .map(|_| F128 {
                lo: ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
                hi: ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64,
            })
            .collect();

        let expected = sparse.fold_alpha_batched(alpha, &eq_inner);
        let got = walker.fold_alpha_batched(alpha, &eq_inner);
        for c in 0..n_cols {
            assert_eq!(expected[c], got[c], "comb mismatch at col {c}");
        }

        let csc = flock_core::lincheck::CscCircuit::from_matrices(&a_0, &b_0);
        let got_csc = csc.fold_alpha_batched(alpha, &eq_inner);
        assert_eq!(expected, got_csc, "CSC fold mismatch");
    }

    #[test]
    #[ignore]
    fn prove_fast_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let mut rng = Rng::new(0x5_a2_211e);
        let n = 128;
        let compressions: Vec<([u32; 8], [u32; 16])> =
            (0..n).map(|_| (SHA256_IV, rng.next_block())).collect();
        let setup = Sha256HybridSetup::new(n);
        let mut ch_p = FsChallenger::new(b"flock-sha2-lig-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&compressions, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-sha2-lig-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("ligerito verify rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    #[test]
    fn prove_ligerito_generic_matches_prove_fast() {
        use flock_core::challenger::FsChallenger;
        let n = 128;
        let setup = Sha256HybridSetup::new(n);
        let mut rng = Rng::new(0x5A2_63112);
        let comps: Vec<Compression> = (0..n)
            .map(|_| {
                (
                    std::array::from_fn(|_| rng.next_u32()),
                    std::array::from_fn(|_| rng.next_u32()),
                )
            })
            .collect();
        let mut ch_f = FsChallenger::new(b"flock-sha2-gvf");
        let (proof_f, commit_f, claim_f) = setup.prove_fast(&comps, &mut ch_f);
        let mut ch_g = FsChallenger::new(b"flock-sha2-gvf");
        let (proof_g, commit_g, claim_g) = setup.prove_ligerito(&comps, &mut ch_g);
        assert_eq!(commit_f.root, commit_g.root);
        assert_eq!(claim_f, claim_g);
        assert_eq!(
            bincode::serialize(&proof_f).unwrap(),
            bincode::serialize(&proof_g).unwrap(),
            "generic and fused Ligerito proofs must be byte-identical"
        );
        let mut ch_v = FsChallenger::new(b"flock-sha2-gvf");
        let claim_v = setup
            .verify(&commit_g, &proof_g, &mut ch_v)
            .expect("verify ok");
        assert_eq!(claim_v, claim_g);
    }

    #[test]
    fn const_pin_all_zero_rejected() {
        use flock_core::challenger::FsChallenger;

        let n = 120;
        let setup = Sha256HybridSetup::new(n);

        let mut rng = Rng::new(0x5EED_50A2);
        let compressions: Vec<([u32; 8], [u32; 16])> =
            (0..n).map(|_| (SHA256_IV, rng.next_block())).collect();
        let mut ch_p = FsChallenger::new(b"honest");
        let (proof, commit, claim_p) = setup.prove_fast(&compressions, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"honest");
        let claim_v = setup
            .verify(&commit, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("honest padded proof rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        let zeros: Vec<([u32; 8], [u32; 16])> = vec![([0u32; 8], [0u32; 16]); n];
        let (mut z, mut a, mut b, mut zlc) =
            generate_witness_with_ab_packed_and_lincheck(&zeros, setup.n_blocks_log());
        z.iter_mut()
            .for_each(|v| *v = flock_core::field::F128::ZERO);
        a.iter_mut()
            .for_each(|v| *v = flock_core::field::F128::ZERO);
        b.iter_mut()
            .for_each(|v| *v = flock_core::field::F128::ZERO);
        zlc.iter_mut().for_each(|v| *v = 0);
        let circuit = setup.r1cs.csc_lincheck_circuit();
        let mut ch_p = FsChallenger::new(b"poc");
        let (proof, commit, _) = crate::prover::prove_fast_ligerito_from_witness(
            &setup.r1cs,
            &setup.pcs_params,
            z,
            a,
            b,
            zlc,
            circuit,
            None,
            &mut ch_p,
        );
        let mut ch_v = FsChallenger::new(b"poc");
        let res = setup.verify(&commit, &proof, &mut ch_v);
        assert!(
            matches!(res, Err(flock_core::verifier::VerifyError::Lincheck(_))),
            "all-zero witness must be rejected by the constant-wire pin; got {res:?}"
        );
    }

    #[test]
    fn block_r1cs_satisfies_for_one_block() {
        let r1cs = build_block_r1cs(3);
        let n_blocks = 1 << 3;
        let z_block = build_block_witness(&SHA256_IV, &[0u32; 16]);
        let mut z = vec![false; n_blocks * K];
        z[..K].copy_from_slice(&z_block);
        assert!(r1cs.satisfies(&z));
    }

    fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
        let mut rng = Rng::new(seed);
        let mut cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
        let cv0 = cv;
        let mut blocks = Vec::with_capacity(n);
        for _ in 0..n {
            let m = rng.next_block();
            blocks.push((cv, m));
            cv = sha256_compress(&cv, &m);
        }
        (blocks, cv0, cv)
    }

    #[test]
    #[ignore]
    fn prove_chain_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(128);
        let (blocks, cv_0, cv_last) = honest_chain(setup.n_compressions, 0x511_3E_C0DE);
        let mut ch = FsChallenger::new(b"sha2-chain-lig");
        let (proof, commitment) = setup.prove_chain(&blocks, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-chain-lig");
        setup
            .verify_chain(&commitment, &proof, &cv_0, &cv_last, &mut chv)
            .expect("ligerito chain must verify");

        let mut mutated_last = cv_last;
        mutated_last[0] ^= 1;
        let mut chv = FsChallenger::new(b"sha2-chain-lig");
        let res = setup.verify_chain(&commitment, &proof, &cv_0, &mutated_last, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong cv_last");

        let mut mutated_0 = cv_0;
        mutated_0[7] ^= 1 << 31;
        let mut chv = FsChallenger::new(b"sha2-chain-lig");
        let res = setup.verify_chain(&commitment, &proof, &mutated_0, &cv_last, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong cv_0");
    }

    #[test]
    #[should_panic(expected = "prove_chain requires n_compressions to exactly fill n_block_slots")]
    fn prove_chain_panics_on_padding() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(5);
        assert_eq!(setup.n_compressions, 5);
        assert_eq!(setup.n_block_slots(), 8);
        let (blocks, _, _) = honest_chain(setup.n_compressions, 0xDEADBEEF);
        let mut ch = FsChallenger::new(b"sha2-chain-test");
        let _ = setup.prove_chain(&blocks, &mut ch);
    }

    fn honest_merkle_path(
        n: usize,
        seed: u64,
    ) -> (Vec<Compression>, [u32; 8], [u32; 8], Vec<bool>) {
        let mut rng = Rng::new(seed);
        let leaf: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
        let mut b_bits = vec![false; n];
        for bit in b_bits.iter_mut().skip(1) {
            *bit = rng.next_u32() & 1 == 1;
        }
        let mut blocks = Vec::with_capacity(n);
        let mut current = leaf;
        for i in 0..n {
            let sibling: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = if !b_bits[i] {
                let mut m = [0u32; 16];
                m[..8].copy_from_slice(&current);
                m[8..].copy_from_slice(&sibling);
                m
            } else {
                let mut m = [0u32; 16];
                m[..8].copy_from_slice(&sibling);
                m[8..].copy_from_slice(&current);
                m
            };
            blocks.push((SHA256_IV, m));
            current = sha256_compress(&SHA256_IV, &m);
        }
        let root = current;
        (blocks, leaf, root, b_bits)
    }

    fn honest_merkle_paths_identical(
        total: usize,
        path_log: usize,
        seed: u64,
    ) -> (Vec<Compression>, Vec<[u32; 8]>, [u32; 8], Vec<bool>) {
        let n_paths = 1usize << path_log;
        let l = total / n_paths;
        assert_eq!(n_paths * l, total, "total must factor as n_paths · L");

        let (path_blocks, leaf, root, path_b) = honest_merkle_path(l, seed);
        let mut compressions = Vec::with_capacity(total);
        let mut b_bits = Vec::with_capacity(total);
        for _ in 0..n_paths {
            compressions.extend_from_slice(&path_blocks);
            b_bits.extend_from_slice(&path_b);
        }
        let leaves = vec![leaf; n_paths];
        (compressions, leaves, root, b_bits)
    }

    #[test]
    fn multi_path_log0_matches_single_path() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(128);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0xC0FFEE);

        let mut ch_single = FsChallenger::new(b"sha2-merkle-equiv");
        let (proof_single, commit_single) =
            setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch_single);

        let mut ch_multi = FsChallenger::new(b"sha2-merkle-equiv");
        let (proof_multi, commit_multi) =
            setup.prove_merkle_paths_ligerito(0, &blocks, &b, &mut ch_multi);

        let cb_single = bincode::serialize(&commit_single).unwrap();
        let cb_multi = bincode::serialize(&commit_multi).unwrap();
        assert_eq!(cb_single, cb_multi, "commitments must match");
        let bytes_single = bincode::serialize(&proof_single).unwrap();
        let bytes_multi = bincode::serialize(&proof_multi).unwrap();
        assert_eq!(
            bytes_single, bytes_multi,
            "path_log=0 must serialize identically to single-path"
        );

        let mut chv = FsChallenger::new(b"sha2-merkle-equiv");
        setup
            .verify_merkle_paths_ligerito(
                0,
                &commit_multi,
                &proof_multi,
                &[leaf],
                &root,
                &b,
                &mut chv,
            )
            .expect("multi-path with path_log=0 must verify");
    }

    #[test]
    fn prove_merkle_paths_ligerito_roundtrip_p4() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(128);
        let (blocks, leaves, root, b) =
            honest_merkle_paths_identical(setup.n_compressions, 2, 0xF00D);
        let mut ch = FsChallenger::new(b"sha2-merkle-paths-p4");
        let (proof, commitment) = setup.prove_merkle_paths_ligerito(2, &blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-p4");
        setup
            .verify_merkle_paths_ligerito(2, &commitment, &proof, &leaves, &root, &b, &mut chv)
            .expect("honest 4-path proof must verify");
    }

    #[test]
    fn prove_merkle_path_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(128);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0x5EED_F00D);
        let mut ch = FsChallenger::new(b"sha2-merkle-lig");
        let (proof, commitment) = setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-lig");
        setup
            .verify_merkle_path_ligerito(&commitment, &proof, &leaf, &root, &b, &mut chv)
            .expect("ligerito merkle path must verify");

        let mut bad_leaf = leaf;
        bad_leaf[0] ^= 1;
        let mut chv = FsChallenger::new(b"sha2-merkle-lig");
        let res =
            setup.verify_merkle_path_ligerito(&commitment, &proof, &bad_leaf, &root, &b, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong leaf");

        let mut bad_root = root;
        bad_root[7] ^= 1 << 31;
        let mut chv = FsChallenger::new(b"sha2-merkle-lig");
        let res =
            setup.verify_merkle_path_ligerito(&commitment, &proof, &leaf, &bad_root, &b, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong root");

        let mut bad_b = b.clone();
        bad_b[1] = !bad_b[1];
        let mut chv = FsChallenger::new(b"sha2-merkle-lig");
        let res =
            setup.verify_merkle_path_ligerito(&commitment, &proof, &leaf, &root, &bad_b, &mut chv);
        assert!(res.is_err(), "verifier must reject wrong bit vector");
    }

    #[test]
    #[ignore]
    fn batch_major_prove_merkle_path_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;

        let setup = Sha256HybridSetup::new_batch_major(128);
        let (blocks, leaf, root, b) = honest_merkle_path(setup.n_compressions, 0xBA7C_BEEF);
        let mut ch = FsChallenger::new(b"sha2-merkle-batch-major");
        let (proof, commitment) = setup.prove_merkle_path_ligerito(&blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-batch-major");
        setup
            .verify_merkle_path_ligerito(&commitment, &proof, &leaf, &root, &b, &mut chv)
            .expect("batch-major Merkle path must verify");

        let mut bad_leaf = leaf;
        bad_leaf[0] ^= 1;
        let mut chv = FsChallenger::new(b"sha2-merkle-batch-major");
        assert!(
            setup
                .verify_merkle_path_ligerito(&commitment, &proof, &bad_leaf, &root, &b, &mut chv,)
                .is_err(),
            "wrong leaf accepted under batch-major"
        );
    }

    #[test]
    fn prove_merkle_paths_ligerito_roundtrip_p2() {
        use flock_core::challenger::FsChallenger;
        let setup = Sha256HybridSetup::new(128);
        let (blocks, leaves, root, b) =
            honest_merkle_paths_identical(setup.n_compressions, 1, 0xC0DE_BABE);
        let mut ch = FsChallenger::new(b"sha2-merkle-paths-lig");
        let (proof, commitment) = setup.prove_merkle_paths_ligerito(1, &blocks, &b, &mut ch);
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-lig");
        setup
            .verify_merkle_paths_ligerito(1, &commitment, &proof, &leaves, &root, &b, &mut chv)
            .expect("ligerito 2-path proof must verify");

        let mut bad_leaves = leaves.clone();
        bad_leaves[0][0] ^= 1;
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-lig");
        let res = setup.verify_merkle_paths_ligerito(
            1,
            &commitment,
            &proof,
            &bad_leaves,
            &root,
            &b,
            &mut chv,
        );
        assert!(res.is_err(), "verifier must reject wrong leaf in path 0");

        let mut bad_root = root;
        bad_root[7] ^= 1 << 31;
        let mut chv = FsChallenger::new(b"sha2-merkle-paths-lig");
        let res = setup.verify_merkle_paths_ligerito(
            1,
            &commitment,
            &proof,
            &leaves,
            &bad_root,
            &b,
            &mut chv,
        );
        assert!(res.is_err(), "verifier must reject wrong shared root");
    }

    #[test]
    fn cv_to_phys_bits_roundtrips() {
        let cv: [u32; 8] = [
            0x01234567, 0x89ABCDEF, 0xDEADBEEF, 0xFEEDC0DE, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0,
            0x0F1E2D3C,
        ];
        let phys = cv_to_phys_bits(&cv);
        assert_eq!(phys.len(), 256);
        for w in 0..8 {
            let mut recovered = 0u32;
            for b in 0..WORD_BITS {
                if phys[WORD_BITS * w + b] {
                    recovered |= 1 << b;
                }
            }
            assert_eq!(recovered, cv[w], "word {w} mismatch");
        }
    }
}
