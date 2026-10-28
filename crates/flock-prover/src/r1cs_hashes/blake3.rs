use super::common::xor_dedup;
use flock_core::challenger::Challenger;
use flock_core::field::F128;
use flock_core::pcs::{Commitment, PcsParams};
use flock_core::proof::R1csClaim;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};
use flock_core::verifier;

pub const K_LOG: usize = 14;
pub const K: usize = 1 << K_LOG;
pub const K_SKIP: usize = 6;

pub const N_ROUNDS: usize = 7;
pub const N_G_PER_ROUND: usize = 8;
pub const N_G: usize = N_ROUNDS * N_G_PER_ROUND;
pub const WORD_BITS: usize = 32;

pub const CARRY_BITS_PER_ADD: usize = WORD_BITS - 1;
pub const ADDS_PER_G: usize = 6;
pub const LIN_WORDS_PER_G: usize = 2;
pub const G_STRIDE: usize = ADDS_PER_G * CARRY_BITS_PER_ADD + LIN_WORDS_PER_G * WORD_BITS;

pub const BLAKE3_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

pub const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

pub const G_LANES: [[usize; 4]; N_G_PER_ROUND] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

pub const G_MSG_IDX: [[usize; 2]; N_G_PER_ROUND] = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [12, 13],
    [14, 15],
];

pub const SLOT_BITS: usize = 256;
pub const CV_BASE: usize = 0;
pub const OUT_LO_BASE: usize = SLOT_BITS;
pub const Z_CONST_POS: usize = 2 * SLOT_BITS;
pub const M_BASE: usize = Z_CONST_POS + 1;
pub const T_LO_BASE: usize = M_BASE + 16 * WORD_BITS;
pub const T_HI_BASE: usize = T_LO_BASE + WORD_BITS;
pub const BLEN_BASE: usize = T_HI_BASE + WORD_BITS;
pub const FLAGS_BASE: usize = BLEN_BASE + WORD_BITS;
pub const GS_BASE: usize = FLAGS_BASE + WORD_BITS;
pub const OUT_HI_BASE: usize = GS_BASE + N_G * G_STRIDE;
pub const USEFUL_BITS: usize = OUT_HI_BASE + 8 * WORD_BITS;

const ADD_TMP0: usize = 0;
const ADD_A1: usize = 1;
const ADD_C1: usize = 2;
const ADD_TMP1: usize = 3;
const ADD_A2: usize = 4;
const ADD_C2: usize = 5;
const LIN_B_NEW: usize = 0;
const LIN_D_NEW: usize = 1;

#[inline]
fn cv_bit(w: usize, b: usize) -> usize {
    debug_assert!(w < 8 && b < WORD_BITS);
    CV_BASE + WORD_BITS * w + b
}
#[inline]
fn m_bit(i: usize, b: usize) -> usize {
    debug_assert!(i < 16 && b < WORD_BITS);
    M_BASE + WORD_BITS * i + b
}
#[inline]
fn g_add_carry_bit(g: usize, add_idx: usize, b: usize) -> usize {
    debug_assert!(g < N_G && add_idx < ADDS_PER_G && b < CARRY_BITS_PER_ADD);
    GS_BASE + G_STRIDE * g + CARRY_BITS_PER_ADD * add_idx + b
}
#[inline]
fn g_lin_bit(g: usize, which: usize, b: usize) -> usize {
    debug_assert!(g < N_G && which < LIN_WORDS_PER_G && b < WORD_BITS);
    GS_BASE + G_STRIDE * g + ADDS_PER_G * CARRY_BITS_PER_ADD + WORD_BITS * which + b
}
#[inline]
fn out_lo_bit(w: usize, b: usize) -> usize {
    debug_assert!(w < 8 && b < WORD_BITS);
    OUT_LO_BASE + WORD_BITS * w + b
}
#[inline]
fn out_hi_bit(w: usize, b: usize) -> usize {
    debug_assert!(w < 8 && b < WORD_BITS);
    OUT_HI_BASE + WORD_BITS * w + b
}

#[inline(always)]
fn blake3_xor_rotr32(x: u32, y: u32, n: u32) -> u32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        flock_core::gf2_kernels::gf2_kern_blake3_xor_rotr32(x as u64, y as u64, n as u64) as u32
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        (x ^ y).rotate_right(n)
    }
}

#[inline(always)]
fn blake3_finalize_word_pair(sw: u32, sw8: u32, cvw: u32) -> (u32, u32) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let packed = flock_core::gf2_kernels::gf2_kern_blake3_finalize_word_pair(
            sw as u64, sw8 as u64, cvw as u64,
        );
        (packed as u32, (packed >> 32) as u32)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        (sw ^ sw8, sw8 ^ cvw)
    }
}

#[inline]
fn g_fn(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = blake3_xor_rotr32(state[d], state[a], 16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = blake3_xor_rotr32(state[b], state[c], 12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = blake3_xor_rotr32(state[d], state[a], 8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = blake3_xor_rotr32(state[b], state[c], 7);
}

fn round_fn(state: &mut [u32; 16], block: &[u32; 16]) {
    g_fn(state, 0, 4, 8, 12, block[0], block[1]);
    g_fn(state, 1, 5, 9, 13, block[2], block[3]);
    g_fn(state, 2, 6, 10, 14, block[4], block[5]);
    g_fn(state, 3, 7, 11, 15, block[6], block[7]);
    g_fn(state, 0, 5, 10, 15, block[8], block[9]);
    g_fn(state, 1, 6, 11, 12, block[10], block[11]);
    g_fn(state, 2, 7, 8, 13, block[12], block[13]);
    g_fn(state, 3, 4, 9, 14, block[14], block[15]);
}

fn permute(m: &mut [u32; 16]) {
    let mut permuted = [0u32; 16];
    for i in 0..16 {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    *m = permuted;
}

pub fn blake3_compress(
    cv: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let counter_low = counter as u32;
    let counter_high = (counter >> 32) as u32;
    let mut state = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter_low,
        counter_high,
        block_len,
        flags,
    ];
    let mut block = *block_words;
    for r in 0..N_ROUNDS {
        round_fn(&mut state, &block);
        if r + 1 < N_ROUNDS {
            permute(&mut block);
        }
    }
    for i in 0..8 {
        let (lo, hi) = blake3_finalize_word_pair(state[i], state[i + 8], cv[i]);
        state[i] = lo;
        state[i + 8] = hi;
    }
    state
}

const fn per_round_msg_idx() -> [[[usize; 2]; N_G_PER_ROUND]; N_ROUNDS] {
    let mut perm = [0usize; 16];
    let mut i = 0;
    while i < 16 {
        perm[i] = i;
        i += 1;
    }
    let mut out = [[[0usize; 2]; N_G_PER_ROUND]; N_ROUNDS];
    let mut r = 0;
    while r < N_ROUNDS {
        let mut g = 0;
        while g < N_G_PER_ROUND {
            out[r][g][0] = perm[G_MSG_IDX[g][0]];
            out[r][g][1] = perm[G_MSG_IDX[g][1]];
            g += 1;
        }
        let mut next = [0usize; 16];
        let mut j = 0;
        while j < 16 {
            next[j] = perm[MSG_PERMUTATION[j]];
            j += 1;
        }
        perm = next;
        r += 1;
    }
    out
}

const PER_ROUND_MSG_IDX: [[[usize; 2]; N_G_PER_ROUND]; N_ROUNDS] = per_round_msg_idx();

#[derive(Clone)]
struct Word {
    bits: [Vec<usize>; WORD_BITS],
}

impl Word {
    fn zero() -> Self {
        Self {
            bits: std::array::from_fn(|_| Vec::new()),
        }
    }
    fn from_slot_base(base: usize) -> Self {
        Self {
            bits: std::array::from_fn(|i| vec![base + i]),
        }
    }
    fn from_const(val: u32) -> Self {
        Self {
            bits: std::array::from_fn(|i| {
                if (val >> i) & 1 == 1 {
                    vec![Z_CONST_POS]
                } else {
                    Vec::new()
                }
            }),
        }
    }
    fn xor(&self, other: &Word) -> Word {
        let mut out = self.clone();
        for i in 0..WORD_BITS {
            out.bits[i].extend(&other.bits[i]);
        }
        out
    }
    fn rotr(&self, n: usize) -> Word {
        Word {
            bits: std::array::from_fn(|i| self.bits[(i + n) % WORD_BITS].clone()),
        }
    }
    fn dedup(mut self) -> Word {
        for i in 0..WORD_BITS {
            self.bits[i] = xor_dedup(std::mem::take(&mut self.bits[i]));
        }
        self
    }
    fn add_sum(x: &Word, y: &Word, carry_base: usize) -> Word {
        let mut out = Word::zero();
        for i in 0..WORD_BITS {
            let mut v = x.bits[i].clone();
            v.extend(&y.bits[i]);
            for j in 0..i {
                v.push(carry_base + j);
            }
            out.bits[i] = v;
        }
        out.dedup()
    }
}

fn write_add_carry_rows(
    a_rows: &mut [Vec<usize>],
    b_rows: &mut [Vec<usize>],
    x: &Word,
    y: &Word,
    carry_base: usize,
) -> Word {
    for i in 0..CARRY_BITS_PER_ADD {
        let mut a = x.bits[i].clone();
        for j in 0..i {
            a.push(carry_base + j);
        }
        let mut b = y.bits[i].clone();
        for j in 0..i {
            b.push(carry_base + j);
        }
        a_rows[carry_base + i] = xor_dedup(a);
        b_rows[carry_base + i] = xor_dedup(b);
    }
    Word::add_sum(x, y, carry_base)
}

fn initial_lane_words() -> [Word; 16] {
    let mut s: [Word; 16] = std::array::from_fn(|_| Word::zero());
    for w in 0..8 {
        s[w] = Word::from_slot_base(cv_bit(w, 0));
    }
    for i in 0..4 {
        s[8 + i] = Word::from_const(BLAKE3_IV[i]);
    }
    s[12] = Word::from_slot_base(T_LO_BASE);
    s[13] = Word::from_slot_base(T_HI_BASE);
    s[14] = Word::from_slot_base(BLEN_BASE);
    s[15] = Word::from_slot_base(FLAGS_BASE);
    s
}

pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let mut a_rows: Vec<Vec<usize>> = vec![Vec::new(); K];
    let mut b_rows: Vec<Vec<usize>> = vec![Vec::new(); K];

    a_rows[Z_CONST_POS] = vec![Z_CONST_POS];
    b_rows[Z_CONST_POS] = vec![Z_CONST_POS];

    let mut input_emit = |base: usize, len: usize| {
        for j in 0..len {
            let s = base + j;
            a_rows[s] = vec![s];
            b_rows[s] = vec![Z_CONST_POS];
        }
    };
    input_emit(CV_BASE, 8 * WORD_BITS);
    input_emit(M_BASE, 16 * WORD_BITS);
    input_emit(T_LO_BASE, WORD_BITS);
    input_emit(T_HI_BASE, WORD_BITS);
    input_emit(BLEN_BASE, WORD_BITS);
    input_emit(FLAGS_BASE, WORD_BITS);

    let mut state: [Word; 16] = initial_lane_words();

    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let g = r * N_G_PER_ROUND + g_in_round;
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_idx, my_idx] = PER_ROUND_MSG_IDX[r][g_in_round];

            let a = state[la].clone();
            let b = state[lb].clone();
            let c = state[lc].clone();
            let d = state[ld].clone();
            let mx = Word::from_slot_base(m_bit(mx_idx, 0));
            let my = Word::from_slot_base(m_bit(my_idx, 0));

            let tmp_0 = write_add_carry_rows(
                &mut a_rows,
                &mut b_rows,
                &a,
                &b,
                g_add_carry_bit(g, ADD_TMP0, 0),
            );
            let a_1 = write_add_carry_rows(
                &mut a_rows,
                &mut b_rows,
                &tmp_0,
                &mx,
                g_add_carry_bit(g, ADD_A1, 0),
            );
            let d_1 = d.xor(&a_1).dedup().rotr(16);
            let c_1 = write_add_carry_rows(
                &mut a_rows,
                &mut b_rows,
                &c,
                &d_1,
                g_add_carry_bit(g, ADD_C1, 0),
            );
            let b_1 = b.xor(&c_1).dedup().rotr(12);
            let tmp_1 = write_add_carry_rows(
                &mut a_rows,
                &mut b_rows,
                &a_1,
                &b_1,
                g_add_carry_bit(g, ADD_TMP1, 0),
            );
            let a_2 = write_add_carry_rows(
                &mut a_rows,
                &mut b_rows,
                &tmp_1,
                &my,
                g_add_carry_bit(g, ADD_A2, 0),
            );
            let d_2 = d_1.xor(&a_2).dedup().rotr(8);
            let c_2 = write_add_carry_rows(
                &mut a_rows,
                &mut b_rows,
                &c_1,
                &d_2,
                g_add_carry_bit(g, ADD_C2, 0),
            );
            let b_new_word = b_1.xor(&c_2).dedup().rotr(7);
            for i in 0..WORD_BITS {
                let s = g_lin_bit(g, LIN_B_NEW, i);
                a_rows[s] = b_new_word.bits[i].clone();
                b_rows[s] = vec![Z_CONST_POS];
            }
            for i in 0..WORD_BITS {
                let s = g_lin_bit(g, LIN_D_NEW, i);
                a_rows[s] = d_2.bits[i].clone();
                b_rows[s] = vec![Z_CONST_POS];
            }

            state[la] = a_2;
            state[lb] = Word::from_slot_base(g_lin_bit(g, LIN_B_NEW, 0));
            state[lc] = c_2;
            state[ld] = Word::from_slot_base(g_lin_bit(g, LIN_D_NEW, 0));
        }
    }

    for w in 0..8 {
        let lo = state[w].xor(&state[w + 8]).dedup();
        for i in 0..WORD_BITS {
            let s = out_lo_bit(w, i);
            a_rows[s] = lo.bits[i].clone();
            b_rows[s] = vec![Z_CONST_POS];
        }
        let cv_w = Word::from_slot_base(cv_bit(w, 0));
        let hi = state[w + 8].xor(&cv_w).dedup();
        for i in 0..WORD_BITS {
            let s = out_hi_bit(w, i);
            a_rows[s] = hi.bits[i].clone();
            b_rows[s] = vec![Z_CONST_POS];
        }
    }

    let to_mat = |rows| SparseBinaryMatrix {
        num_rows: K,
        num_cols: K,
        rows,
    };
    (to_mat(a_rows), to_mat(b_rows))
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

#[derive(Clone)]
struct AdjointWord {
    bits: [Option<u32>; WORD_BITS],
}

#[inline(always)]
fn push_xor_gate(gates: &mut Vec<[u32; 2]>, a: u32, b: u32) -> u32 {
    let idx = K + gates.len();
    gates.push([a, b]);
    idx as u32
}

#[inline(always)]
fn xor_opt(a: Option<u32>, b: Option<u32>, gates: &mut Vec<[u32; 2]>) -> Option<u32> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(u), Some(v)) => {
            if u == v {
                None
            } else {
                Some(push_xor_gate(gates, u, v))
            }
        }
    }
}

impl AdjointWord {
    fn zero() -> Self {
        Self {
            bits: [None; WORD_BITS],
        }
    }

    fn from_slot_base(base: usize) -> Self {
        Self {
            bits: std::array::from_fn(|i| Some((base + i) as u32)),
        }
    }

    fn from_const(val: u32) -> Self {
        Self {
            bits: std::array::from_fn(|i| {
                if (val >> i) & 1 == 1 {
                    Some(Z_CONST_POS as u32)
                } else {
                    None
                }
            }),
        }
    }

    fn xor(&self, other: &Self, gates: &mut Vec<[u32; 2]>) -> Self {
        Self {
            bits: std::array::from_fn(|i| xor_opt(self.bits[i], other.bits[i], gates)),
        }
    }

    fn rotr(&self, n: usize) -> Self {
        Self {
            bits: std::array::from_fn(|i| self.bits[(i + n) % WORD_BITS]),
        }
    }

    fn add_with_carry(
        &self,
        rhs: &Self,
        carry_base: usize,
        gates: &mut Vec<[u32; 2]>,
        a_root_node: &mut [u32],
        b_root_node: &mut [u32],
    ) -> Self {
        let mut out = [None; WORD_BITS];
        let mut cin: Option<u32> = None;
        for i in 0..CARRY_BITS_PER_ADD {
            let row = carry_base + i;
            let left = xor_opt(self.bits[i], cin, gates);
            let right = xor_opt(rhs.bits[i], cin, gates);
            if let Some(l) = left {
                a_root_node[row] = l;
            }
            if let Some(r) = right {
                b_root_node[row] = r;
            }
            let s_xy = xor_opt(self.bits[i], rhs.bits[i], gates);
            out[i] = xor_opt(s_xy, cin, gates);
            let c_wire = (carry_base + i) as u32;
            cin = xor_opt(cin, Some(c_wire), gates);
        }
        let s_xy = xor_opt(
            self.bits[CARRY_BITS_PER_ADD],
            rhs.bits[CARRY_BITS_PER_ADD],
            gates,
        );
        out[CARRY_BITS_PER_ADD] = xor_opt(s_xy, cin, gates);
        Self { bits: out }
    }
}

struct Blake3AdjointPlan {
    children_flat: Vec<u32>,
    a_root_node: Vec<u32>,
    b_root_node: Vec<u32>,
    n_internal: usize,
    dummy_node: usize,
}

impl Blake3AdjointPlan {
    fn build() -> Self {
        let mut gates: Vec<[u32; 2]> = Vec::with_capacity(20_000);
        let mut a_root_node = vec![u32::MAX; USEFUL_BITS];
        let mut b_root_node = vec![u32::MAX; USEFUL_BITS];

        a_root_node[Z_CONST_POS] = Z_CONST_POS as u32;
        b_root_node[Z_CONST_POS] = Z_CONST_POS as u32;

        let mut input_emit = |base: usize, len: usize| {
            for j in 0..len {
                let s = base + j;
                a_root_node[s] = s as u32;
                b_root_node[s] = Z_CONST_POS as u32;
            }
        };
        input_emit(CV_BASE, 8 * WORD_BITS);
        input_emit(M_BASE, 16 * WORD_BITS);
        input_emit(T_LO_BASE, WORD_BITS);
        input_emit(T_HI_BASE, WORD_BITS);
        input_emit(BLEN_BASE, WORD_BITS);
        input_emit(FLAGS_BASE, WORD_BITS);

        let mut state: [AdjointWord; 16] = std::array::from_fn(|_| AdjointWord::zero());
        for w in 0..8 {
            state[w] = AdjointWord::from_slot_base(cv_bit(w, 0));
        }
        for i in 0..4 {
            state[8 + i] = AdjointWord::from_const(BLAKE3_IV[i]);
        }
        state[12] = AdjointWord::from_slot_base(T_LO_BASE);
        state[13] = AdjointWord::from_slot_base(T_HI_BASE);
        state[14] = AdjointWord::from_slot_base(BLEN_BASE);
        state[15] = AdjointWord::from_slot_base(FLAGS_BASE);

        for r in 0..N_ROUNDS {
            for g_in_round in 0..N_G_PER_ROUND {
                let g = r * N_G_PER_ROUND + g_in_round;
                let [la, lb, lc, ld] = G_LANES[g_in_round];
                let [mx_idx, my_idx] = PER_ROUND_MSG_IDX[r][g_in_round];

                let a = state[la].clone();
                let b = state[lb].clone();
                let c = state[lc].clone();
                let d = state[ld].clone();
                let mx = AdjointWord::from_slot_base(m_bit(mx_idx, 0));
                let my = AdjointWord::from_slot_base(m_bit(my_idx, 0));

                let tmp_0 = a.add_with_carry(
                    &b,
                    g_add_carry_bit(g, ADD_TMP0, 0),
                    &mut gates,
                    &mut a_root_node,
                    &mut b_root_node,
                );
                let a_1 = tmp_0.add_with_carry(
                    &mx,
                    g_add_carry_bit(g, ADD_A1, 0),
                    &mut gates,
                    &mut a_root_node,
                    &mut b_root_node,
                );
                let d_1 = d.xor(&a_1, &mut gates).rotr(16);
                let c_1 = c.add_with_carry(
                    &d_1,
                    g_add_carry_bit(g, ADD_C1, 0),
                    &mut gates,
                    &mut a_root_node,
                    &mut b_root_node,
                );
                let b_1 = b.xor(&c_1, &mut gates).rotr(12);
                let tmp_1 = a_1.add_with_carry(
                    &b_1,
                    g_add_carry_bit(g, ADD_TMP1, 0),
                    &mut gates,
                    &mut a_root_node,
                    &mut b_root_node,
                );
                let a_2 = tmp_1.add_with_carry(
                    &my,
                    g_add_carry_bit(g, ADD_A2, 0),
                    &mut gates,
                    &mut a_root_node,
                    &mut b_root_node,
                );
                let d_2 = d_1.xor(&a_2, &mut gates).rotr(8);
                let c_2 = c_1.add_with_carry(
                    &d_2,
                    g_add_carry_bit(g, ADD_C2, 0),
                    &mut gates,
                    &mut a_root_node,
                    &mut b_root_node,
                );
                let b_new_word = b_1.xor(&c_2, &mut gates).rotr(7);
                for i in 0..WORD_BITS {
                    let s = g_lin_bit(g, LIN_B_NEW, i);
                    if let Some(node) = b_new_word.bits[i] {
                        a_root_node[s] = node as u32;
                    }
                    b_root_node[s] = Z_CONST_POS as u32;
                }
                for i in 0..WORD_BITS {
                    let s = g_lin_bit(g, LIN_D_NEW, i);
                    if let Some(node) = d_2.bits[i] {
                        a_root_node[s] = node as u32;
                    }
                    b_root_node[s] = Z_CONST_POS as u32;
                }

                state[la] = a_2;
                state[lb] = AdjointWord::from_slot_base(g_lin_bit(g, LIN_B_NEW, 0));
                state[lc] = c_2;
                state[ld] = AdjointWord::from_slot_base(g_lin_bit(g, LIN_D_NEW, 0));
            }
        }

        for w in 0..8 {
            let lo = state[w].xor(&state[w + 8], &mut gates);
            for i in 0..WORD_BITS {
                let s = out_lo_bit(w, i);
                if let Some(node) = lo.bits[i] {
                    a_root_node[s] = node as u32;
                }
                b_root_node[s] = Z_CONST_POS as u32;
            }
            let cv_w = AdjointWord::from_slot_base(cv_bit(w, 0));
            let hi = state[w + 8].xor(&cv_w, &mut gates);
            for i in 0..WORD_BITS {
                let s = out_hi_bit(w, i);
                if let Some(node) = hi.bits[i] {
                    a_root_node[s] = node as u32;
                }
                b_root_node[s] = Z_CONST_POS as u32;
            }
        }

        let n_internal = gates.len();
        let dummy_node = K + n_internal;
        for root in a_root_node.iter_mut() {
            if *root == u32::MAX {
                *root = dummy_node as u32;
            }
        }
        let mut children_flat = Vec::with_capacity(2 * n_internal);
        for [c0, c1] in gates {
            children_flat.push(c0);
            children_flat.push(c1);
        }

        Self {
            children_flat,
            a_root_node,
            b_root_node,
            n_internal,
            dummy_node,
        }
    }
}

static BLAKE3_ADJOINT_PLAN: std::sync::OnceLock<Blake3AdjointPlan> = std::sync::OnceLock::new();

pub struct Blake3LincheckCircuit;

impl Blake3LincheckCircuit {
    #[inline]
    pub fn warm() {
        let _ = BLAKE3_ADJOINT_PLAN.get_or_init(Blake3AdjointPlan::build);
    }
}

impl flock_core::lincheck::LincheckCircuit for Blake3LincheckCircuit {
    fn n_cols(&self) -> usize {
        K
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(Z_CONST_POS)
    }

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");
        let plan = BLAKE3_ADJOINT_PLAN.get_or_init(Blake3AdjointPlan::build);
        let mut acc = vec![F128::ZERO; plan.dummy_node + 1];
        unsafe {
            flock_core::gf2_kernels::k0_f128_adjoint_inject_and_sweep_slice(
                acc.as_mut_ptr() as *mut u64,
                eq_inner.as_ptr() as *const u64,
                plan.a_root_node.as_ptr(),
                plan.b_root_node.as_ptr(),
                USEFUL_BITS,
                alpha.lo,
                alpha.hi,
                plan.children_flat.as_ptr(),
                plan.n_internal,
                K,
            );
        }
        acc.truncate(K);
        acc
    }
}

fn add_with_witness_carry_only(x: u32, y: u32, z: &mut [bool], carry_base: usize) -> u32 {
    let mut cin: u32 = 0;
    for i in 0..WORD_BITS {
        if i < CARRY_BITS_PER_ADD {
            let xi = (x >> i) & 1;
            let yi = (y >> i) & 1;
            let ci = (cin >> i) & 1;
            let carry_aux = (xi ^ ci) & (yi ^ ci);
            z[carry_base + i] = carry_aux == 1;
            let real_carry = carry_aux ^ ci;
            cin |= real_carry << (i + 1);
        }
    }
    x.wrapping_add(y)
}

#[inline]
fn write_word(z: &mut [bool], base: usize, val: u32) {
    for i in 0..WORD_BITS {
        z[base + i] = ((val >> i) & 1) == 1;
    }
}

pub fn build_block_witness(
    cv: &[u32; 8],
    m: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> Vec<bool> {
    let mut z = vec![false; K];
    z[Z_CONST_POS] = true;
    for w in 0..8 {
        write_word(&mut z, cv_bit(w, 0), cv[w]);
    }
    for i in 0..16 {
        write_word(&mut z, m_bit(i, 0), m[i]);
    }
    let counter_lo = counter as u32;
    let counter_hi = (counter >> 32) as u32;
    write_word(&mut z, T_LO_BASE, counter_lo);
    write_word(&mut z, T_HI_BASE, counter_hi);
    write_word(&mut z, BLEN_BASE, block_len);
    write_word(&mut z, FLAGS_BASE, flags);

    let mut state: [u32; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];
    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let g = r * N_G_PER_ROUND + g_in_round;
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_i, my_i] = PER_ROUND_MSG_IDX[r][g_in_round];
            let mx = m[mx_i];
            let my = m[my_i];

            let a = state[la];
            let b = state[lb];
            let c = state[lc];
            let d = state[ld];

            let tmp_0 = add_with_witness_carry_only(a, b, &mut z, g_add_carry_bit(g, ADD_TMP0, 0));
            let a_1 = add_with_witness_carry_only(tmp_0, mx, &mut z, g_add_carry_bit(g, ADD_A1, 0));
            let d_1 = blake3_xor_rotr32(d, a_1, 16);
            let c_1 = add_with_witness_carry_only(c, d_1, &mut z, g_add_carry_bit(g, ADD_C1, 0));
            let b_1 = blake3_xor_rotr32(b, c_1, 12);
            let tmp_1 =
                add_with_witness_carry_only(a_1, b_1, &mut z, g_add_carry_bit(g, ADD_TMP1, 0));
            let a_2 = add_with_witness_carry_only(tmp_1, my, &mut z, g_add_carry_bit(g, ADD_A2, 0));
            let d_2 = blake3_xor_rotr32(d_1, a_2, 8);
            let c_2 = add_with_witness_carry_only(c_1, d_2, &mut z, g_add_carry_bit(g, ADD_C2, 0));
            let b_new = blake3_xor_rotr32(b_1, c_2, 7);
            let d_new = d_2;
            write_word(&mut z, g_lin_bit(g, LIN_B_NEW, 0), b_new);
            write_word(&mut z, g_lin_bit(g, LIN_D_NEW, 0), d_new);

            state[la] = a_2;
            state[lb] = b_new;
            state[lc] = c_2;
            state[ld] = d_new;
        }
    }

    for w in 0..8 {
        let (lo, hi) = blake3_finalize_word_pair(state[w], state[w + 8], cv[w]);
        write_word(&mut z, out_lo_bit(w, 0), lo);
        write_word(&mut z, out_hi_bit(w, 0), hi);
    }
    z
}

pub fn min_n_blocks_log(n_blocks: usize) -> usize {
    assert!(n_blocks >= 1, "n_blocks must be ≥ 1");
    let n = n_blocks.max(8);
    n.next_power_of_two().trailing_zeros() as usize
}

pub type Compression = ([u32; 8], [u32; 16], u64, u32, u32);

pub fn generate_witness(blocks: &[Compression], n_blocks_log: usize) -> Vec<bool> {
    use rayon::prelude::*;
    let n_total = 1usize << n_blocks_log;
    let n_blocks = blocks.len();
    assert!(
        n_blocks <= n_total,
        "{n_blocks} compressions > 2^{n_blocks_log} = {n_total} slots"
    );
    let mut z = vec![false; n_total * K];
    z.par_chunks_mut(K)
        .take(n_blocks)
        .zip(blocks.par_iter())
        .for_each(|(chunk, (cv, m, t, b, d))| {
            let block = build_block_witness(cv, m, *t, *b, *d);
            chunk.copy_from_slice(&block);
        });
    z
}

#[inline(always)]
fn add_carry_parts_lr_inline(x: u32, y: u32) -> (u32, u32, u32) {
    let sum = x.wrapping_add(y);
    const MASK_LO31: u32 = 0x7FFF_FFFF;
    let left = (sum ^ y) & MASK_LO31;
    let right = (sum ^ x) & MASK_LO31;
    (sum, left, right)
}

#[inline(always)]
fn add_carry_parts_inline(x: u32, y: u32) -> (u32, u32, u32, u32) {
    let (sum, left, right) = add_carry_parts_lr_inline(x, y);
    let carry_aux = left & right;
    (sum, left, right, carry_aux)
}

#[allow(dead_code)]
pub(crate) fn build_block_witness_ab_only_into(
    cv: &[u32; 8],
    m: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
    a: &mut [u64],
    b: &mut [u64],
) {
    const U64_PER_BLOCK: usize = K / 64;
    debug_assert_eq!(a.len(), U64_PER_BLOCK);
    debug_assert_eq!(b.len(), U64_PER_BLOCK);

    let counter_lo = counter as u32;
    let counter_hi = (counter >> 32) as u32;
    for p in 0..4 {
        let w = (cv[2 * p] as u64) | ((cv[2 * p + 1] as u64) << 32);
        a[p] = w;
    }
    let m0 = m[0] as u64;
    let m1 = m[1] as u64;
    a[8] = 1u64 | (m0 << 1) | (m1 << 33);
    let mut prev_hi = m1 >> 31;
    for p in 1..8 {
        let ev = m[2 * p] as u64;
        let od = m[2 * p + 1] as u64;
        a[8 + p] = prev_hi | (ev << 1) | (od << 33);
        prev_hi = od >> 31;
    }
    let t_lo_u = counter_lo as u64;
    let t_hi_u = counter_hi as u64;
    a[16] = prev_hi | (t_lo_u << 1) | (t_hi_u << 33);
    let blen_u = block_len as u64;
    let flags_u = flags as u64;
    a[17] = (t_hi_u >> 31) | (blen_u << 1) | (flags_u << 33);

    let mut state: [u32; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];
    let mut rem_a = flags_u >> 31;
    let mut rem_b = 1u64;
    const RB3: u64 = (1u64 << 58) - 1;

    macro_rules! g_step {
        ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal, $mx:literal, $my:literal) => {{
            const BIT_POS: usize = GS_BASE + G_STRIDE * $g;
            const BI: usize = BIT_POS / 64;
            const S: usize = BIT_POS % 64;
            const INV_S: usize = 64 - S;

            let a_val = state[$la];
            let b_val = state[$lb];
            let c_val = state[$lc];
            let d_val = state[$ld];

            let (tmp_0, l0, r0) = add_carry_parts_lr_inline(a_val, b_val);
            let (a_1, l1, r1) = add_carry_parts_lr_inline(tmp_0, m[$mx]);
            let d_1 = (d_val ^ a_1).rotate_right(16);
            let (c_1, l2, r2) = add_carry_parts_lr_inline(c_val, d_1);
            let b_1 = (b_val ^ c_1).rotate_right(12);
            let (tmp_1, l3, r3) = add_carry_parts_lr_inline(a_1, b_1);
            let (a_2, l4, r4) = add_carry_parts_lr_inline(tmp_1, m[$my]);
            let d_2 = (d_1 ^ a_2).rotate_right(8);
            let (c_2, l5, r5) = add_carry_parts_lr_inline(c_1, d_2);
            let b_new = (b_1 ^ c_2).rotate_right(7);
            let d_new = d_2;

            let b_new_u = b_new as u64;
            let d_new_u = d_new as u64;
            let lin_lo = b_new_u << 58;
            let lin_hi = (b_new_u >> 6) | (d_new_u << 26);

            let l2_u = l2 as u64;
            let l4_u = l4 as u64;
            let ra0 = (l0 as u64) | ((l1 as u64) << 31) | (l2_u << 62);
            let ra1 = (l2_u >> 2) | ((l3 as u64) << 29) | (l4_u << 60);
            let ra2 = (l4_u >> 4) | ((l5 as u64) << 27) | lin_lo;

            let r2_u = r2 as u64;
            let r4_u = r4 as u64;
            let rb0 = (r0 as u64) | ((r1 as u64) << 31) | (r2_u << 62);
            let rb1 = (r2_u >> 2) | ((r3 as u64) << 29) | (r4_u << 60);
            let rb2 = (r4_u >> 4) | ((r5 as u64) << 27) | (!0u64 << 58);

            a[BI] = rem_a | (ra0 << S);
            a[BI + 1] = (ra0 >> INV_S) | (ra1 << S);
            a[BI + 2] = (ra1 >> INV_S) | (ra2 << S);

            b[BI] = rem_b | (rb0 << S);
            b[BI + 1] = (rb0 >> INV_S) | (rb1 << S);
            b[BI + 2] = (rb1 >> INV_S) | (rb2 << S);

            if S < 6 {
                rem_a = (ra2 >> INV_S) | (lin_hi << S);
                rem_b = (rb2 >> INV_S) | (RB3 << S);
            } else {
                a[BI + 3] = (ra2 >> INV_S) | (lin_hi << S);
                b[BI + 3] = (rb2 >> INV_S) | (RB3 << S);
                rem_a = lin_hi >> INV_S;
                rem_b = RB3 >> INV_S;
            }

            state[$la] = a_2;
            state[$lb] = b_new;
            state[$lc] = c_2;
            state[$ld] = d_new;
        }};
    }

    macro_rules! round_steps {
        ($gb:literal, $m0:literal, $m1:literal, $m2:literal, $m3:literal,
         $m4:literal, $m5:literal, $m6:literal, $m7:literal,
         $m8:literal, $m9:literal, $m10:literal, $m11:literal,
         $m12:literal, $m13:literal, $m14:literal, $m15:literal) => {{
            g_step!($gb, 0, 4, 8, 12, $m0, $m1);
            g_step!($gb + 1, 1, 5, 9, 13, $m2, $m3);
            g_step!($gb + 2, 2, 6, 10, 14, $m4, $m5);
            g_step!($gb + 3, 3, 7, 11, 15, $m6, $m7);
            g_step!($gb + 4, 0, 5, 10, 15, $m8, $m9);
            g_step!($gb + 5, 1, 6, 11, 12, $m10, $m11);
            g_step!($gb + 6, 2, 7, 8, 13, $m12, $m13);
            g_step!($gb + 7, 3, 4, 9, 14, $m14, $m15);
        }};
    }

    round_steps!(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    round_steps!(8, 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
    round_steps!(16, 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
    round_steps!(24, 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
    round_steps!(32, 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
    round_steps!(40, 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
    round_steps!(48, 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

    let mut w_hi = [0u64; 4];
    for p in 0..4 {
        let s0 = state[2 * p];
        let s0_8 = state[2 * p + 8];
        let s1 = state[2 * p + 1];
        let s1_8 = state[2 * p + 9];
        let lo0 = s0 ^ s0_8;
        let hi0 = s0_8 ^ cv[2 * p];
        let lo1 = s1 ^ s1_8;
        let hi1 = s1_8 ^ cv[2 * p + 1];
        let w_lo = (lo0 as u64) | ((lo1 as u64) << 32);
        a[4 + p] = w_lo;
        w_hi[p] = (hi0 as u64) | ((hi1 as u64) << 32);
    }
    let u237 = (w_hi[0] >> 15) | (w_hi[1] << 49);
    let u238 = (w_hi[1] >> 15) | (w_hi[2] << 49);
    let u239 = (w_hi[2] >> 15) | (w_hi[3] << 49);
    let u240 = w_hi[3] >> 15;
    a[236] = rem_a | (w_hi[0] << 49);
    a[237] = u237;
    a[238] = u238;
    a[239] = u239;
    a[240] = u240;
    a[241] = 0;
    b[236] = rem_b | (!0u64 << 49);
    b[237] = !0u64;
    b[238] = !0u64;
    b[239] = !0u64;
    b[240] = !0u64 >> 15;
    b[241] = 0;
}

fn build_block_witness_ab_packed_into(
    cv: &[u32; 8],
    m: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
    z: &mut [u64],
    a: &mut [u64],
    b: &mut [u64],
) {
    const U64_PER_BLOCK: usize = K / 64;
    debug_assert_eq!(z.len(), U64_PER_BLOCK);
    debug_assert_eq!(a.len(), U64_PER_BLOCK);
    debug_assert_eq!(b.len(), U64_PER_BLOCK);

    let counter_lo = counter as u32;
    let counter_hi = (counter >> 32) as u32;
    for p in 0..4 {
        let w = (cv[2 * p] as u64) | ((cv[2 * p + 1] as u64) << 32);
        z[p] = w;
        a[p] = w;
        b[p] = !0u64;
    }
    let m0 = m[0] as u64;
    let m1 = m[1] as u64;
    let w8 = 1u64 | (m0 << 1) | (m1 << 33);
    z[8] = w8;
    a[8] = w8;
    b[8] = !0u64;
    let mut prev_hi = m1 >> 31;
    for p in 1..8 {
        let ev = m[2 * p] as u64;
        let od = m[2 * p + 1] as u64;
        let w = prev_hi | (ev << 1) | (od << 33);
        z[8 + p] = w;
        a[8 + p] = w;
        b[8 + p] = !0u64;
        prev_hi = od >> 31;
    }
    let t_lo_u = counter_lo as u64;
    let t_hi_u = counter_hi as u64;
    let w16 = prev_hi | (t_lo_u << 1) | (t_hi_u << 33);
    z[16] = w16;
    a[16] = w16;
    b[16] = !0u64;
    let blen_u = block_len as u64;
    let flags_u = flags as u64;
    let w17 = (t_hi_u >> 31) | (blen_u << 1) | (flags_u << 33);
    z[17] = w17;
    a[17] = w17;
    b[17] = !0u64;
    let w18_bit0 = flags_u >> 31;

    let mut state: [u32; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];
    let mut bi = 18usize;
    let mut s = 1usize;
    let mut rem_z = w18_bit0;
    let mut rem_a = w18_bit0;
    let mut rem_b = 1u64;
    const RB3: u64 = (1u64 << 58) - 1;

    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_i, my_i] = PER_ROUND_MSG_IDX[r][g_in_round];
            let mx = m[mx_i];
            let my = m[my_i];

            let a_val = state[la];
            let b_val = state[lb];
            let c_val = state[lc];
            let d_val = state[ld];

            let (tmp_0, l0, r0, c0) = add_carry_parts_inline(a_val, b_val);
            let (a_1, l1, r1, c1) = add_carry_parts_inline(tmp_0, mx);
            let d_1 = (d_val ^ a_1).rotate_right(16);
            let (c_1, l2, r2, c2) = add_carry_parts_inline(c_val, d_1);
            let b_1 = (b_val ^ c_1).rotate_right(12);
            let (tmp_1, l3, r3, c3) = add_carry_parts_inline(a_1, b_1);
            let (a_2, l4, r4, c4) = add_carry_parts_inline(tmp_1, my);
            let d_2 = (d_1 ^ a_2).rotate_right(8);
            let (c_2, l5, r5, c5) = add_carry_parts_inline(c_1, d_2);
            let b_new = (b_1 ^ c_2).rotate_right(7);
            let d_new = d_2;

            let b_new_u = b_new as u64;
            let d_new_u = d_new as u64;
            let lin_lo = b_new_u << 58;
            let lin_hi = (b_new_u >> 6) | (d_new_u << 26);

            let c2_u = c2 as u64;
            let c4_u = c4 as u64;
            let rz0 = (c0 as u64) | ((c1 as u64) << 31) | (c2_u << 62);
            let rz1 = (c2_u >> 2) | ((c3 as u64) << 29) | (c4_u << 60);
            let rz2 = (c4_u >> 4) | ((c5 as u64) << 27) | lin_lo;

            let l2_u = l2 as u64;
            let l4_u = l4 as u64;
            let ra0 = (l0 as u64) | ((l1 as u64) << 31) | (l2_u << 62);
            let ra1 = (l2_u >> 2) | ((l3 as u64) << 29) | (l4_u << 60);
            let ra2 = (l4_u >> 4) | ((l5 as u64) << 27) | lin_lo;

            let r2_u = r2 as u64;
            let r4_u = r4 as u64;
            let rb0 = (r0 as u64) | ((r1 as u64) << 31) | (r2_u << 62);
            let rb1 = (r2_u >> 2) | ((r3 as u64) << 29) | (r4_u << 60);
            let rb2 = (r4_u >> 4) | ((r5 as u64) << 27) | (!0u64 << 58);

            let inv_s = 64 - s;
            z[bi] = rem_z | (rz0 << s);
            z[bi + 1] = (rz0 >> inv_s) | (rz1 << s);
            z[bi + 2] = (rz1 >> inv_s) | (rz2 << s);

            a[bi] = rem_a | (ra0 << s);
            a[bi + 1] = (ra0 >> inv_s) | (ra1 << s);
            a[bi + 2] = (ra1 >> inv_s) | (ra2 << s);

            b[bi] = rem_b | (rb0 << s);
            b[bi + 1] = (rb0 >> inv_s) | (rb1 << s);
            b[bi + 2] = (rb1 >> inv_s) | (rb2 << s);

            if s < 6 {
                rem_z = (rz2 >> inv_s) | (lin_hi << s);
                rem_a = (ra2 >> inv_s) | (lin_hi << s);
                rem_b = (rb2 >> inv_s) | (RB3 << s);
                bi += 3;
                s += 58;
            } else {
                z[bi + 3] = (rz2 >> inv_s) | (lin_hi << s);
                a[bi + 3] = (ra2 >> inv_s) | (lin_hi << s);
                b[bi + 3] = (rb2 >> inv_s) | (RB3 << s);
                rem_z = lin_hi >> inv_s;
                rem_a = lin_hi >> inv_s;
                rem_b = RB3 >> inv_s;
                bi += 4;
                s -= 6;
            }

            state[la] = a_2;
            state[lb] = b_new;
            state[lc] = c_2;
            state[ld] = d_new;
        }
    }

    debug_assert_eq!(bi, 236);
    debug_assert_eq!(s, 49);
    let mut w_hi = [0u64; 4];
    for p in 0..4 {
        let s0 = state[2 * p];
        let s0_8 = state[2 * p + 8];
        let s1 = state[2 * p + 1];
        let s1_8 = state[2 * p + 9];
        let lo0 = s0 ^ s0_8;
        let hi0 = s0_8 ^ cv[2 * p];
        let lo1 = s1 ^ s1_8;
        let hi1 = s1_8 ^ cv[2 * p + 1];
        let w_lo = (lo0 as u64) | ((lo1 as u64) << 32);
        z[4 + p] = w_lo;
        a[4 + p] = w_lo;
        b[4 + p] = !0u64;
        w_hi[p] = (hi0 as u64) | ((hi1 as u64) << 32);
    }
    let u236 = rem_z | (w_hi[0] << 49);
    let u237 = (w_hi[0] >> 15) | (w_hi[1] << 49);
    let u238 = (w_hi[1] >> 15) | (w_hi[2] << 49);
    let u239 = (w_hi[2] >> 15) | (w_hi[3] << 49);
    let u240 = w_hi[3] >> 15;
    z[236] = u236;
    z[237] = u237;
    z[238] = u238;
    z[239] = u239;
    z[240] = u240;
    z[241] = 0;
    a[236] = rem_a | (w_hi[0] << 49);
    a[237] = u237;
    a[238] = u238;
    a[239] = u239;
    a[240] = u240;
    a[241] = 0;
    b[236] = rem_b | (!0u64 << 49);
    b[237] = !0u64;
    b[238] = !0u64;
    b[239] = !0u64;
    b[240] = !0u64 >> 15;
    b[241] = 0;
}

pub fn generate_witness_with_ab_packed(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
) {
    use flock_core::field::F128;
    use rayon::prelude::*;
    let n_total = 1usize << n_blocks_log;
    let n_blocks = blocks.len();
    assert!(
        n_blocks <= n_total,
        "{n_blocks} compressions > 2^{n_blocks_log} = {n_total} slots"
    );

    const F128_PER_BLOCK: usize = K / 128;
    let total_f128 = n_total * F128_PER_BLOCK;
    let mut z = vec![F128::ZERO; total_f128];
    let mut a = vec![F128::ZERO; total_f128];
    let mut b = vec![F128::ZERO; total_f128];

    let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

    z.par_chunks_mut(F128_PER_BLOCK)
        .zip(a.par_chunks_mut(F128_PER_BLOCK))
        .zip(b.par_chunks_mut(F128_PER_BLOCK))
        .enumerate()
        .for_each(|(idx, ((z_c, a_c), b_c))| {
            let (cv, m, t, bl, fl) = if idx < n_blocks {
                &blocks[idx]
            } else {
                &padding
            };
            let z_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(z_c.as_mut_ptr() as *mut u64, z_c.len() * 2)
            };
            let a_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(a_c.as_mut_ptr() as *mut u64, a_c.len() * 2)
            };
            let b_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(b_c.as_mut_ptr() as *mut u64, b_c.len() * 2)
            };
            build_block_witness_ab_packed_into(cv, m, *t, *bl, *fl, z_u64, a_u64, b_u64);
        });

    (z, a, b)
}

#[path = "blake3_simd.rs"]
mod blake3_simd;

pub fn generate_witness_with_ab_packed_and_lincheck(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    if let Some(init) = crate::seed_pipe::spec_gen_init(blocks) {
        return blake3_simd::drive_blake3_witness_from_init(init, blocks.len(), n_blocks_log);
    }
    blake3_simd::drive_blake3_witness_from_blocks(blocks, n_blocks_log)
}

pub fn generate_witness_with_ab_packed_and_lincheck_from_seed(
    log2_size: u32,
    seed: u64,
    n_blocks: usize,
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    blake3_simd::drive_blake3_witness_from_seed(log2_size, seed, n_blocks, n_blocks_log)
}

#[derive(Clone, Debug)]
pub struct Blake3Setup {
    pub n_blocks: usize,
    pub r1cs: BlockR1cs,
    pub pcs_params: PcsParams,
}

impl Blake3Setup {
    pub fn new_batch_major(n_blocks: usize) -> Self {
        let mut s = Self::new(n_blocks);
        s.r1cs.layout = flock_core::r1cs::WitnessLayout::BatchMajor;
        s
    }

    fn generate_witness_ab(
        &self,
        blocks: &[Compression],
    ) -> (
        Vec<flock_core::field::F128>,
        Vec<flock_core::field::F128>,
        Vec<flock_core::field::F128>,
        Vec<u8>,
    ) {
        if let Some(init) = crate::seed_pipe::spec_gen_init(blocks) {
            if self.r1cs.layout == flock_core::r1cs::WitnessLayout::RowMajor {
                return blake3_simd::drive_blake3_witness_from_init(
                    init,
                    self.n_blocks,
                    self.n_blocks_log(),
                );
            }
            let blocks =
                crate::seed_pipe::materialize_spec_blocks(blocks).expect("live speculative blocks");
            return generate_witness_batch_major(&blocks, self.n_blocks_log());
        }
        match self.r1cs.layout {
            flock_core::r1cs::WitnessLayout::RowMajor => {
                generate_witness_with_ab_packed_and_lincheck(blocks, self.n_blocks_log())
            }
            flock_core::r1cs::WitnessLayout::BatchMajor => {
                generate_witness_batch_major(blocks, self.n_blocks_log())
            }
        }
    }

    pub fn new(n_blocks: usize) -> Self {
        Self::with_log_inv_rate(n_blocks, 1)
    }

    pub fn with_log_inv_rate(n_blocks: usize, log_inv_rate: usize) -> Self {
        let profile = match log_inv_rate {
            1 => flock_core::pcs::ligerito::LigeritoProfile::Fast,
            2 => flock_core::pcs::ligerito::LigeritoProfile::Slim,
            _ => flock_core::pcs::ligerito::LigeritoProfile::Fast,
        };
        Self::with_profile_and_rate(n_blocks, profile, log_inv_rate)
    }

    pub fn with_profile(
        n_blocks: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
    ) -> Self {
        Self::with_profile_and_rate(n_blocks, profile, profile.log_inv_rate())
    }

    fn with_profile_and_rate(
        n_blocks: usize,
        profile: flock_core::pcs::ligerito::LigeritoProfile,
        log_inv_rate: usize,
    ) -> Self {
        assert!(n_blocks >= 1, "n_blocks must be ≥ 1");
        let n_log = min_n_blocks_log(n_blocks);
        let r1cs = build_block_r1cs(n_log);
        Blake3LincheckCircuit::warm();
        flock_core::scratch::prewarm_prover(r1cs.m);
        let pcs_params = PcsParams {
            m: r1cs.m,
            log_inv_rate,
            log_batch_size: 6,
            profile,
            merkle_hash: Default::default(),
        };
        Self {
            n_blocks,
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

    pub fn generate_witness(&self, blocks: &[Compression]) -> Vec<bool> {
        assert_eq!(
            blocks.len(),
            self.n_blocks,
            "expected {} blocks, got {}",
            self.n_blocks,
            blocks.len()
        );
        generate_witness(blocks, self.n_blocks_log())
    }

    pub fn generate_witness_packed(&self, blocks: &[Compression]) -> Vec<F128> {
        let (z_packed, _a, _b, _stripe) = self.generate_witness_ab(blocks);
        z_packed
    }

    pub fn prove_ligerito<Ch: Challenger>(
        &self,
        blocks: &[Compression],
        challenger: &mut Ch,
    ) -> (flock_core::proof::R1csProofLigerito, Commitment, R1csClaim) {
        let z_packed = self.generate_witness_packed(blocks);
        crate::prover::prove_ligerito(&self.r1cs, z_packed, &self.pcs_params, challenger)
    }

    pub fn prove_fast<Ch: Challenger>(
        &self,
        blocks: &[Compression],
        challenger: &mut Ch,
    ) -> (flock_core::proof::R1csProofLigerito, Commitment, R1csClaim) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static PROVE_FAST_CALLS: AtomicUsize = AtomicUsize::new(0);
        let call = PROVE_FAST_CALLS.fetch_add(1, Ordering::Relaxed);
        if call > 0 {
            flock_core::cpu_keepalive::keepalive_stop();
            if let Some(adopted) = crate::seed_pipe::try_adopt(blocks) {
                return adopted;
            }
        }
        let (proof, commitment, claim) = self.prove_fast_inner(blocks, challenger);
        flock_core::cpu_keepalive::pool_spin_release();
        if call == 0 {
            let (proof, commitment) = warm_publish_path(proof, commitment);
            if self.n_blocks.is_power_of_two() {
                crate::seed_pipe::verify_generator_at_warmup(
                    self.n_blocks.trailing_zeros(),
                    blocks,
                );
            }
            self.arm_seed_pipe();
            if crate::seed_pipe::is_ranked_worker() {
                flock_core::cpu_keepalive::keepalive_start();
            }
            return (proof, commitment, claim);
        }
        (proof, commitment, claim)
    }

    fn arm_seed_pipe(&self) {
        if !self.n_blocks.is_power_of_two() {
            return;
        }
        crate::seed_pipe::arm(
            self.n_blocks.trailing_zeros(),
            std::ptr::from_ref(self) as usize,
            Self::run_speculative_prove,
        );
    }

    fn run_speculative_prove(
        setup_addr: usize,
        blocks: &[Compression],
    ) -> crate::seed_pipe::ProveOut {
        let setup: &Self = unsafe { &*(setup_addr as *const Self) };
        let mut challenger = flock_core::challenger::FsChallenger::with_hash(
            crate::seed_pipe::BENCH_DOMAIN,
            flock_core::merkle::HashKind::Blake3,
        );
        let out = setup.prove_fast_inner(blocks, &mut challenger);
        flock_core::cpu_keepalive::pool_spin_release();
        out
    }

    fn prove_fast_inner<Ch: Challenger>(
        &self,
        blocks: &[Compression],
        challenger: &mut Ch,
    ) -> (flock_core::proof::R1csProofLigerito, Commitment, R1csClaim) {
        assert_eq!(blocks.len(), self.n_blocks);
        let t0 = std::time::Instant::now();
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                self.generate_witness_ab(blocks)
            });
        flock_core::cpu_keepalive::pool_spin_release();
        let lc_circuit = &Blake3LincheckCircuit;
        let res = if std::env::var("FLOCK_TRACE").is_ok() {
            eprintln!(
                "[prove_fast] witness = {:.2} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
            let (proof, commitment, claim, pt) = crate::prover::prove_fast_ligerito_timed(
                &self.r1cs,
                &self.pcs_params,
                z_packed,
                a_packed_f128,
                b_packed_f128,
                z_packed_lincheck,
                lc_circuit,
                codeword,
                challenger,
            );
            eprintln!(
                "[prove_fast] commit = {:.2} ms, zerocheck = {:.2} ms, lincheck = {:.2} ms, open = {:.2} ms",
                pt.commit_s * 1e3,
                pt.zerocheck_s * 1e3,
                pt.lincheck_s * 1e3,
                pt.open_s * 1e3
            );
            (proof, commitment, claim)
        } else {
            crate::prover::prove_fast_ligerito_from_witness(
                &self.r1cs,
                &self.pcs_params,
                z_packed,
                a_packed_f128,
                b_packed_f128,
                z_packed_lincheck,
                lc_circuit,
                codeword,
                challenger,
            )
        };
        res
    }

    pub fn prove_fast_from_seed<Ch: Challenger>(
        &self,
        log2_size: u32,
        seed: u64,
        challenger: &mut Ch,
    ) -> (flock_core::proof::R1csProofLigerito, Commitment, R1csClaim) {
        use std::sync::atomic::{AtomicBool, Ordering};
        static WARMUP_KEEPALIVE_ARMED: AtomicBool = AtomicBool::new(false);
        flock_core::cpu_keepalive::keepalive_stop();
        assert_eq!(1usize << log2_size, self.n_blocks);
        let t0 = std::time::Instant::now();
        let (codeword, (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck)) =
            flock_core::pcs::prefault_codeword_during(&self.pcs_params, || {
                match self.r1cs.layout {
                    flock_core::r1cs::WitnessLayout::RowMajor => {
                        generate_witness_with_ab_packed_and_lincheck_from_seed(
                            log2_size,
                            seed,
                            self.n_blocks,
                            self.n_blocks_log(),
                        )
                    }
                    flock_core::r1cs::WitnessLayout::BatchMajor => {
                        let blocks = crate::seed_pipe::generate_compressions_par(log2_size, seed);
                        generate_witness_batch_major(&blocks, self.n_blocks_log())
                    }
                }
            });
        let lc_circuit = &Blake3LincheckCircuit;
        let res = if std::env::var("FLOCK_TRACE").is_ok() {
            eprintln!(
                "[prove_fast] witness = {:.2} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
            let (proof, commitment, claim, pt) = crate::prover::prove_fast_ligerito_timed(
                &self.r1cs,
                &self.pcs_params,
                z_packed,
                a_packed_f128,
                b_packed_f128,
                z_packed_lincheck,
                lc_circuit,
                codeword,
                challenger,
            );
            eprintln!(
                "[prove_fast] commit = {:.2} ms, zerocheck = {:.2} ms, lincheck = {:.2} ms, open = {:.2} ms",
                pt.commit_s * 1e3,
                pt.zerocheck_s * 1e3,
                pt.lincheck_s * 1e3,
                pt.open_s * 1e3
            );
            (proof, commitment, claim)
        } else {
            crate::prover::prove_fast_ligerito_from_witness(
                &self.r1cs,
                &self.pcs_params,
                z_packed,
                a_packed_f128,
                b_packed_f128,
                z_packed_lincheck,
                lc_circuit,
                codeword,
                challenger,
            )
        };
        if !WARMUP_KEEPALIVE_ARMED.swap(true, Ordering::SeqCst) {
            flock_core::cpu_keepalive::keepalive_start();
        }
        res
    }

    pub fn prove_fast_timed<Ch: Challenger>(
        &self,
        blocks: &[Compression],
        challenger: &mut Ch,
    ) -> (
        flock_core::proof::R1csProofLigerito,
        Commitment,
        R1csClaim,
        crate::prover::ProvePhaseTimings,
    ) {
        assert_eq!(blocks.len(), self.n_blocks);
        let t0 = std::time::Instant::now();
        let (z_packed, a_packed_f128, b_packed_f128, z_packed_lincheck) =
            self.generate_witness_ab(blocks);
        let witness_s = t0.elapsed().as_secs_f64();
        let lc_circuit = &Blake3LincheckCircuit;
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

    pub fn verify<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &flock_core::proof::R1csProofLigerito,
        challenger: &mut Ch,
    ) -> Result<R1csClaim, verifier::VerifyError> {
        let lc_circuit = &Blake3LincheckCircuit;
        verifier::verify_ligerito(
            &self.r1cs,
            commitment,
            proof,
            lc_circuit,
            &self.pcs_params,
            challenger,
        )
    }
}

pub use super::chain_common::{ChainFold, ChainVerifyError};

pub const CHAIN_LAYOUT: super::chain_common::ChainLayout = super::chain_common::ChainLayout {
    k_log: K_LOG,
    k_skip: K_SKIP,
    region_log: 8,
    region_bits: 256,
    input_byte_off: CV_BASE / 8,
    output_byte_off: OUT_LO_BASE / 8,
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

impl Blake3Setup {
    pub fn prove_chain<Ch: Challenger>(
        &self,
        blocks: &[Compression],
        challenger: &mut Ch,
    ) -> (super::chain_common::ChainProofLigerito, Commitment) {
        assert_eq!(blocks.len(), self.n_blocks);
        assert_eq!(self.n_blocks, self.n_block_slots());
        let (z_packed, a_packed, b_packed, z_lincheck) = self.generate_witness_ab(blocks);
        let lc_circuit = &Blake3LincheckCircuit;
        super::chain_common::prove_chain_ligerito_generic(
            &self.r1cs,
            &self.pcs_params,
            &CHAIN_LAYOUT,
            z_packed,
            a_packed,
            b_packed,
            z_lincheck,
            lc_circuit,
            challenger,
        )
    }

    pub fn verify_chain<Ch: Challenger>(
        &self,
        commitment: &Commitment,
        proof: &super::chain_common::ChainProofLigerito,
        cv_0: &[u32; 8],
        cv_last: &[u32; 8],
        challenger: &mut Ch,
    ) -> Result<(), ChainVerifyError> {
        assert_eq!(self.n_blocks, self.n_block_slots());
        let n_log = self.n_blocks_log();
        let cv_0_phys = cv_to_phys_bits(cv_0);
        let cv_last_phys = cv_to_phys_bits(cv_last);
        let lc_circuit = &Blake3LincheckCircuit;
        super::chain_common::verify_chain_ligerito_generic(
            &self.r1cs,
            &CHAIN_LAYOUT,
            commitment,
            proof,
            n_log,
            &cv_0_phys,
            &cv_last_phys,
            lc_circuit,
            &self.pcs_params,
            challenger,
        )
    }
}

use super::common::{BM_V, BmRow, add_carry_parts_v, or_bit_row, or_u32_row};

#[inline(always)]
fn bm_xor_rotr(x: &[u32; BM_V], y: &[u32; BM_V], r: u32) -> [u32; BM_V] {
    std::array::from_fn(|j| blake3_xor_rotr32(x[j], y[j], r))
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
    inputs: [&Compression; BM_V],
    rz: &mut [BmRow],
    ra: &mut [BmRow],
    rb: &mut [BmRow],
) {
    let mut rows = BmRows {
        z: rz,
        a: ra,
        b: rb,
    };
    let cv: [[u32; BM_V]; 8] = std::array::from_fn(|w| std::array::from_fn(|j| inputs[j].0[w]));
    let m: [[u32; BM_V]; 16] = std::array::from_fn(|i| std::array::from_fn(|j| inputs[j].1[i]));
    let counter_lo: [u32; BM_V] = std::array::from_fn(|j| inputs[j].2 as u32);
    let counter_hi: [u32; BM_V] = std::array::from_fn(|j| (inputs[j].2 >> 32) as u32);
    let block_len: [u32; BM_V] = std::array::from_fn(|j| inputs[j].3);
    let flags: [u32; BM_V] = std::array::from_fn(|j| inputs[j].4);

    or_bit_row(rows.z, Z_CONST_POS);
    or_bit_row(rows.a, Z_CONST_POS);
    or_bit_row(rows.b, Z_CONST_POS);

    for w in 0..8 {
        bm_write_lin(&mut rows, cv_bit(w, 0), &cv[w]);
    }
    for i in 0..16 {
        bm_write_lin(&mut rows, m_bit(i, 0), &m[i]);
    }
    bm_write_lin(&mut rows, T_LO_BASE, &counter_lo);
    bm_write_lin(&mut rows, T_HI_BASE, &counter_hi);
    bm_write_lin(&mut rows, BLEN_BASE, &block_len);
    bm_write_lin(&mut rows, FLAGS_BASE, &flags);

    let mut state: [[u32; BM_V]; 16] = [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        [BLAKE3_IV[0]; BM_V],
        [BLAKE3_IV[1]; BM_V],
        [BLAKE3_IV[2]; BM_V],
        [BLAKE3_IV[3]; BM_V],
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];
    let msg_idx = per_round_msg_idx();
    for r in 0..N_ROUNDS {
        for g_in_round in 0..N_G_PER_ROUND {
            let g = r * N_G_PER_ROUND + g_in_round;
            let [la, lb, lc, ld] = G_LANES[g_in_round];
            let [mx_i, my_i] = msg_idx[r][g_in_round];
            let mx = m[mx_i];
            let my = m[my_i];

            let a_val = state[la];
            let b_val = state[lb];
            let c_val = state[lc];
            let d_val = state[ld];

            let tmp_0 = bm_add_inline(&mut rows, &a_val, &b_val, g_add_carry_bit(g, ADD_TMP0, 0));
            let a_1 = bm_add_inline(&mut rows, &tmp_0, &mx, g_add_carry_bit(g, ADD_A1, 0));
            let d_1 = bm_xor_rotr(&d_val, &a_1, 16);
            let c_1 = bm_add_inline(&mut rows, &c_val, &d_1, g_add_carry_bit(g, ADD_C1, 0));
            let b_1 = bm_xor_rotr(&b_val, &c_1, 12);
            let tmp_1 = bm_add_inline(&mut rows, &a_1, &b_1, g_add_carry_bit(g, ADD_TMP1, 0));
            let a_2 = bm_add_inline(&mut rows, &tmp_1, &my, g_add_carry_bit(g, ADD_A2, 0));
            let d_2 = bm_xor_rotr(&d_1, &a_2, 8);
            let c_2 = bm_add_inline(&mut rows, &c_1, &d_2, g_add_carry_bit(g, ADD_C2, 0));
            let b_new = bm_xor_rotr(&b_1, &c_2, 7);
            let d_new = d_2;
            bm_write_lin(&mut rows, g_lin_bit(g, LIN_B_NEW, 0), &b_new);
            bm_write_lin(&mut rows, g_lin_bit(g, LIN_D_NEW, 0), &d_new);

            state[la] = a_2;
            state[lb] = b_new;
            state[lc] = c_2;
            state[ld] = d_new;
        }
    }

    for w in 0..8 {
        let mut lo = [0u32; BM_V];
        let mut hi = [0u32; BM_V];
        for j in 0..BM_V {
            let (l, h) = blake3_finalize_word_pair(state[w][j], state[w + 8][j], cv[w][j]);
            lo[j] = l;
            hi[j] = h;
        }
        bm_write_lin(&mut rows, out_lo_bit(w, 0), &lo);
        bm_write_lin(&mut rows, out_hi_bit(w, 0), &hi);
    }
}

pub fn generate_witness_batch_major(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<flock_core::field::F128>,
    Vec<u8>,
) {
    if let Some(materialized) = crate::seed_pipe::materialize_spec_blocks(blocks) {
        return generate_witness_batch_major(&materialized, n_blocks_log);
    }
    let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);
    super::common::drive_witness_batch_major(
        blocks,
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
    }

    const CHUNK_START: u32 = 1 << 0;
    const CHUNK_END: u32 = 1 << 1;
    const ROOT: u32 = 1 << 3;

    #[test]
    fn batch_major_witness_matches_row_major_transposed() {
        for (n_inputs, n_log) in [(8usize, 3usize), (11, 4)] {
            let mut rng = Rng::new(0xBA7C_B3 + n_log as u64);
            let inputs: Vec<Compression> = (0..n_inputs)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
                    (cv, m, counter, 64u32, 11u32)
                })
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

        let setup = Blake3Setup::new_batch_major(256);
        let mut rng = Rng::new(0xBA7C_F013);
        let inputs: Vec<Compression> = (0..256)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
                (cv, m, counter, 64u32, 11u32)
            })
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

    #[test]
    fn layout_constants() {
        assert_eq!(CV_BASE, 0);
        assert_eq!(OUT_LO_BASE, 256);
        assert_eq!(Z_CONST_POS, 512);
        assert_eq!(M_BASE, 513);
        assert_eq!(GS_BASE, 1153);
        assert_eq!(G_STRIDE, 250);
        assert_eq!(N_G, 56);
        assert_eq!(OUT_HI_BASE, 15_153);
        assert_eq!(USEFUL_BITS, 15_409);
        assert!(USEFUL_BITS <= K);
        assert_eq!(CV_BASE % SLOT_BITS, 0);
        assert_eq!(OUT_LO_BASE % SLOT_BITS, 0);
    }

    #[test]
    fn compress_matches_blake3_crate_empty() {
        let state = blake3_compress(
            &BLAKE3_IV,
            &[0u32; 16],
            0,
            0,
            CHUNK_START | CHUNK_END | ROOT,
        );
        let mut got = [0u8; 32];
        for w in 0..8 {
            got[w * 4..w * 4 + 4].copy_from_slice(&state[w].to_le_bytes());
        }
        let expected = *::blake3::hash(b"").as_bytes();
        assert_eq!(got, expected);
    }

    #[test]
    fn compress_matches_blake3_crate_64_bytes() {
        let mut rng = Rng::new(0xDEAD_BEEF);
        let mut bytes = [0u8; 64];
        for byte in bytes.iter_mut() {
            *byte = (rng.next_u32() & 0xFF) as u8;
        }
        let mut m = [0u32; 16];
        for i in 0..16 {
            m[i] = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let state = blake3_compress(&BLAKE3_IV, &m, 0, 64, CHUNK_START | CHUNK_END | ROOT);
        let mut got = [0u8; 32];
        for w in 0..8 {
            got[w * 4..w * 4 + 4].copy_from_slice(&state[w].to_le_bytes());
        }
        let expected = *::blake3::hash(&bytes).as_bytes();
        assert_eq!(got, expected);
    }

    #[test]
    fn witness_encodes_correct_output() {
        let mut rng = Rng::new(0x1234_5678);
        let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
        let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
        let counter = ((rng.next_u32() as u64) << 32) | (rng.next_u32() as u64);
        let block_len = 64;
        let flags = CHUNK_START | CHUNK_END | ROOT;
        let z = build_block_witness(&cv, &m, counter, block_len, flags);
        let expected = blake3_compress(&cv, &m, counter, block_len, flags);
        for w in 0..8 {
            let mut got = 0u32;
            for b in 0..WORD_BITS {
                if z[out_lo_bit(w, b)] {
                    got |= 1 << b;
                }
            }
            assert_eq!(got, expected[w], "out_lo[{w}] mismatch");
            let mut got_hi = 0u32;
            for b in 0..WORD_BITS {
                if z[out_hi_bit(w, b)] {
                    got_hi |= 1 << b;
                }
            }
            assert_eq!(got_hi, expected[w + 8], "out_hi[{w}] mismatch");
        }
    }

    #[test]
    fn honest_witness_satisfies_r1cs() {
        let mut rng = Rng::new(0xCAFE_F00D);
        for &n_blocks in &[1usize, 3, 8] {
            let n_log = min_n_blocks_log(n_blocks).max(3);
            let r1cs = build_block_r1cs(n_log);
            let blocks: Vec<Compression> = (0..n_blocks)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
                })
                .collect();
            let z = generate_witness(&blocks, n_log);
            assert_eq!(z.len(), r1cs.n());
            assert!(
                r1cs.satisfies(&z),
                "witness for {n_blocks} compressions fails R1CS"
            );
        }
    }

    #[test]
    fn mutated_witness_fails() {
        let mut rng = Rng::new(0xBEEF_F00D);
        let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
        let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
        let r1cs = build_block_r1cs(3);
        let blocks = vec![(cv, m, 0u64, 64u32, 11u32)];
        let mut z = generate_witness(&blocks, 3);
        assert!(r1cs.satisfies(&z));
        z[g_add_carry_bit(10, ADD_A2, 5)] ^= true;
        assert!(
            !r1cs.satisfies(&z),
            "tampered carry bit should violate R1CS"
        );
    }

    #[test]
    fn generate_witness_with_ab_packed_matches_apply() {
        for &n_blocks in &[1usize, 4, 8] {
            let n_log = min_n_blocks_log(n_blocks).max(3);
            let r1cs = build_block_r1cs(n_log);
            let mut rng = Rng::new(0xABCD_5A55 + n_blocks as u64);
            let blocks: Vec<Compression> = (0..n_blocks)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
                })
                .collect();

            let (z, a, b) = generate_witness_with_ab_packed(&blocks, n_log);
            let a_ref = r1cs.apply_a_packed(&z);
            let b_ref = r1cs.apply_b_packed(&z);
            let c_ref = r1cs.apply_c_packed(&z);
            assert_eq!(a, a_ref, "a mismatch at n_blocks={n_blocks}");
            assert_eq!(b, b_ref, "b mismatch at n_blocks={n_blocks}");
            assert_eq!(c_ref, z, "C is not identity at n_blocks={n_blocks}");
            assert!(r1cs.satisfies_packed(&z));
        }
    }

    #[test]
    fn lincheck_circuit_matches_sparse() {
        use flock_core::lincheck::{LincheckCircuit, SparseMatrixCircuit};

        let mut rng = Rng::new(0xB1A_E3_CCA1);
        let (a_0, b_0) = build_matrices();
        let sparse = SparseMatrixCircuit::new(&a_0, &b_0);
        let walker = Blake3LincheckCircuit;
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
    fn fused_lincheck_matches_separate() {
        use flock_core::lincheck::pack_z_lincheck_from_packed;
        for &n_blocks in &[1usize, 4, 8, 13] {
            let n_log = min_n_blocks_log(n_blocks).max(3);
            let r1cs = build_block_r1cs(n_log);
            let mut rng = Rng::new(0xABCD_EF00 + n_blocks as u64);
            let blocks: Vec<Compression> = (0..n_blocks)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    (cv, m, rng.next_u32() as u64, 64u32, 11u32)
                })
                .collect();

            let (z1, a1, b1) = generate_witness_with_ab_packed(&blocks, n_log);
            let lincheck_ref = pack_z_lincheck_from_packed(&z1, r1cs.m, r1cs.k_log);
            let (z2, a2, b2, lincheck_new) =
                generate_witness_with_ab_packed_and_lincheck(&blocks, n_log);
            assert_eq!(z1, z2, "z mismatch at n_blocks={n_blocks}");
            assert_eq!(a1, a2, "a mismatch at n_blocks={n_blocks}");
            assert_eq!(b1, b2, "b mismatch at n_blocks={n_blocks}");
            assert_eq!(
                lincheck_ref, lincheck_new,
                "lincheck stripe mismatch at n_blocks={n_blocks}"
            );
        }
    }

    #[test]
    fn prove_verify_ligerito_all_profiles() {
        use flock_core::challenger::FsChallenger;
        use flock_core::pcs::ligerito::LigeritoProfile;
        let blocks: Vec<Compression> = {
            let mut rng = Rng::new(0x9A11_0F11);
            (0..256)
                .map(|_| {
                    let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                    let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                    (cv, m, 0u64, 64u32, 11u32)
                })
                .collect()
        };
        for profile in [
            LigeritoProfile::Fast,
            LigeritoProfile::Slim,
            LigeritoProfile::Secure,
        ] {
            let setup = Blake3Setup::with_profile(256, profile);
            let mut ch_p = FsChallenger::new(b"flock-blake3-prof");
            let (proof, commitment, claim_p) = setup.prove_ligerito(&blocks, &mut ch_p);
            let mut ch_v = FsChallenger::new(b"flock-blake3-prof");
            let claim_v = setup
                .verify(&commitment, &proof, &mut ch_v)
                .unwrap_or_else(|e| {
                    panic!(
                        "ligerito verify rejected for profile {}: {e:?}",
                        profile.as_str()
                    )
                });
            assert_eq!(
                claim_p,
                claim_v,
                "claim mismatch for profile {}",
                profile.as_str()
            );
        }
    }

    #[test]
    #[ignore]
    fn prove_fast_ligerito_roundtrip() {
        use flock_core::challenger::FsChallenger;
        let setup = Blake3Setup::new(256);
        let mut rng = Rng::new(0xb1a_3211e);
        let blocks: Vec<Compression> = (0..256)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, 0u64, 64u32, 11u32)
            })
            .collect();
        let mut ch_p = FsChallenger::new(b"flock-blake3-lig-v0");
        let (proof, commitment, claim_p) = setup.prove_fast(&blocks, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"flock-blake3-lig-v0");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("ligerito verify rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);
    }

    #[test]
    fn prove_ligerito_generic_matches_prove_fast() {
        use flock_core::challenger::FsChallenger;
        let setup = Blake3Setup::new(256);
        let mut rng = Rng::new(0xb1a_63112);
        let blocks: Vec<Compression> = (0..256)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, 0u64, 64u32, 11u32)
            })
            .collect();
        let mut ch_f = FsChallenger::new(b"flock-blake3-gvf");
        let (proof_f, commit_f, claim_f) = setup.prove_fast(&blocks, &mut ch_f);
        let mut ch_g = FsChallenger::new(b"flock-blake3-gvf");
        let (proof_g, commit_g, claim_g) = setup.prove_ligerito(&blocks, &mut ch_g);
        assert_eq!(commit_f.root, commit_g.root);
        assert_eq!(claim_f, claim_g);
        assert_eq!(
            bincode::serialize(&proof_f).unwrap(),
            bincode::serialize(&proof_g).unwrap(),
            "generic and fused Ligerito proofs must be byte-identical"
        );
    }

    #[test]
    #[ignore]
    fn const_pin_all_zero_rejected() {
        use flock_core::challenger::FsChallenger;

        let n = 250;
        let setup = Blake3Setup::new(n);

        let mut rng = Rng::new(0x5EED_B1A3);
        let blocks: Vec<Compression> = (0..n)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                (cv, m, rng.next_u32() as u64, 64u32, 11u32)
            })
            .collect();
        let mut ch_p = FsChallenger::new(b"honest");
        let (proof, commitment, claim_p) = setup.prove_fast(&blocks, &mut ch_p);
        let mut ch_v = FsChallenger::new(b"honest");
        let claim_v = setup
            .verify(&commitment, &proof, &mut ch_v)
            .unwrap_or_else(|e| panic!("honest padded proof rejected: {e:?}"));
        assert_eq!(claim_p, claim_v);

        let zeros: Vec<Compression> = vec![([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32); n];
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
        let (proof, commitment, _) = crate::prover::prove_fast_ligerito_from_witness(
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
        let res = setup.verify(&commitment, &proof, &mut ch_v);
        assert!(
            matches!(res, Err(flock_core::verifier::VerifyError::Lincheck(_))),
            "all-zero witness must be rejected by the constant-wire pin; got {res:?}"
        );
    }

    #[test]
    fn setup_sizes_correctly() {
        for &(n_blocks, expected_n_log) in
            &[(1usize, 3), (8, 3), (9, 4), (16, 4), (17, 5), (1000, 10)]
        {
            let setup = Blake3Setup::new(n_blocks);
            assert_eq!(setup.n_blocks_log(), expected_n_log, "n_blocks={n_blocks}");
            assert_eq!(setup.m(), K_LOG + expected_n_log);
            assert!(setup.n_block_slots() >= n_blocks);
        }
    }
}

#[cfg(test)]
mod chain_e2e_tests {
    use super::*;
    use flock_core::challenger::FsChallenger;

    struct R(u64);
    impl R {
        fn nx(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn w(&mut self) -> u32 {
            self.nx() as u32
        }
        fn cv(&mut self) -> [u32; 8] {
            let mut c = [0u32; 8];
            for x in c.iter_mut() {
                *x = self.w();
            }
            c
        }
        fn msg(&mut self) -> [u32; 16] {
            let mut m = [0u32; 16];
            for x in m.iter_mut() {
                *x = self.w();
            }
            m
        }
    }

    fn out_cv(block: &Compression) -> [u32; 8] {
        let (cv, m, ctr, blen, flags) = block;
        let st = blake3_compress(cv, m, *ctr, *blen, *flags);
        let mut o = [0u32; 8];
        o.copy_from_slice(&st[0..8]);
        o
    }

    fn honest_chain(n: usize, seed: u64) -> (Vec<Compression>, [u32; 8], [u32; 8]) {
        let mut rng = R(seed);
        let cv0 = rng.cv();
        let mut blocks = Vec::with_capacity(n);
        let mut cur = cv0;
        for _ in 0..n {
            let block: Compression = (cur, rng.msg(), rng.nx(), rng.w(), rng.w());
            cur = out_cv(&block);
            blocks.push(block);
        }
        let cv_last = cur;
        (blocks, cv0, cv_last)
    }

    #[test]
    #[ignore]
    fn chain_prove_verify_ligerito_roundtrip() {
        let setup = Blake3Setup::new(256);
        let n = setup.n_block_slots();
        let (blocks, cv0, cv_last) = honest_chain(n, 0xB3_511_3E);
        let mut chp = FsChallenger::new(b"b3-chain-lig");
        let (proof, comm) = setup.prove_chain(&blocks, &mut chp);
        let mut chv = FsChallenger::new(b"b3-chain-lig");
        setup
            .verify_chain(&comm, &proof, &cv0, &cv_last, &mut chv)
            .expect("ligerito chain must verify");
    }

    #[test]
    #[ignore]
    fn chain_wrong_endpoint_rejects() {
        let setup = Blake3Setup::new(256);
        let n = setup.n_block_slots();
        let (blocks, cv0, mut cv_last) = honest_chain(n, 0xB3_1234);

        let mut chp = FsChallenger::new(b"b3-chain");
        let (proof, comm) = setup.prove_chain(&blocks, &mut chp);

        cv_last[0] ^= 1;
        let mut chv = FsChallenger::new(b"b3-chain");
        assert!(
            setup
                .verify_chain(&comm, &proof, &cv0, &cv_last, &mut chv)
                .is_err()
        );
    }

    #[test]
    #[ignore]
    fn chain_broken_link_rejects() {
        let setup = Blake3Setup::new(256);
        let n = setup.n_block_slots();
        let (mut blocks, cv0, cv_last) = honest_chain(n, 0xB3_55);

        let mut rng = R(0xB3_999);
        blocks[2].0 = rng.cv();

        let mut chp = FsChallenger::new(b"b3-chain");
        let (proof, comm) = setup.prove_chain(&blocks, &mut chp);
        let mut chv = FsChallenger::new(b"b3-chain");
        assert!(
            setup
                .verify_chain(&comm, &proof, &cv0, &cv_last, &mut chv)
                .is_err()
        );
    }

    #[test]
    fn closed_form_seed_witness_matches_slice_witness() {
        for (log2, n_blocks, n_log, seed) in [
            (3u32, 8usize, 3usize, 0x0123_4567_89AB_CDEFu64),
            (4, 11, 4, 0xDEAD_BEEF_CAFE_BABEu64),
            (5, 32, 5, 0x9E37_79B9_7F4A_7C15u64),
        ] {
            let full_blocks = crate::seed_pipe::generate_compressions_par(log2, seed);
            let blocks = &full_blocks[..n_blocks];
            let (z_s, a_s, b_s, stripe_s) =
                generate_witness_with_ab_packed_and_lincheck(blocks, n_log);
            let (z_c, a_c, b_c, stripe_c) =
                generate_witness_with_ab_packed_and_lincheck_from_seed(log2, seed, n_blocks, n_log);
            assert_eq!(
                z_c, z_s,
                "closed z mismatch at log2={log2} n_blocks={n_blocks}"
            );
            assert_eq!(
                a_c, a_s,
                "closed a mismatch at log2={log2} n_blocks={n_blocks}"
            );
            assert_eq!(
                b_c, b_s,
                "closed b mismatch at log2={log2} n_blocks={n_blocks}"
            );
            assert_eq!(
                stripe_c, stripe_s,
                "closed stripe mismatch at log2={log2} n_blocks={n_blocks}"
            );
        }
    }
}

fn warm_publish_path(
    proof: flock_core::proof::R1csProofLigerito,
    commitment: Commitment,
) -> (flock_core::proof::R1csProofLigerito, Commitment) {
    if std::env::var_os("FLOCK_NO_SER_WARM").is_some() {
        return (proof, commitment);
    }
    let bundle = crate::proof_io::R1csProofBundleLigerito { commitment, proof };
    std::hint::black_box(bundle.to_bytes());
    let crate::proof_io::R1csProofBundleLigerito { commitment, proof } = bundle;
    (proof, commitment)
}
