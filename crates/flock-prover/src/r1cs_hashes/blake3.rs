1|//! Monolithic BLAKE3 compression-function R1CS — one R1CS instance per
2|//! `compress(cv, m, counter, block_len, flags) → state[16]` call. Encodes
3|//! the 16-word state init, all 7 rounds (8 G's per round + the message
4|//! permutation), and the final output XORs in one big sparse system.
5|//!
6|//! ## Encoding choice — "Option D" (minimum-slot)
7|//!
8|//! BLAKE3 has no AND-based Ch/Maj; the only nonlinear constraints are the
9|//! carry_aux bits of 32-bit ADDs. Per compression: 7 rounds × 8 G × 6 ADDs
10|//! × 31 carry_aux = **10,416 ANDs**. We materialize **only the irreducible
11|//! slots**:
12|//!
13|//! - **No sum-bit slots**. Each ADD's 32 sum bits expand into lin_funcs at
14|//!   the use site (`s[i] = X[i] ⊕ Y[i] ⊕ ⊕_{j<i} carry_aux[j]`).
15|//! - **No `a_new` / `c_new` lin-id slots**. Lanes 0–3 ("a" positions) and
16|//!   8–11 ("c" positions) cascade — every read of these lanes inlines the
17|//!   full chain of carry_aux references from prior G's that touched the
18|//!   lane. After 7 rounds this chain is deep, but the slot count stays
19|//!   tight enough to fit `k_log = 14`.
20|//! - **`b_new` / `d_new` lin-id slots only**. Lanes 4–7 ("b" positions) and
21|//!   12–15 ("d" positions) are materialized as 32-bit lin-id slots per G,
22|//!   so the next G's read of these lanes is a single-slot lookup. This
23|//!   breaks the cascade for half the lanes — without it, `prove`-time
24|//!   matrix density would blow up further.
25|//!
26|//! Trade-off: matrix is **substantially denser** than a "materialize all
27|//! sums" encoding, so the slow-path
28|//! `apply_{a,b,c}_packed` and `sparse_row_fold` are slower per K-block.
29|//! But K halves (2^15 → 2^14), which speeds up PCS commit/open and lets
30|//! more instances fit at the same `m`. Picks favor `prove_fast` over `prove`.
31|//!
32|//! ## Witness layout per compression block (`k_log = 14`, `k = 16,384`)
33|//!
34|//! ```text
35|//!   z[0]                       = 1                    (constant)
36|//!   z[1     ..    257)         = cv[0..8]   (8 × 32-bit words)
37|//!   z[257   ..    769)         = m[0..16]   (16 × 32-bit words)
38|//!   z[769   ..    801)         = counter_lo
39|//!   z[801   ..    833)         = counter_hi
40|//!   z[833   ..    865)         = block_len
41|//!   z[865   ..    897)         = flags
42|//!   z[897   .. 14,897)         = 56 G blocks × 250 bits each
43|//!   z[14,897 .. 15,153)        = out_lo[0..8] = state[0..8] ^ state[8..16]
44|//!   z[15,153 .. 15,409)        = out_hi[0..8] = state[8..16] ^ cv[0..8]
45|//!   z[15,409 .. 16,384)        = padding (forced to 0 by empty rows)
46|//! ```
47|//!
48|//! Per G block layout (250 bits):
49|//! ```text
50|//!   [0   .. 31)    carry_aux for ADD_TMP0  = a + b
51|//!   [31  .. 62)    carry_aux for ADD_A1    = ADD_TMP0 + mx        (→ a_1)
52|//!   [62  .. 93)    carry_aux for ADD_C1    = c + d_1              (→ c_1)
53|//!   [93  .. 124)   carry_aux for ADD_TMP1  = a_1 + b_1
54|//!   [124 .. 155)   carry_aux for ADD_A2    = ADD_TMP1 + my        (→ a_new)
55|//!   [155 .. 186)   carry_aux for ADD_C2    = c_1 + d_2            (→ c_new)
56|//!   [186 .. 218)   b_new = rotr7(b_1 ^ c_2)                (lin-id)
57|//!   [218 .. 250)   d_new = rotr8(d_1 ^ a_2)                (lin-id)
58|//! ```
59|//!
60|//! `tmp_0`, `a_1`, `c_1`, `tmp_1`, `a_2 (a_new)`, `c_2 (c_new)`, `d_1`,
61|//! `b_1`, `d_2` are NEVER materialized as slots — they're lin_funcs
62|//! evaluated at row-build time and threaded forward in the state cascade.
63|//!
64|//! ## Constraint shape (`C = I`)
65|//!
66|//! Every z-slot is the output of one R1CS row:
67|//!
68|//! | Row kind            | A_row            | B_row           | Output       |
69|//! |---------------------|------------------|-----------------|--------------|
70|//! | Constant `z[0]`     | `[0]`            | `[0]`           | `z[0]·z[0]`  |
71|//! | Input slot          | `[slot]`         | `[Z_CONST]`     | `z[slot]·1`  |
72|//! | lin-id slot         | lin_func         | `[Z_CONST]`     | lin_func·1   |
73|//! | carry_aux           | lin_func_L       | lin_func_R      | (L)·(R)      |
74|//! | Padding             | `[]`             | `[]`            | `0·0`        |
75|//!
76|//! ## What this enforces
77|//!
78|//! - The 56 G-functions execute correctly: each ADD's carry_aux witness is
79|//!   constrained to `(X[i] ⊕ cin[i]) · (Y[i] ⊕ cin[i])`, so the sum bits
80|//!   `X[i] ⊕ Y[i] ⊕ cin[i]` are the correct 32-bit sum modulo 2³².
81|//! - `b_new`, `d_new` lin-id slots equal the right XOR-rotate of prior values.
82|//! - `out_lo[w] = state[w] ^ state[w+8]` and `out_hi[w] = state[w+8] ^ cv[w]`
83|//!   (BLAKE3 finalization).
84|//!
85|//! ## What this does NOT enforce
86|//!
87|//! - **Public-input pinning**: `cv`, `m`, `counter_*`, `block_len`, `flags`
88|//!   are "free" witness bits. PCS-level openings at fixed indices will
89|//!   eventually pin them to claimed public inputs.
90|
91|use super::common::{BitRecord, add_carry_parts, or_bit_at, or_u32_at_bit, xor_dedup};
92|use flock_core::challenger::Challenger;
93|use flock_core::field::F128;
94|#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
95|use flock_core::field::gf2_128::aarch64::ghash_mul_const_vec2_neon;
96|use flock_core::merkle::HashKind;
97|use flock_core::pcs::{Commitment, PcsParams};
98|use flock_core::proof::R1csClaim;
99|use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix};
100|use flock_core::verifier;
101|
102|// ---------------------------------------------------------------------------
103|// Public constants
104|// ---------------------------------------------------------------------------
105|
106|/// Block dim: one BLAKE3 compression occupies `2^K_LOG = 16,384` z slots.
107|pub const K_LOG: usize = 14;
108|/// `k = 2^K_LOG`.
109|pub const K: usize = 1 << K_LOG;
110|/// Univariate-skip dim — must match [`flock_core::zerocheck::K_SKIP`].
111|pub const K_SKIP: usize = 6;
112|
113|/// Number of BLAKE3 rounds.
114|pub const N_ROUNDS: usize = 7;
115|/// Number of G calls per round (4 column + 4 diagonal).
116|pub const N_G_PER_ROUND: usize = 8;
117|/// Total G calls per compression.
118|pub const N_G: usize = N_ROUNDS * N_G_PER_ROUND;
119|/// Bits per BLAKE3 word.
120|pub const WORD_BITS: usize = 32;
121|
122|/// Carry_aux bits per 32-bit ADD (bit 0..30; bit 31 is the discarded
123|/// mod-2³² carry-out and isn't allocated).
124|pub const CARRY_BITS_PER_ADD: usize = WORD_BITS - 1; // 31
125|/// ADDs per G.
126|pub const ADDS_PER_G: usize = 6;
127|/// Lin-id 32-bit words per G (b_new, d_new).
128|pub const LIN_WORDS_PER_G: usize = 2;
129|/// Bits per G block (no sum-bit slots — see module docs).
130|pub const G_STRIDE: usize = ADDS_PER_G * CARRY_BITS_PER_ADD + LIN_WORDS_PER_G * WORD_BITS; // 250
131|
132|/// BLAKE3 initial hash values (identical to SHA-256 IV).
133|pub const BLAKE3_IV: [u32; 8] = [
134|    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
135|];
136|
137|/// BLAKE3 message permutation applied between rounds.
138|pub const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
139|
140|/// Lanes touched by G index `g` within a round: `[a, b, c, d]`.
141|/// First 4 are column G's, last 4 are diagonal G's.
142|pub const G_LANES: [[usize; 4]; N_G_PER_ROUND] = [
143|    [0, 4, 8, 12],
144|    [1, 5, 9, 13],
145|    [2, 6, 10, 14],
146|    [3, 7, 11, 15],
147|    [0, 5, 10, 15],
148|    [1, 6, 11, 12],
149|    [2, 7, 8, 13],
150|    [3, 4, 9, 14],
151|];
152|
153|/// Message-index pairs `(mx, my)` consumed by G index `g` within a round,
154|/// indexing into the (already-permuted) per-round message buffer.
155|pub const G_MSG_IDX: [[usize; 2]; N_G_PER_ROUND] = [
156|    [0, 1],
157|    [2, 3],
158|    [4, 5],
159|    [6, 7],
160|    [8, 9],
161|    [10, 11],
162|    [12, 13],
163|    [14, 15],
164|];
165|
166|// ---------------------------------------------------------------------------
167|// Layout positions (bit indices into the per-block z slice of length K)
168|// ---------------------------------------------------------------------------
169|
170|// **I/O-aligned layout** for the hash chain (forked from `blake3`): the input
171|// chaining value `cv` lives in aligned slot 0 and the output chaining value
172|// `out_lo` (= state[0..8] ^ state[8..16]) in aligned slot 1 — each a clean
173|// 256-bit (`2^8`) window, so the chain shift argument folds them via a single
174|// tensor opening. cv/out_lo are *exactly* 256 bits, so the slots have NO
175|// interior padding. Everything else (const, m, counters, flags, G-blocks,
176|// out_hi) packs after the two slots. The re-layout is purely a change of these
177|// base offsets — all bit placement goes through the `*_bit` accessors below.
178|pub const SLOT_BITS: usize = 256; // 2^8, one 256-bit chaining value
179|pub const CV_BASE: usize = 0; // input region, slot 0: [0, 256)
180|pub const OUT_LO_BASE: usize = SLOT_BITS; // output region, slot 1: [256, 512)
181|pub const Z_CONST_POS: usize = 2 * SLOT_BITS; // 512
182|pub const M_BASE: usize = Z_CONST_POS + 1; // 513
183|pub const T_LO_BASE: usize = M_BASE + 16 * WORD_BITS; // 1025
184|pub const T_HI_BASE: usize = T_LO_BASE + WORD_BITS; // 1057
185|pub const BLEN_BASE: usize = T_HI_BASE + WORD_BITS; // 1089
186|pub const FLAGS_BASE: usize = BLEN_BASE + WORD_BITS; // 1121
187|pub const GS_BASE: usize = FLAGS_BASE + WORD_BITS; // 1153
188|pub const OUT_HI_BASE: usize = GS_BASE + N_G * G_STRIDE; // 15,153
189|pub const USEFUL_BITS: usize = OUT_HI_BASE + 8 * WORD_BITS; // 15,409
190|
191|// G sub-block: ADD `add_idx` ∈ 0..6 (carry_aux only), then lin-id
192|// `which` ∈ 0..2.
193|const ADD_TMP0: usize = 0;
194|const ADD_A1: usize = 1;
195|const ADD_C1: usize = 2;
196|const ADD_TMP1: usize = 3;
197|const ADD_A2: usize = 4;
198|const ADD_C2: usize = 5;
199|const LIN_B_NEW: usize = 0;
200|const LIN_D_NEW: usize = 1;
201|
202|#[inline]
203|fn cv_bit(w: usize, b: usize) -> usize {
204|    debug_assert!(w < 8 && b < WORD_BITS);
205|    CV_BASE + WORD_BITS * w + b
206|}
207|#[inline]
208|fn m_bit(i: usize, b: usize) -> usize {
209|    debug_assert!(i < 16 && b < WORD_BITS);
210|    M_BASE + WORD_BITS * i + b
211|}
212|#[inline]
213|fn g_add_carry_bit(g: usize, add_idx: usize, b: usize) -> usize {
214|    debug_assert!(g < N_G && add_idx < ADDS_PER_G && b < CARRY_BITS_PER_ADD);
215|    GS_BASE + G_STRIDE * g + CARRY_BITS_PER_ADD * add_idx + b
216|}
217|#[inline]
218|fn g_lin_bit(g: usize, which: usize, b: usize) -> usize {
219|    debug_assert!(g < N_G && which < LIN_WORDS_PER_G && b < WORD_BITS);
220|    GS_BASE + G_STRIDE * g + ADDS_PER_G * CARRY_BITS_PER_ADD + WORD_BITS * which + b
221|}
222|#[inline]
223|fn out_lo_bit(w: usize, b: usize) -> usize {
224|    debug_assert!(w < 8 && b < WORD_BITS);
225|    OUT_LO_BASE + WORD_BITS * w + b
226|}
227|#[inline]
228|fn out_hi_bit(w: usize, b: usize) -> usize {
229|    debug_assert!(w < 8 && b < WORD_BITS);
230|    OUT_HI_BASE + WORD_BITS * w + b
231|}
232|
233|// ---------------------------------------------------------------------------
234|// Reference BLAKE3 compression — the witness oracle. Cross-checked against
235|// the `blake3` crate in tests.
236|// ---------------------------------------------------------------------------
237|
238|#[inline]
239|fn g_fn(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
240|    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
241|    state[d] = (state[d] ^ state[a]).rotate_right(16);
242|    state[c] = state[c].wrapping_add(state[d]);
243|    state[b] = (state[b] ^ state[c]).rotate_right(12);
244|    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
245|    state[d] = (state[d] ^ state[a]).rotate_right(8);
246|    state[c] = state[c].wrapping_add(state[d]);
247|    state[b] = (state[b] ^ state[c]).rotate_right(7);
248|}
249|
250|fn round_fn(state: &mut [u32; 16], block: &[u32; 16]) {
251|    g_fn(state, 0, 4, 8, 12, block[0], block[1]);
252|    g_fn(state, 1, 5, 9, 13, block[2], block[3]);
253|    g_fn(state, 2, 6, 10, 14, block[4], block[5]);
254|    g_fn(state, 3, 7, 11, 15, block[6], block[7]);
255|    g_fn(state, 0, 5, 10, 15, block[8], block[9]);
256|    g_fn(state, 1, 6, 11, 12, block[10], block[11]);
257|    g_fn(state, 2, 7, 8, 13, block[12], block[13]);
258|    g_fn(state, 3, 4, 9, 14, block[14], block[15]);
259|}
260|
261|fn permute(m: &mut [u32; 16]) {
262|    let mut permuted = [0u32; 16];
263|    for i in 0..16 {
264|        permuted[i] = m[MSG_PERMUTATION[i]];
265|    }
266|    *m = permuted;
267|}
268|
269|/// BLAKE3 compression function. Returns the full 16-word output state
270|/// (post-finalization XOR). For chaining, the new CV is `out[0..8]`.
271|pub fn blake3_compress(
272|    cv: &[u32; 8],
273|    block_words: &[u32; 16],
274|    counter: u64,
275|    block_len: u32,
276|    flags: u32,
277|) -> [u32; 16] {
278|    let counter_low = counter as u32;
279|    let counter_high = (counter >> 32) as u32;
280|    let mut state = [
281|        cv[0],
282|        cv[1],
283|        cv[2],
284|        cv[3],
285|        cv[4],
286|        cv[5],
287|        cv[6],
288|        cv[7],
289|        BLAKE3_IV[0],
290|        BLAKE3_IV[1],
291|        BLAKE3_IV[2],
292|        BLAKE3_IV[3],
293|        counter_low,
294|        counter_high,
295|        block_len,
296|        flags,
297|    ];
298|    let mut block = *block_words;
299|    for r in 0..N_ROUNDS {
300|        round_fn(&mut state, &block);
301|        if r + 1 < N_ROUNDS {
302|            permute(&mut block);
303|        }
304|    }
305|    for i in 0..8 {
306|        state[i] ^= state[i + 8];
307|        state[i + 8] ^= cv[i];
308|    }
309|    state
310|}
311|
312|/// Build `PER_ROUND_MSG_IDX[r][g] = (mx_idx, my_idx)` for round `r`, G index
313|/// `g` — i.e., `PERM^r [G_MSG_IDX[g]]`.
314|const fn build_per_round_msg_idx() -> [[[usize; 2]; N_G_PER_ROUND]; N_ROUNDS] {
315|    let mut perm = [0usize; 16];
316|    let mut i = 0;
317|    while i < 16 {
318|        perm[i] = i;
319|        i += 1;
320|    }
321|    let mut out = [[[0usize; 2]; N_G_PER_ROUND]; N_ROUNDS];
322|    let mut r = 0;
323|    while r < N_ROUNDS {
324|        let mut g = 0;
325|        while g < N_G_PER_ROUND {
326|            out[r][g][0] = perm[G_MSG_IDX[g][0]];
327|            out[r][g][1] = perm[G_MSG_IDX[g][1]];
328|            g += 1;
329|        }
330|        let mut next = [0usize; 16];
331|        i = 0;
332|        while i < 16 {
333|            next[i] = perm[MSG_PERMUTATION[i]];
334|            i += 1;
335|        }
336|        perm = next;
337|        r += 1;
338|    }
339|    out
340|}
341|
342|/// The BLAKE3 message schedule is input-independent. Keeping it in static
343|/// storage avoids rebuilding and copying 112 `usize` indices for every
344|/// compression during witness generation.
345|const PER_ROUND_MSG_IDX: [[[usize; 2]; N_G_PER_ROUND]; N_ROUNDS] = build_per_round_msg_idx();
346|
347|// ---------------------------------------------------------------------------
348|// Lin_func cascade — per-bit lists of slot indices XOR'd to evaluate one bit.
349|//
350|// In Option D, sum bits aren't materialized as slots; instead, the "value" of
351|// any intermediate bit is a `LinBits[i] = Vec<usize>` whose XOR equals that
352|// bit. The G-builder threads these lin_funcs forward through the state, so
353|// each lane's value at any point in the protocol is represented as a `Word`.
354|// ---------------------------------------------------------------------------
355|
356|/// A 32-bit symbolic word. `bits[i]` is a list of slot indices whose XOR
357|/// equals bit `i` of the word.
358|#[derive(Clone)]
359|struct Word {
360|    bits: [Vec<usize>; WORD_BITS],
361|}
362|
363|impl Word {
364|    fn zero() -> Self {
365|        Self {
366|            bits: std::array::from_fn(|_| Vec::new()),
367|        }
368|    }
369|    /// Construct from a 32-bit witness or lin-id slot whose 32 bits live at
370|    /// `[base + 0, base + 1, …, base + 31]`.
371|    fn from_slot_base(base: usize) -> Self {
372|        Self {
373|            bits: std::array::from_fn(|i| vec![base + i]),
374|        }
375|    }
376|    /// Construct from a 32-bit constant — bit `i` is `[Z_CONST]` if set,
377|    /// `[]` otherwise.
378|    fn from_const(val: u32) -> Self {
379|        Self {
380|            bits: std::array::from_fn(|i| {
381|                if (val >> i) & 1 == 1 {
382|                    vec![Z_CONST_POS]
383|                } else {
384|                    Vec::new()
385|                }
386|            }),
387|        }
388|    }
389|    /// Bitwise XOR, no dedup. Caller calls `dedup()` after a chain if it
390|    /// wants canonical rows.
391|    fn xor(&self, other: &Word) -> Word {
392|        let mut out = self.clone();
393|        for i in 0..WORD_BITS {
394|            out.bits[i].extend(&other.bits[i]);
395|        }
396|        out
397|    }
398|    /// `rotr(n)` — pure index permutation; doesn't touch slot lists.
399|    fn rotr(&self, n: usize) -> Word {
400|        Word {
401|            bits: std::array::from_fn(|i| self.bits[(i + n) % WORD_BITS].clone()),
402|        }
403|    }
404|    /// Sort + cancel duplicates per bit.
405|    fn dedup(mut self) -> Word {
406|        for i in 0..WORD_BITS {
407|            self.bits[i] = xor_dedup(std::mem::take(&mut self.bits[i]));
408|        }
409|        self
410|    }
411|    /// "Sum bit" lin_func of an ADD `x + y` whose carry_aux slots live at
412|    /// `[carry_base, carry_base + 31)`.
413|    ///
414|    ///   sum[i] = x[i] ⊕ y[i] ⊕ ⊕_{j<i} carry_aux[j]
415|    fn add_sum(x: &Word, y: &Word, carry_base: usize) -> Word {
416|        let mut out = Word::zero();
417|        for i in 0..WORD_BITS {
418|            let mut v = x.bits[i].clone();
419|            v.extend(&y.bits[i]);
420|            for j in 0..i {
421|                v.push(carry_base + j);
422|            }
423|            out.bits[i] = v;
424|        }
425|        out.dedup()
426|    }
427|}
428|
429|// ---------------------------------------------------------------------------
430|// Per-ADD: write the 31 carry_aux rows and return the sum-bit `Word`.
431|//
432|//   carry_aux[i] = (X[i] ⊕ cin[i]) · (Y[i] ⊕ cin[i])   (R1CS AND row)
433|//   sum[i]       = X[i] ⊕ Y[i] ⊕ cin[i]                (no slot, lin_func)
434|//
435|// where cin[i] = ⊕_{j<i} carry_aux[j].
436|// ---------------------------------------------------------------------------
437|
438|fn write_add_carry_rows(
439|    a_rows: &mut [Vec<usize>],
440|    b_rows: &mut [Vec<usize>],
441|    x: &Word,
442|    y: &Word,
443|    carry_base: usize,
444|) -> Word {
445|    for i in 0..CARRY_BITS_PER_ADD {
446|        let mut a = x.bits[i].clone();
447|        for j in 0..i {
448|            a.push(carry_base + j);
449|        }
450|        let mut b = y.bits[i].clone();
451|        for j in 0..i {
452|            b.push(carry_base + j);
453|        }
454|        a_rows[carry_base + i] = xor_dedup(a);
455|        b_rows[carry_base + i] = xor_dedup(b);
456|    }
457|    Word::add_sum(x, y, carry_base)
458|}
459|
460|// ---------------------------------------------------------------------------
461|// Initial lane sources at the start of compression.
462|// ---------------------------------------------------------------------------
463|
464|fn initial_lane_words() -> [Word; 16] {
465|    let mut s: [Word; 16] = std::array::from_fn(|_| Word::zero());
466|    for w in 0..8 {
467|        s[w] = Word::from_slot_base(cv_bit(w, 0));
468|    }
469|    for i in 0..4 {
470|        s[8 + i] = Word::from_const(BLAKE3_IV[i]);
471|    }
472|    s[12] = Word::from_slot_base(T_LO_BASE);
473|    s[13] = Word::from_slot_base(T_HI_BASE);
474|    s[14] = Word::from_slot_base(BLEN_BASE);
475|    s[15] = Word::from_slot_base(FLAGS_BASE);
476|    s
477|}
478|
479|// ---------------------------------------------------------------------------
480|// Matrix builder
481|// ---------------------------------------------------------------------------
482|
483|/// Build the per-block base matrices `(A_0, B_0)`. `C_0 = I_k` (circuit-shape
484|/// R1CS — every z slot is the output of its row).
485|pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
486|    let mut a_rows: Vec<Vec<usize>> = vec![Vec::new(); K];
487|    let mut b_rows: Vec<Vec<usize>> = vec![Vec::new(); K];
488|
489|    // Constant z[0]: z[0]·z[0] = z[0]. Trivially satisfied for any boolean.
490|    a_rows[Z_CONST_POS] = vec![Z_CONST_POS];
491|    b_rows[Z_CONST_POS] = vec![Z_CONST_POS];
492|
493|    // Input rows for cv, m, counter_lo, counter_hi, block_len, flags.
494|    let mut input_emit = |base: usize, len: usize| {
495|        for j in 0..len {
496|            let s = base + j;
497|            a_rows[s] = vec![s];
498|            b_rows[s] = vec![Z_CONST_POS];
499|        }
500|    };
501|    input_emit(CV_BASE, 8 * WORD_BITS);
502|    input_emit(M_BASE, 16 * WORD_BITS);
503|    input_emit(T_LO_BASE, WORD_BITS);
504|    input_emit(T_HI_BASE, WORD_BITS);
505|    input_emit(BLEN_BASE, WORD_BITS);
506|    input_emit(FLAGS_BASE, WORD_BITS);
507|
508|    let msg_idx = &PER_ROUND_MSG_IDX;
509|    let mut state: [Word; 16] = initial_lane_words();
510|
511|    for r in 0..N_ROUNDS {
512|        for g_in_round in 0..N_G_PER_ROUND {
513|            let g = r * N_G_PER_ROUND + g_in_round;
514|            let [la, lb, lc, ld] = G_LANES[g_in_round];
515|            let [mx_idx, my_idx] = msg_idx[r][g_in_round];
516|
517|            // Snapshot inputs before any state mutation. Cloning is cheap
518|            // (lane Words point at the same slot lists — we never alias).
519|            let a = state[la].clone();
520|            let b = state[lb].clone();
521|            let c = state[lc].clone();
522|            let d = state[ld].clone();
523|            let mx = Word::from_slot_base(m_bit(mx_idx, 0));
524|            let my = Word::from_slot_base(m_bit(my_idx, 0));
525|
526|            // tmp_0 = a + b
527|            let tmp_0 = write_add_carry_rows(
528|                &mut a_rows,
529|                &mut b_rows,
530|                &a,
531|                &b,
532|                g_add_carry_bit(g, ADD_TMP0, 0),
533|            );
534|            // a_1 = tmp_0 + mx
535|            let a_1 = write_add_carry_rows(
536|                &mut a_rows,
537|                &mut b_rows,
538|                &tmp_0,
539|                &mx,
540|                g_add_carry_bit(g, ADD_A1, 0),
541|            );
542|            // d_1 = rotr16(d ^ a_1)
543|            let d_1 = d.xor(&a_1).dedup().rotr(16);
544|            // c_1 = c + d_1
545|            let c_1 = write_add_carry_rows(
546|                &mut a_rows,
547|                &mut b_rows,
548|                &c,
549|                &d_1,
550|                g_add_carry_bit(g, ADD_C1, 0),
551|            );
552|            // b_1 = rotr12(b ^ c_1)
553|            let b_1 = b.xor(&c_1).dedup().rotr(12);
554|            // tmp_1 = a_1 + b_1
555|            let tmp_1 = write_add_carry_rows(
556|                &mut a_rows,
557|                &mut b_rows,
558|                &a_1,
559|                &b_1,
560|                g_add_carry_bit(g, ADD_TMP1, 0),
561|            );
562|            // a_2 = tmp_1 + my   (= a_new — cascades)
563|            let a_2 = write_add_carry_rows(
564|                &mut a_rows,
565|                &mut b_rows,
566|                &tmp_1,
567|                &my,
568|                g_add_carry_bit(g, ADD_A2, 0),
569|            );
570|            // d_2 = rotr8(d_1 ^ a_2)
571|            let d_2 = d_1.xor(&a_2).dedup().rotr(8);
572|            // c_2 = c_1 + d_2    (= c_new — cascades)
573|            let c_2 = write_add_carry_rows(
574|                &mut a_rows,
575|                &mut b_rows,
576|                &c_1,
577|                &d_2,
578|                g_add_carry_bit(g, ADD_C2, 0),
579|            );
580|            // b_new = rotr7(b_1 ^ c_2)    (materialized lin-id)
581|            let b_new_word = b_1.xor(&c_2).dedup().rotr(7);
582|            for i in 0..WORD_BITS {
583|                let s = g_lin_bit(g, LIN_B_NEW, i);
584|                a_rows[s] = b_new_word.bits[i].clone();
585|                b_rows[s] = vec![Z_CONST_POS];
586|            }
587|            // d_new = d_2                  (materialized lin-id)
588|            for i in 0..WORD_BITS {
589|                let s = g_lin_bit(g, LIN_D_NEW, i);
590|                a_rows[s] = d_2.bits[i].clone();
591|                b_rows[s] = vec![Z_CONST_POS];
592|            }
593|
594|            // Advance the symbolic state. `a_2` and `c_2` keep cascading;
595|            // `b_new` and `d_new` reset to single-slot lookups.
596|            state[la] = a_2;
597|            state[lb] = Word::from_slot_base(g_lin_bit(g, LIN_B_NEW, 0));
598|            state[lc] = c_2;
599|            state[ld] = Word::from_slot_base(g_lin_bit(g, LIN_D_NEW, 0));
600|        }
601|    }
602|
603|    // Finalization XORs.
604|    //   out_lo[w] = state[w] ^ state[w+8]
605|    //   out_hi[w] = state[w+8] ^ cv[w]
606|    for w in 0..8 {
607|        let lo = state[w].xor(&state[w + 8]).dedup();
608|        for i in 0..WORD_BITS {
609|            let s = out_lo_bit(w, i);
610|            a_rows[s] = lo.bits[i].clone();
611|            b_rows[s] = vec![Z_CONST_POS];
612|        }
613|        let cv_w = Word::from_slot_base(cv_bit(w, 0));
614|        let hi = state[w + 8].xor(&cv_w).dedup();
615|        for i in 0..WORD_BITS {
616|            let s = out_hi_bit(w, i);
617|            a_rows[s] = hi.bits[i].clone();
618|            b_rows[s] = vec![Z_CONST_POS];
619|        }
620|    }
621|
622|    // Padding rows [USEFUL_BITS..K): A = B = []. Constraint 0·0 = z[i]
623|    // forces z[i] = 0 for all padding bits.
624|
625|    let to_mat = |rows| SparseBinaryMatrix {
626|        num_rows: K,
627|        num_cols: K,
628|        rows,
629|    };
630|    (to_mat(a_rows), to_mat(b_rows))
631|}
632|
633|/// Build a [`BlockR1cs`] batching `2^n_blocks_log` independent BLAKE3
634|/// compressions. `n_blocks_log ≥ 3` is required (lincheck needs `n_outer ≥ 8`).
635|pub fn build_block_r1cs(n_blocks_log: usize) -> BlockR1cs {
636|    let (a_0, b_0) = build_matrices();
637|    super::common::build_block_r1cs_with_matrices(
638|        n_blocks_log,
639|        K_LOG,
640|        K_SKIP,
641|        USEFUL_BITS,
642|        a_0,
643|        b_0,
644|        // Constant-wire pin (docs/const-wire-pin.md): forces z[Z_CONST_POS] = 1
645|        // in every block. Requires padding blocks filled with valid compressions.
646|        Some(Z_CONST_POS),
647|    )
648|}
649|
650|// ---------------------------------------------------------------------------
651|// Lincheck circuit walker — mirrors `build_matrices`. Same structure as
652|// `blake3::Blake3LincheckCircuit` but uses this module's I/O-aligned slot
653|// positions (cv_bit/m_bit/etc.).
654|// ---------------------------------------------------------------------------
655|
656|/// One node in the compact linear-expression DAG used by the reverse
657|/// transpose.  Unlike [`Word`], this never expands an intermediate into its
658|/// (potentially very large) set of source columns.
659|#[derive(Clone, Copy)]
660|enum ReverseWordOp {
661|    Leaf(usize),
662|    Constant(u32),
663|    Add {
664|        x: usize,
665|        y: usize,
666|        carry_base: usize,
667|    },
668|    XorRot {
669|        x: usize,
670|        y: usize,
671|        rotation: usize,
672|    },
673|}
674|
675|#[inline]
676|fn mul_alpha_pair(alpha: F128, values: [F128; 2]) -> [F128; 2] {
677|    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
678|    {
679|        // SAFETY: this branch is compiled only when AES/PMULL is enabled.
680|        unsafe { ghash_mul_const_vec2_neon(alpha, values) }
681|    }
682|    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
683|    {
684|        [alpha * values[0], alpha * values[1]]
685|    }
686|}
687|
688|/// Reverse-mode evaluator for `alpha * A_0^T * eq + B_0^T * eq`.
689|///
690|/// Each logical 32-bit value is a DAG node.  Row weights are first attached
691|/// to the values read by carry and lin-id rows, then one reverse sweep moves
692|/// those weights to witness columns.  This makes work proportional to the
693|/// BLAKE3 dependency graph rather than to the ~21M entries in its expanded
694|/// sparse matrices.
695|struct ReverseTranspose<'a> {
696|    alpha: F128,
697|    eq_inner: &'a [F128],
698|    ops: Vec<ReverseWordOp>,
699|    adjoints: Vec<[F128; WORD_BITS]>,
700|    comb: Vec<F128>,
701|}
702|
703|impl<'a> ReverseTranspose<'a> {
704|    fn new(alpha: F128, eq_inner: &'a [F128]) -> Self {
705|        Self {
706|            alpha,
707|            eq_inner,
708|            ops: Vec::with_capacity(32 + 12 * N_G + 16),
709|            adjoints: Vec::with_capacity(32 + 12 * N_G + 16),
710|            comb: vec![F128::ZERO; K],
711|        }
712|    }
713|
714|    #[inline]
715|    fn push(&mut self, op: ReverseWordOp) -> usize {
716|        let id = self.ops.len();
717|        self.ops.push(op);
718|        self.adjoints.push([F128::ZERO; WORD_BITS]);
719|        id
720|    }
721|
722|    #[inline]
723|    fn leaf(&mut self, base: usize) -> usize {
724|        self.push(ReverseWordOp::Leaf(base))
725|    }
726|
727|    #[inline]
728|    fn constant(&mut self, value: u32) -> usize {
729|        self.push(ReverseWordOp::Constant(value))
730|    }
731|
732|    #[inline]
733|    fn xor_rot(&mut self, x: usize, y: usize, rotation: usize) -> usize {
734|        self.push(ReverseWordOp::XorRot { x, y, rotation })
735|    }
736|
737|    /// Register one carry-only addition and all 31 nonlinear rows that define
738|    /// its carry columns.
739|    #[inline]
740|    fn add(&mut self, x: usize, y: usize, carry_base: usize) -> usize {
741|        let out = self.push(ReverseWordOp::Add { x, y, carry_base });
742|
743|        // Row i reads x[i] in A, y[i] in B, and carry[0..i] in both.
744|        // Accumulating the latter backwards turns the triangular row walk into
745|        // one suffix scan over the 31 carry columns.
746|        let mut suffix = F128::ZERO;
747|        let mut remaining = CARRY_BITS_PER_ADD;
748|        while remaining >= 2 {
749|            let hi = remaining - 1;
750|            let lo = remaining - 2;
751|            let e_hi = self.eq_inner[carry_base + hi];
752|            let e_lo = self.eq_inner[carry_base + lo];
753|            let [alpha_e_hi, alpha_e_lo] = mul_alpha_pair(self.alpha, [e_hi, e_lo]);
754|
755|            self.comb[carry_base + hi] += suffix;
756|            self.adjoints[x][hi] += alpha_e_hi;
757|            self.adjoints[y][hi] += e_hi;
758|            suffix += alpha_e_hi + e_hi;
759|
760|            // The lower row sees the suffix after the higher row is included.
761|            self.comb[carry_base + lo] += suffix;
762|            self.adjoints[x][lo] += alpha_e_lo;
763|            self.adjoints[y][lo] += e_lo;
764|            suffix += alpha_e_lo + e_lo;
765|            remaining -= 2;
766|        }
767|
768|        if remaining == 1 {
769|            let e = self.eq_inner[carry_base];
770|            let alpha_e = self.alpha * e;
771|            self.comb[carry_base] += suffix;
772|            self.adjoints[x][0] += alpha_e;
773|            self.adjoints[y][0] += e;
774|        }
775|        out
776|    }
777|
778|    /// Attach the A-side weight of a lin-id row to its defining expression;
779|    /// its B-side is the constant-one column.
780|    #[inline]
781|    fn seed_lin_row(&mut self, value: usize, bit: usize, row: usize) {
782|        let e = self.eq_inner[row];
783|        self.adjoints[value][bit] += self.alpha * e;
784|        self.comb[Z_CONST_POS] += e;
785|    }
786|
787|    /// Attach two independent lin-id rows while sharing their A-side scale.
788|    #[inline]
789|    fn seed_lin_rows2(
790|        &mut self,
791|        first_value: usize,
792|        second_value: usize,
793|        bit: usize,
794|        first_row: usize,
795|        second_row: usize,
796|    ) {
797|        let first_e = self.eq_inner[first_row];
798|        let second_e = self.eq_inner[second_row];
799|        let [first_alpha_e, second_alpha_e] = mul_alpha_pair(self.alpha, [first_e, second_e]);
800|
801|        self.adjoints[first_value][bit] += first_alpha_e;
802|        self.comb[Z_CONST_POS] += first_e;
803|        self.adjoints[second_value][bit] += second_alpha_e;
804|        self.comb[Z_CONST_POS] += second_e;
805|    }
806|
807|    fn finish(mut self) -> Vec<F128> {
808|        for id in (0..self.ops.len()).rev() {
809|            // F128 is Copy, so taking this 32-lane value avoids aliasing the
810|            // current node while predecessor adjoints are updated.
811|            let q = self.adjoints[id];
812|            match self.ops[id] {
813|                ReverseWordOp::Leaf(base) => {
814|                    for (i, value) in q.into_iter().enumerate() {
815|                        self.comb[base + i] += value;
816|                    }
817|                }
818|                ReverseWordOp::Constant(value) => {
819|                    for (i, weight) in q.into_iter().enumerate() {
820|                        if (value >> i) & 1 == 1 {
821|                            self.comb[Z_CONST_POS] += weight;
822|                        }
823|                    }
824|                }
825|                ReverseWordOp::XorRot { x, y, rotation } => {
826|                    for (i, weight) in q.into_iter().enumerate() {
827|                        let source_bit = (i + rotation) % WORD_BITS;
828|                        self.adjoints[x][source_bit] += weight;
829|                        self.adjoints[y][source_bit] += weight;
830|                    }
831|                }
832|                ReverseWordOp::Add { x, y, carry_base } => {
833|                    // sum[i] = x[i] + y[i] + carry[0] + ... + carry[i-1].
834|                    // The reverse of the carry prefix is another suffix scan.
835|                    let mut suffix = F128::ZERO;
836|                    for i in (0..WORD_BITS).rev() {
837|                        if i < CARRY_BITS_PER_ADD {
838|                            self.comb[carry_base + i] += suffix;
839|                        }
840|                        let weight = q[i];
841|                        self.adjoints[x][i] += weight;
842|                        self.adjoints[y][i] += weight;
843|                        suffix += weight;
844|                    }
845|                }
846|            }
847|        }
848|        self.comb
849|    }
850|}
851|
852|pub struct Blake3LincheckCircuit;
853|
854|impl flock_core::lincheck::LincheckCircuit for Blake3LincheckCircuit {
855|    fn n_cols(&self) -> usize {
856|        K
857|    }
858|
859|    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
860|        assert_eq!(eq_inner.len(), K, "eq_inner length must equal n_cols = K");
861|        let mut reverse = ReverseTranspose::new(alpha, eq_inner);
862|
863|        // Rows whose A side is the input itself and whose B side is one.
864|        let e0 = eq_inner[Z_CONST_POS];
865|        reverse.comb[Z_CONST_POS] += alpha * e0 + e0;
866|        let input_emit = |reverse: &mut ReverseTranspose<'_>, base: usize, len: usize| {
867|            for s in base..base + len {
868|                let e = reverse.eq_inner[s];
869|                reverse.comb[s] += reverse.alpha * e;
870|                reverse.comb[Z_CONST_POS] += e;
871|            }
872|        };
873|        input_emit(&mut reverse, CV_BASE, 8 * WORD_BITS);
874|        input_emit(&mut reverse, M_BASE, 16 * WORD_BITS);
875|        input_emit(&mut reverse, T_LO_BASE, WORD_BITS);
876|        input_emit(&mut reverse, T_HI_BASE, WORD_BITS);
877|        input_emit(&mut reverse, BLEN_BASE, WORD_BITS);
878|        input_emit(&mut reverse, FLAGS_BASE, WORD_BITS);
879|
880|        // Unique source nodes preserve matrix column order.  Message words are
881|        // shared across every scheduled use, while each materialized b/d word
882|        // below gets a fresh leaf at its exact G-block offset.
883|        let cv: [usize; 8] = std::array::from_fn(|w| reverse.leaf(cv_bit(w, 0)));
884|        let messages: [usize; 16] = std::array::from_fn(|w| reverse.leaf(m_bit(w, 0)));
885|        let mut state: [usize; 16] = [
886|            cv[0],
887|            cv[1],
888|            cv[2],
889|            cv[3],
890|            cv[4],
891|            cv[5],
892|            cv[6],
893|            cv[7],
894|            reverse.constant(BLAKE3_IV[0]),
895|            reverse.constant(BLAKE3_IV[1]),
896|            reverse.constant(BLAKE3_IV[2]),
897|            reverse.constant(BLAKE3_IV[3]),
898|            reverse.leaf(T_LO_BASE),
899|            reverse.leaf(T_HI_BASE),
900|            reverse.leaf(BLEN_BASE),
901|900|            reverse.leaf(BLEN_BASE),
901|            reverse.leaf(FLAGS_BASE),
902|        ];
903|
904|        for r in 0..N_ROUNDS {
905|            for g_in_round in 0..N_G_PER_ROUND {
906|                let g = r * N_G_PER_ROUND + g_in_round;
907|                let [la, lb, lc, ld] = G_LANES[g_in_round];
908|                let [mx_idx, my_idx] = PER_ROUND_MSG_IDX[r][g_in_round];
909|                let [a, b, c, d] = [state[la], state[lb], state[lc], state[ld]];
910|
911|                let tmp_0 = reverse.add(a, b, g_add_carry_bit(g, ADD_TMP0, 0));
912|                let a_1 = reverse.add(tmp_0, messages[mx_idx], g_add_carry_bit(g, ADD_A1, 0));
913|                let d_1 = reverse.xor_rot(d, a_1, 16);
914|                let c_1 = reverse.add(c, d_1, g_add_carry_bit(g, ADD_C1, 0));
915|                let b_1 = reverse.xor_rot(b, c_1, 12);
916|                let tmp_1 = reverse.add(a_1, b_1, g_add_carry_bit(g, ADD_TMP1, 0));
917|                let a_2 = reverse.add(tmp_1, messages[my_idx], g_add_carry_bit(g, ADD_A2, 0));
918|                let d_2 = reverse.xor_rot(d_1, a_2, 8);
919|                let c_2 = reverse.add(c_1, d_2, g_add_carry_bit(g, ADD_C2, 0));
920|
921|                let b_new = reverse.xor_rot(b_1, c_2, 7);
922|                for i in 0..WORD_BITS {
923|                    reverse.seed_lin_rows2(
924|                        b_new,
925|                        d_2,
926|                        i,
927|                        g_lin_bit(g, LIN_B_NEW, i),
928|                        g_lin_bit(g, LIN_D_NEW, i),
929|                    );
930|                }
931|
932|                state[la] = a_2;
933|                state[lb] = reverse.leaf(g_lin_bit(g, LIN_B_NEW, 0));
934|                state[lc] = c_2;
935|                state[ld] = reverse.leaf(g_lin_bit(g, LIN_D_NEW, 0));
936|            }
937|        }
938|
939|        // Finalization lin-id rows.  These nodes are seeded in physical output
940|        // coordinate order, exactly as build_matrices writes the rows.
941|        for w in 0..8 {
942|            let lo = reverse.xor_rot(state[w], state[w + 8], 0);
943|            let hi = reverse.xor_rot(state[w + 8], cv[w], 0);
944|            for i in 0..WORD_BITS {
945|                reverse.seed_lin_row(lo, i, out_lo_bit(w, i));
946|                reverse.seed_lin_row(hi, i, out_hi_bit(w, i));
947|            }
948|        }
949|
950|        reverse.finish()
951|    }
952|
953|    fn const_pin_col(&self) -> Option<usize> {
954|        Some(Z_CONST_POS)
955|    }
956|}
957|
958|// ---------------------------------------------------------------------------
959|// Witness generation (boolean)
960|// ---------------------------------------------------------------------------
961|
962|/// Compute one 32-bit ADD, writing 31 carry_aux bits into `z` at `carry_base`.
963|/// Returns `x.wrapping_add(y)` (sum bits are NOT materialized in this
964|/// encoding — see module docs).
965|fn add_with_witness_carry_only(x: u32, y: u32, z: &mut [bool], carry_base: usize) -> u32 {
    let mut cin: u32 = 0;
    let mut sum: u32 = 0;

    // Process all 32 bits efficiently
    for i in 0..WORD_BITS {
        let xi = (x >> i) & 1;
        let yi = (y >> i) & 1;
        let ci = (cin >> i) & 1;
        
        // Compute carry_aux: (xi ⊕ ci) ∧ (yi ⊕ ci)  
        let carry_aux = (xi ^ ci) & (yi ^ ci);
        if i < CARRY_BITS_PER_ADD {
            z[carry_base + i] = carry_aux == 1;
        }
        
        // Compute real_carry for next iteration: carry_aux ⊕ ci
        let real_carry = carry_aux ^ ci;
        
        // Build sum bit: xi ⊕ yi ⊕ ci
        let sum_bit = xi ^ yi ^ ci;
        sum |= sum_bit << i;
        
        // Update carry-in for next bit (but don't shift beyond bit 31)
        if i < WORD_BITS - 1 {
            cin |= real_carry << (i + 1);
        }
    }

    sum
979|}
980|
981|#[inline]
982|fn write_word(z: &mut [bool], base: usize, val: u32) {
983|    for i in 0..WORD_BITS {
984|        z[base + i] = ((val >> i) & 1) == 1;
985|    }
986|}
987|
988|/// Build the witness block for ONE compression. Length = `K`.
989|pub fn build_block_witness(
990|    cv: &[u32; 8],
991|    m: &[u32; 16],
992|    counter: u64,
993|    block_len: u32,
994|    flags: u32,
995|) -> Vec<bool> {
996|    let mut z = vec![false; K];
997|    z[Z_CONST_POS] = true;
998|    // Inputs.
999|    for w in 0..8 {
1000|        write_word(&mut z, cv_bit(w, 0), cv[w]);
1001|    }
1002|    for i in 0..16 {
1003|        write_word(&mut z, m_bit(i, 0), m[i]);
1004|    }
1005|    let counter_lo = counter as u32;
1006|    let counter_hi = (counter >> 32) as u32;
1007|    write_word(&mut z, T_LO_BASE, counter_lo);
1008|    write_word(&mut z, T_HI_BASE, counter_hi);
1009|    write_word(&mut z, BLEN_BASE, block_len);
1010|    write_word(&mut z, FLAGS_BASE, flags);
1011|
1012|    // Internal state evolution (matches the matrix builder's symbolic
1013|    // cascade by construction).
1014|    let mut state: [u32; 16] = [
1015|        cv[0],
1016|        cv[1],
1017|        cv[2],
1018|        cv[3],
1019|        cv[4],
1020|        cv[5],
1021|        cv[6],
1022|        cv[7],
1023|        BLAKE3_IV[0],
1024|        BLAKE3_IV[1],
1025|        BLAKE3_IV[2],
1026|        BLAKE3_IV[3],
1027|        counter_lo,
1028|        counter_hi,
1029|        block_len,
1030|        flags,
1031|    ];
1032|    let msg_idx = &PER_ROUND_MSG_IDX;
1033|
1034|    for r in 0..N_ROUNDS {
1035|        for g_in_round in 0..N_G_PER_ROUND {
1036|            let g = r * N_G_PER_ROUND + g_in_round;
1037|            let [la, lb, lc, ld] = G_LANES[g_in_round];
1038|            let [mx_i, my_i] = msg_idx[r][g_in_round];
1039|            let mx = m[mx_i];
1040|            let my = m[my_i];
1041|
1042|            let a = state[la];
1043|            let b = state[lb];
1044|            let c = state[lc];
1045|            let d = state[ld];
1046|
1047|            let tmp_0 = add_with_witness_carry_only(a, b, &mut z, g_add_carry_bit(g, ADD_TMP0, 0));
1048|            let a_1 = add_with_witness_carry_only(tmp_0, mx, &mut z, g_add_carry_bit(g, ADD_A1, 0));
1049|            let d_1 = (d ^ a_1).rotate_right(16);
1050|            let c_1 = add_with_witness_carry_only(c, d_1, &mut z, g_add_carry_bit(g, ADD_C1, 0));
1051|            let b_1 = (b ^ c_1).rotate_right(12);
1052|            let tmp_1 =
1053|                add_with_witness_carry_only(a_1, b_1, &mut z, g_add_carry_bit(g, ADD_TMP1, 0));
1054|            let a_2 = add_with_witness_carry_only(tmp_1, my, &mut z, g_add_carry_bit(g, ADD_A2, 0));
1055|            let d_2 = (d_1 ^ a_2).rotate_right(8);
1056|            let c_2 = add_with_witness_carry_only(c_1, d_2, &mut z, g_add_carry_bit(g, ADD_C2, 0));
1057|            let b_new = (b_1 ^ c_2).rotate_right(7);
1058|            let d_new = d_2;
1059|            write_word(&mut z, g_lin_bit(g, LIN_B_NEW, 0), b_new);
1060|            write_word(&mut z, g_lin_bit(g, LIN_D_NEW, 0), d_new);
1061|
1062|            state[la] = a_2;
1063|            state[lb] = b_new;
1064|            state[lc] = c_2;
1065|            state[ld] = d_new;
1066|        }
1067|    }
1068|
1069|    for w in 0..8 {
1070|        let lo = state[w] ^ state[w + 8];
1071|        let hi = state[w + 8] ^ cv[w];
1072|        write_word(&mut z, out_lo_bit(w, 0), lo);
1073|        write_word(&mut z, out_hi_bit(w, 0), hi);
1074|    }
1075|    z
1076|}
1077|
1078|/// Minimum `n_blocks_log` needed to prove `n_blocks` BLAKE3 compressions,
1079|/// subject to the lincheck floor of `n_blocks_log ≥ 3` (`n_outer ≥ 8`).
1080|pub fn min_n_blocks_log(n_blocks: usize) -> usize {
1081|    assert!(n_blocks >= 1, "n_blocks must be ≥ 1");
1082|    let n = n_blocks.max(8);
1083|    n.next_power_of_two().trailing_zeros() as usize
1084|}
1085|
1086|/// One BLAKE3 compression input: `(cv, m, counter, block_len, flags)`.
1087|pub type Compression = ([u32; 8], [u32; 16], u64, u32, u32);
1088|
1089|/// Generate the boolean witness vector for `blocks.len()` independent BLAKE3
1090|/// compressions, padded to `2^n_blocks_log` slots. Padding blocks are
1091|/// all-zero (trivially satisfy the R1CS). Parallel across instances via rayon.
1092|pub fn generate_witness(blocks: &[Compression], n_blocks_log: usize) -> Vec<bool> {
1093|    use rayon::prelude::*;
1094|    let n_total = 1usize << n_blocks_log;
1095|    let n_blocks = blocks.len();
1096|    assert!(
1097|        n_blocks <= n_total,
1098|        "{n_blocks} compressions > 2^{n_blocks_log} = {n_total} slots"
1099|    );
1100|1100|    let mut z = vec![false; n_total * K];
1101|    z.par_chunks_mut(K)
1102|        .take(n_blocks)
1103|        .zip(blocks.par_iter())
1104|        .for_each(|(chunk, (cv, m, t, b, d))| {
1105|            let block = build_block_witness(cv, m, *t, *b, *d);
1106|            chunk.copy_from_slice(&block);
1107|        });
1108|    z
1109|}
1110|
1111|// ---------------------------------------------------------------------------
1112|// Fast witness generation with (a, b, c) — emits the R1CS row-witnesses
1113|// directly from the BLAKE3 computation, in F_{2^128}-packed form. Skips the
1114|// `apply_block_diag_packed` pass downstream.
1115|//
1116|// Row-witness semantics (matching `build_matrices`):
1117|// - Constant z[0]:       (z, a, b, c) = (1, 1, 1, 1).
1118|// - Input slot:          (z, a, b, c) = (val, val, 1, val).
1119|// - Lin-id slot:         (z, a, b, c) = (lin_val, lin_val, 1, lin_val).
1120|// - Carry_aux row i:     (z, a, b, c) = (carry_aux, X⊕cin, Y⊕cin, carry_aux).
1121|// - Padding row:         all zero (already zero on entry).
1122|// ---------------------------------------------------------------------------
1123|
1124|/// One 32-bit ADD: returns `(sum, left, right, carry_aux)` for the caller to
1125|/// place into the per-G records. Sum bits are NOT materialized in this
1126|/// encoding (Option D).
1127|///
1128|/// **c is not written.** Since `C = I` in this R1CS, `c == z` byte-for-byte,
1129|/// so callers can use `z_packed` directly as the c-side input to zerocheck —
1130|/// no separate c buffer is needed.
1131|///
1132|/// Word-level derivation:
1133|/// ```text
1134|///   sum       = x + y (mod 2^32)
1135|///   cin       = sum ⊕ x ⊕ y          (since sum[i] = x[i] ⊕ y[i] ⊕ cin[i])
1136|///   left      = x ⊕ cin              (per-bit X ⊕ cin → operand_x of carry row)
1137|///   right     = y ⊕ cin              (per-bit Y ⊕ cin → operand_y of carry row)
1138|///   carry_aux = left ∧ right
1139|/// ```
1140|/// Bit 31 is the discarded mod-2³² carry-out and is masked off so the
1141|/// record push doesn't spill into the next slot.
1142|// Record-relative positions: carries at 31·i, lin words after all carries.
1143|const REC_C0: usize = 0;
1144|const REC_C1: usize = CARRY_BITS_PER_ADD;
1145|const REC_C2: usize = 2 * CARRY_BITS_PER_ADD;
1146|const REC_C3: usize = 3 * CARRY_BITS_PER_ADD;
1147|const REC_C4: usize = 4 * CARRY_BITS_PER_ADD;
1148|const REC_C5: usize = 5 * CARRY_BITS_PER_ADD;
1149|const REC_LIN0: usize = ADDS_PER_G * CARRY_BITS_PER_ADD;
1150|const REC_LIN1: usize = REC_LIN0 + WORD_BITS;
1151|
1152|/// Write a 32-bit lin-id (or input) slot: (z, a) = val, b = all-ones.
1153|/// **c is not written** — same `c == z` aliasing trick as above.
1154|#[inline]
1155|fn write_lin_word_ab_packed(bit_off: usize, val: u32, z: &mut [u64], a: &mut [u64], b: &mut [u64]) {
1156|    or_u32_at_bit(z, bit_off, val);
1157|    or_u32_at_bit(a, bit_off, val);
1158|    or_u32_at_bit(b, bit_off, 0xFFFF_FFFF);
1159|}
1160|
1161|/// Sequential full-word writer for one packed block. Unlike the generic
1162|/// OR-based helpers, this never reads the destination and initializes every
1163|/// word, allowing the outer driver to skip its 1.5-GiB ranked zero pass.
1164|struct PackedWordWriter {
1165|    out: *mut u64,
1166|    word: usize,
1167|    pending: u64,
1168|    used: usize,
1169|}
1170|
1171|impl PackedWordWriter {
1172|    #[inline(always)]
1173|    fn at(out: *mut u64, word: usize, pending: u64, used: usize) -> Self {
1174|        Self {
1175|            out,
1176|            word,
1177|            pending,
1178|            used,
1179|        }
1180|    }
1181|
1182|    #[inline(always)]
1183|    fn push(&mut self, value: u64, width: usize) {
1184|        debug_assert!((1..=64).contains(&width));
1185|        let value = if width == 64 {
1186|            value
1187|        } else {
1188|            value & ((1u64 << width) - 1)
1189|        };
1190|        if self.used == 0 && width == 64 {
1191|            // SAFETY: the fixed BLAKE3 layout emits exactly `K / 64` words;
1192|            // the caller supplies a distinct block-sized destination.
1193|            unsafe {
1194|                self.out.add(self.word).write(value);
1195|            }
1196|            self.word += 1;
1197|            return;
1198|        }
1199|        let room = 64 - self.used;
1200|        if width < room {
1201|            self.pending |= value << self.used;
1202|            self.used += width;
1203|        } else {
1204|            // SAFETY: the fixed BLAKE3 layout emits exactly `K / 64` words;
1205|            // the caller supplies a distinct block-sized destination.
1206|            unsafe {
1207|                self.out
1208|                    .add(self.word)
1209|                    .write(self.pending | (value << self.used));
1210|            }
1211|            self.word += 1;
1212|            if width == room {
1213|                self.pending = 0;
1214|                self.used = 0;
1215|            } else {
1216|                self.pending = value >> room;
1217|                self.used = width - room;
1218|            }
1219|        }
1220|    }
1221|
1222|    #[inline(always)]
1223|    fn push_record<const N: usize>(&mut self, record: &BitRecord<N>, bits: usize) {
1224|        let mut left = bits;
1225|        for &value in record.words() {
1226|            if left == 0 {
1227|                break;
1228|            }
1229|            let width = left.min(64);
1230|            self.push(value, width);
1231|            left -= width;
1232|        }
1233|        debug_assert_eq!(left, 0);
1234|    }
1235|
1236|    #[inline(always)]
1237|    fn position(&self) -> usize {
1238|        self.word * 64 + self.used
1239|    }
1240|
1241|    #[inline]
1242|    fn finish(mut self, total_words: usize) {
1243|        if self.used != 0 {
1244|            // SAFETY: see `push`; a partial final word is still within the
1245|            // fixed-size block.
1246|            unsafe {
1247|                self.out.add(self.word).write(self.pending);
1248|            }
1249|            self.word += 1;
1250|        }
1251|        debug_assert!(self.word <= total_words);
1252|        // SAFETY: the unwritten suffix is within the same block-sized output.
1253|        unsafe {
1254|            std::ptr::write_bytes(self.out.add(self.word), 0, total_words - self.word);
1255|        }
1256|    }
1257|}
1258|
1259|#[inline(always)]
1260|fn stream_lin_word(
1261|    value: u32,
1262|    z: &mut PackedWordWriter,
1263|    a: &mut PackedWordWriter,
1264|    b: &mut PackedWordWriter,
1265|) {
1266|    z.push(value as u64, 32);
1267|    a.push(value as u64, 32);
1268|    b.push(u32::MAX as u64, 32);
1269|}
1270|
1271|/// Build the (z, a, b) blocks for ONE compression instance, into u64 views
1272|/// of the F128-packed per-block storage. Buffers must be zero on entry.
1273|///
1274|/// **No c buffer.** Since `C = I` (this is the circuit-shape R1CS), `c == z`
1275|/// byte-for-byte; callers use `z_packed` directly as the c-side input to
1276|/// zerocheck.
1277|fn build_block_witness_ab_packed_into(
1278|    cv: &[u32; 8],
1279|    m: &[u32; 16],
1280|    counter: u64,
1281|    block_len: u32,
1282|    flags: u32,
1283|    z: &mut [u64],
1284|    a: &mut [u64],
1285|    b: &mut [u64],
1286|) {
1287|    const U64_PER_BLOCK: usize = K / 64;
1288|    debug_assert_eq!(z.len(), U64_PER_BLOCK);
1289|    debug_assert_eq!(a.len(), U64_PER_BLOCK);
1290|    debug_assert_eq!(b.len(), U64_PER_BLOCK);
1291|
1292|    // Constant z[0] = 1; a/b also 1 (z[0]·z[0] = z[0]).
1293|    or_bit_at(z, Z_CONST_POS);
1294|    or_bit_at(a, Z_CONST_POS);
1295|    or_bit_at(b, Z_CONST_POS);
1296|
1297|    // Input rows.
1298|    let counter_lo = counter as u32;
1299|    let counter_hi = (counter >> 32) as u32;
1300|    for w in 0..8 {
1301|        write_lin_word_ab_packed(cv_bit(w, 0), cv[w], z, a, b);
1302|    }
1303|    for i in 0..16 {
1304|        write_lin_word_ab_packed(m_bit(i, 0), m[i], z, a, b);
1305|    }
1306|    write_lin_word_ab_packed(T_LO_BASE, counter_lo, z, a, b);
1307|    write_lin_word_ab_packed(T_HI_BASE, counter_hi, z, a, b);
1308|    write_lin_word_ab_packed(BLEN_BASE, block_len, z, a, b);
1309|    write_lin_word_ab_packed(FLAGS_BASE, flags, z, a, b);
1310|
1311|    // BLAKE3 state evolution.
1312|    let mut state: [u32; 16] = [
1313|        cv[0],
1314|        cv[1],
1315|        cv[2],
1316|        cv[3],
1317|        cv[4],
1318|        cv[5],
1319|        cv[6],
1320|        cv[7],
1321|        BLAKE3_IV[0],
1322|        BLAKE3_IV[1],
1323|        BLAKE3_IV[2],
1324|        BLAKE3_IV[3],
1325|        counter_lo,
1326|        counter_hi,
1327|        block_len,
1328|        flags,
1329|    ];
1330|    let msg_idx = &PER_ROUND_MSG_IDX;
1331|    for r in 0..N_ROUNDS {
1332|        for g_in_round in 0..N_G_PER_ROUND {
1333|            let g = r * N_G_PER_ROUND + g_in_round;
1334|            let [la, lb, lc, ld] = G_LANES[g_in_round];
1335|            let [mx_i, my_i] = msg_idx[r][g_in_round];
1336|            let mx = m[mx_i];
1337|            let my = m[my_i];
1338|
1339|            let a_val = state[la];
1340|            let b_val = state[lb];
1341|            let c_val = state[lc];
1342|            let d_val = state[ld];
1343|
1344|            let mut rz = BitRecord::<4>::new();
1345|            let mut ra = BitRecord::<4>::new();
1346|            let mut rb = BitRecord::<4>::new();
1347|
1348|            macro_rules! add_into {
1349|                ($pos:ident, $x:expr, $y:expr) => {{
1350|                    let (sum, left, right, carry) = add_carry_parts($x, $y);
1351|                    rz.push::<$pos>(carry);
1352|                    ra.push::<$pos>(left);
1353|                    rb.push::<$pos>(right);
1354|                    sum
1355|                }};
1356|            }
1357|
1358|            let tmp_0 = add_into!(REC_C0, a_val, b_val);
1359|            let a_1 = add_into!(REC_C1, tmp_0, mx);
1360|            let d_1 = (d_val ^ a_1).rotate_right(16);
1361|            let c_1 = add_into!(REC_C2, c_val, d_1);
1362|            let b_1 = (b_val ^ c_1).rotate_right(12);
1363|            let tmp_1 = add_into!(REC_C3, a_1, b_1);
1364|            let a_2 = add_into!(REC_C4, tmp_1, my);
1365|            let d_2 = (d_1 ^ a_2).rotate_right(8);
1366|            let c_2 = add_into!(REC_C5, c_1, d_2);
1367|            let b_new = (b_1 ^ c_2).rotate_right(7);
1368|            let d_new = d_2;
1369|            rz.push::<REC_LIN0>(b_new);
1370|            ra.push::<REC_LIN0>(b_new);
1371|            rb.push::<REC_LIN0>(0xFFFF_FFFF);
1372|            rz.push::<REC_LIN1>(d_new);
1373|            ra.push::<REC_LIN1>(d_new);
1374|            rb.push::<REC_LIN1>(0xFFFF_FFFF);
1375|
1376|            let g_base = GS_BASE + G_STRIDE * g;
1377|            rz.flush(z, g_base);
1378|            ra.flush(a, g_base);
1379|            rb.flush(b, g_base);
1380|
1381|            state[la] = a_2;
1382|            state[lb] = b_new;
1383|            state[lc] = c_2;
1384|            state[ld] = d_new;
1385|        }
1386|    }
1387|
1388|    // Finalization XOR rows.
1389|    for w in 0..8 {
1390|        let lo = state[w] ^ state[w + 8];
1391|        let hi = state[w + 8] ^ cv[w];
1392|        write_lin_word_ab_packed(out_lo_bit(w, 0), lo, z, a, b);
1393|        write_lin_word_ab_packed(out_hi_bit(w, 0), hi, z, a, b);
1394|    }
1395|}
1396|
1397|/// Full-write counterpart of [`build_block_witness_ab_packed_into`]. The
1398|/// circuit rows are contiguous through `USEFUL_BITS`, so three streaming bit
1399|/// writers can publish complete u64s without a destination read-modify-write.
1400|/// The only out-of-order region is the aligned `out_lo` slot, reserved while
1401|/// the compression runs and overwritten once the final state is known.
1402|fn build_block_witness_ab_stream_into(
1403|    cv: &[u32; 8],
1404|    m: &[u32; 16],
1405|    counter: u64,
1406|    block_len: u32,
1407|    flags: u32,
1408|    z: &mut [u64],
1409|    a: &mut [u64],
1410|    b: &mut [u64],
1411|) {
1412|    const U64_PER_BLOCK: usize = K / 64;
1413|    debug_assert_eq!(z.len(), U64_PER_BLOCK);
1414|    debug_assert_eq!(a.len(), U64_PER_BLOCK);
1415|    debug_assert_eq!(b.len(), U64_PER_BLOCK);
1416|
1417|    let counter_lo = counter as u32;
1418|    let counter_hi = (counter >> 32) as u32;
1419|
1420|    // Initialize the fixed 1,153-bit prefix directly. This leaves each writer
1421|    // at word 18 with exactly one pending bit, which makes the subsequent
1422|    // generated G sequence start from a compile-time-known packing phase.
1423|    let z_ptr = z.as_mut_ptr();
1424|    let a_ptr = a.as_mut_ptr();
1425|    let b_ptr = b.as_mut_ptr();
1426|    unsafe {
1427|        for i in 0..4 {
1428|            let value = (cv[2 * i] as u64) | ((cv[2 * i + 1] as u64) << 32);
1429|            z_ptr.add(i).write(value);
1430|            a_ptr.add(i).write(value);
1431|            b_ptr.add(i).write(u64::MAX);
1432|        }
1433|        std::ptr::write_bytes(z_ptr.add(4), 0, 4);
1434|        std::ptr::write_bytes(a_ptr.add(4), 0, 4);
1435|        std::ptr::write_bytes(b_ptr.add(4), 0, 4);
1436|
1437|        let values = [
1438|            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
1439|            m[14], m[15], counter_lo, counter_hi, block_len, flags,
1440|        ];
1441|        for i in 0..10 {
1442|            let low = if i == 0 {
1443|                1
1444|            } else {
1445|                (values[2 * i - 1] >> 31) as u64
1446|            };
1447|            let value = low | ((values[2 * i] as u64) << 1) | ((values[2 * i + 1] as u64) << 33);
1448|            z_ptr.add(8 + i).write(value);
1449|            a_ptr.add(8 + i).write(value);
1450|            b_ptr.add(8 + i).write(u64::MAX);
1451|        }
1452|    }
1453|    let pending = (flags >> 31) as u64;
1454|    let mut wz = PackedWordWriter::at(z_ptr, 18, pending, 1);
1455|    let mut wa = PackedWordWriter::at(a_ptr, 18, pending, 1);
1456|    let mut wb = PackedWordWriter::at(b_ptr, 18, 1, 1);
1457|    debug_assert_eq!(wz.position(), GS_BASE);
1458|
1459|    let mut state: [u32; 16] = [
1460|        cv[0],
1461|        cv[1],
1462|        cv[2],
1463|        cv[3],
1464|        cv[4],
1465|        cv[5],
1466|        cv[6],
1467|        cv[7],
1468|        BLAKE3_IV[0],
1469|        BLAKE3_IV[1],
1470|        BLAKE3_IV[2],
1471|        BLAKE3_IV[3],
1472|        counter_lo,
1473|        counter_hi,
1474|        block_len,
1475|        flags,
1476|    ];
1477|    // The circuit shape and message schedule are fixed. Expanding all 56 Gs
1478|    // gives LLVM literal state/message indices and exposes the complete
1479|    // dependency graph to register allocation. This is also the source-level
1480|    // model for a generated AArch64 kernel: allocation and Rayon stay in Rust,
1481|    // while only this fixed inner computation is specialized.
1482|    macro_rules! g {
1483|        ($la:literal, $lb:literal, $lc:literal, $ld:literal, $mx:literal, $my:literal) => {{
1484|            let mx = m[$mx];
1485|            let my = m[$my];
1486|            let a_val = state[$la];
1487|            let b_val = state[$lb];
1488|            let c_val = state[$lc];
1489|            let d_val = state[$ld];
1490|
1491|            let mut rz = BitRecord::<4>::new();
1492|            let mut ra = BitRecord::<4>::new();
1493|            let mut rb = BitRecord::<4>::new();
1494|
1495|            macro_rules! add_into_stream {
1496|                ($pos:ident, $x:expr, $y:expr) => {{
1497|                    let (sum, left, right, carry) = add_carry_parts($x, $y);
1498|                    rz.push::<$pos>(carry);
1499|                    ra.push::<$pos>(left);
1500|                    rb.push::<$pos>(right);
1501|                    sum
1502|                }};
1503|            }
1504|
1505|            let tmp_0 = add_into_stream!(REC_C0, a_val, b_val);
1506|            let a_1 = add_into_stream!(REC_C1, tmp_0, mx);
1507|            let d_1 = (d_val ^ a_1).rotate_right(16);
1508|            let c_1 = add_into_stream!(REC_C2, c_val, d_1);
1509|            let b_1 = (b_val ^ c_1).rotate_right(12);
1510|            let tmp_1 = add_into_stream!(REC_C3, a_1, b_1);
1511|            let a_2 = add_into_stream!(REC_C4, tmp_1, my);
1512|            let d_2 = (d_1 ^ a_2).rotate_right(8);
1513|            let c_2 = add_into_stream!(REC_C5, c_1, d_2);
1514|            let b_new = (b_1 ^ c_2).rotate_right(7);
1515|            let d_new = d_2;
1516|            rz.push::<REC_LIN0>(b_new);
1517|            ra.push::<REC_LIN0>(b_new);
1518|            rb.push::<REC_LIN0>(u32::MAX);
1519|            rz.push::<REC_LIN1>(d_new);
1520|            ra.push::<REC_LIN1>(d_new);
1521|            rb.push::<REC_LIN1>(u32::MAX);
1522|
1523|            wz.push_record(&rz, G_STRIDE);
1524|            wa.push_record(&ra, G_STRIDE);
1525|            wb.push_record(&rb, G_STRIDE);
1526|
1527|            state[$la] = a_2;
1528|            state[$lb] = b_new;
1529|            state[$lc] = c_2;
1530|            state[$ld] = d_new;
1531|        }};
1532|    }
1533|    macro_rules! round {
1534|        ($m0:literal, $m1:literal, $m2:literal, $m3:literal,
1535|         $m4:literal, $m5:literal, $m6:literal, $m7:literal,
1536|         $m8:literal, $m9:literal, $m10:literal, $m11:literal,
1537|         $m12:literal, $m13:literal, $m14:literal, $m15:literal) => {{
1538|            g!(0, 4, 8, 12, $m0, $m1);
1539|            g!(1, 5, 9, 13, $m2, $m3);
1540|            g!(2, 6, 10, 14, $m4, $m5);
1541|            g!(3, 7, 11, 15, $m6, $m7);
1542|            g!(0, 5, 10, 15, $m8, $m9);
1543|            g!(1, 6, 11, 12, $m10, $m11);
1544|            g!(2, 7, 8, 13, $m12, $m13);
1545|            g!(3, 4, 9, 14, $m14, $m15);
1546|        }};
1547|    }
1548|    round!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
1549|    round!(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
1550|    round!(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
1551|    round!(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
1552|    round!(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
1553|    round!(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
1554|    round!(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);
1555|    debug_assert_eq!(wz.position(), OUT_HI_BASE);
1556|
1557|    let out_lo: [u32; 8] = std::array::from_fn(|w| state[w] ^ state[w + 8]);
1558|    for w in 0..8 {
1559|        stream_lin_word(state[w + 8] ^ cv[w], &mut wz, &mut wa, &mut wb);
1560|    }
1561|    debug_assert_eq!(wz.position(), USEFUL_BITS);
1562|
1563|    wz.finish(U64_PER_BLOCK);
1564|    wa.finish(U64_PER_BLOCK);
1565|    wb.finish(U64_PER_BLOCK);
1566|
1567|    // OUT_LO_BASE is 256-bit aligned, so the four reserved words can be
1568|    // replaced without touching neighboring rows.
1569|    const OUT_LO_WORD: usize = OUT_LO_BASE / 64;
1570|    debug_assert_eq!(OUT_LO_BASE % 64, 0);
1571|    for i in 0..4 {
1572|        let value = (out_lo[2 * i] as u64) | ((out_lo[2 * i + 1] as u64) << 32);
1573|        z[OUT_LO_WORD + i] = value;
1574|        a[OUT_LO_WORD + i] = value;
1575|        b[OUT_LO_WORD + i] = u64::MAX;
1576|    }
1577|}
1578|
1579|// ---------------------------------------------------------------------------
1580|// W-H2: SIMD-lockstep witness materialization (aarch64). Derivation and
1581|// pricing: notes/witgen-simd.md. Four compressions run in u32-lane lockstep
1582|// ("quad"); the row-major output is produced by a fixed 4x4 u32 register
1583|// transpose at the store point. Bit-exact with
1584|// [`build_block_witness_ab_stream_into`]: `vaddq_u32` wraps mod 2^32 per lane
1585|// (no arithmetic wider than u32 exists, so carries never cross lanes),
1586|// rotate-XOR is shr/shl/or, and the bit packing is a const-shift sequential
1587|// push network mirroring `PackedWordWriter`'s algebra lane-wise.
1588|// Kill switch: `FLOCK_NO_WITGEN_SIMD=1` restores the scalar driver.
1589|// `FLOCK_WITGEN_SIMD_PLAIN_STORES=1` replaces every z/a/b NT drain with plain
1590|// stores (same-binary store-flavor A/B). `FLOCK_NO_WITGEN_Z_NT=1` disables
1591|// only z's deferred-stream NT drain while preserving the incumbent a/b mode.
1592|// ---------------------------------------------------------------------------
1593|
1594|#[cfg(target_arch = "aarch64")]
1595|pub(crate) mod witgen_simd {
1596|    use super::{
1597|        BLAKE3_IV, Compression, G_STRIDE, GS_BASE, K, N_G, OUT_HI_BASE, REC_C0, REC_C1, REC_C2,
1598|        REC_C3, REC_C4, REC_C5, REC_LIN0, REC_LIN1, USEFUL_BITS,
1599|    };
1600|    use core::arch::aarch64::*;
1601|    use flock_core::bits::transpose_8_u64s_to_64_bytes;
1602|    use flock_core::field::F128;
1603|    use std::sync::LazyLock;
1604|
1605|    use crate::seed_pipe::CompressionQuadSoa;
1606|
1607|    const U32_PER_BLOCK: usize = K / 32; // 512
1608|    const F128_PER_BLOCK: usize = K / 128;
1609|    /// [`dump`] drains a block in 64 chunks of 8 u32 words (32 bytes).
1610|    const DUMP_CHUNKS: usize = U32_PER_BLOCK / 8; // 64
1611|
1612|    // -----------------------------------------------------------------------
1613|    // Recycled-scratch constant-region elision (witgen-stack item B).
1614|    //
1615|    // z/a/b come from the recycling scratch pool. At this fixed layout the
1616|    // builder rewrites the same per-block constants every prove: the zero
1617|    // fill (u32 words 482..512 of every block, all three buffers), b's MAX
1618|    // prefix (words 0..36), and b's fixed final lin/output/padding suffix.
1619|    // When the pool proves — via a provenance
1620|    // token attached at the previous release and dropped by any other
1621|    // custody event — that the handed-out allocation still holds exactly a
1622|    // previous prove's output of this same layout, those regions already
1623|    // contain the right bytes and their dump chunks are skipped. Skips are
1624|    // dump-chunk (32 B/block) granular and stay strictly INSIDE the
1625|    // constant regions: z/a's zero tail skips words 488..512 (chunk 60 still
1626|    // carries data words 480/481 and is always written), while b can skip
1627|    // from word 472 because its remaining lin-id/output bits are fixed ones
1628|    // before the zero padding. b's prefix skips words 0..32 (chunk 4 carries
1629|    // data words 36..39 and the residual constant words 32..35, always
1630|    // written).
1631|    //
1632|    // The constants are content-independent — every completed witgen of
1633|    // this layout writes identical bytes there (padding blocks included) —
1634|    // so a token hit only ever elides rewriting bytes with themselves.
1635|    // `FLOCK_NO_SCRATCH_CONST_ELIDE=1` (exact) restores plain takes and
1636|    // full incumbent writes; any token miss independently falls back to
1637|    // full writes for that buffer.
1638|    // -----------------------------------------------------------------------
1639|
1640|    /// First skippable chunk of the zero tail: words 488..512.
1641|    const ELIDE_ZERO_CHUNK: usize = 61;
1642|    /// First skippable b suffix chunk: words 472..512.
1643|    const ELIDE_B_TAIL_CHUNK: usize = 59;
1644|    /// Leading skippable chunks of b's MAX prefix: words 0..32.
1645|    const ELIDE_B_PREFIX_CHUNKS: usize = 4;
1646|    const BLOCK_BYTES: usize = U32_PER_BLOCK * 4; // 2048
1647|    const ZERO_TAIL_BYTE: usize = ELIDE_ZERO_CHUNK * 32; // 1952
1648|    const B_TAIL_BYTE: usize = ELIDE_B_TAIL_CHUNK * 32; // 1888
1649|    const B_FULL_ONES_END_BYTE: usize = USEFUL_BITS / 8; // 1926
1650|    const B_LAST_BYTE_VALUE: u8 = (1u8 << (USEFUL_BITS % 8)) - 1; // 0x01
1651|    const B_ZERO_START_BYTE: usize = USEFUL_BITS.div_ceil(8); // 1927
1652|    const B_PREFIX_BYTES: usize = ELIDE_B_PREFIX_CHUNKS * 32; // 128
1653|    const _ELIDE_GEOMETRY: () = {
1654|        // Skipped zero-tail words start at or after the zero fill's first
1655|        // word (USEFUL_BITS.div_ceil(32) = 482)...
1656|        assert!(8 * ELIDE_ZERO_CHUNK >= USEFUL_BITS.div_ceil(32));
1657|        assert!(8 * ELIDE_ZERO_CHUNK < U32_PER_BLOCK);
1658|        // The final G's two B-side lin-id rows and every B-side out_hi row are
1659|        // ones, so the chunk-aligned B suffix begins inside that fixed run.
1660|        let b_fixed_one_start = GS_BASE + (N_G - 1) * G_STRIDE + REC_LIN0;
1661|        assert!(256 * (ELIDE_B_TAIL_CHUNK - 1) < b_fixed_one_start);
1662|        assert!(256 * ELIDE_B_TAIL_CHUNK >= b_fixed_one_start);
1663|        assert!(256 * ELIDE_B_TAIL_CHUNK < USEFUL_BITS);
1664|        assert!(USEFUL_BITS % 8 == 1);
1665|        assert!(B_ZERO_START_BYTE <= ZERO_TAIL_BYTE);
1666|        // ...and skipped b-prefix words end at or before the MAX prefix's
1667|        // last word (36).
1668|        assert!(8 * ELIDE_B_PREFIX_CHUNKS <= 36);
1669|    };
1670|
1671|    /// Provenance-tag layout version: bump on ANY change to the witness
1672|    /// block layout or to the elision geometry above.
1673|    const WITGEN_SCRATCH_LAYOUT_V: u64 = 2;
1674|    pub(crate) const ROLE_Z: u64 = 1;
1675|    pub(crate) const ROLE_A: u64 = 2;
1676|    pub(crate) const ROLE_B: u64 = 3;
1677|
1678|    /// Scratch provenance tag: magic | role | layout version | K_LOG |
1679|    /// USEFUL_BITS | n_blocks_log. Combined with the pool's exact-length
1680|    /// check this uniquely names "witness buffer `role` of the ranked
1681|    /// BLAKE3 witgen layout at this size".
1682|    pub(crate) fn witgen_scratch_tag(role: u64, n_blocks_log: usize) -> u64 {
1683|        (0x57u64 << 56)
1684|            | (role << 48)
1685|            | (WITGEN_SCRATCH_LAYOUT_V << 40)
1686|            | ((super::K_LOG as u64) << 32)
1687|            | ((USEFUL_BITS as u64) << 16)
1688|            | (n_blocks_log as u64)
1689|    }
1690|
1691|    /// Exact-`1` kill switch for the constant-region elision (item B).
1692|    /// Read per witgen call (uncached) so same-process A/B tests can
1693|    /// toggle it.
1694|    fn const_elide_killed() -> bool {
1695|        std::env::var("FLOCK_NO_SCRATCH_CONST_ELIDE").is_ok_and(|v| v == "1")
1696|    }
1697|
1698|    /// Bitmask of token hits (bit0 z, bit1 a, bit2 b) of the most recent
1699|    /// `generate_impl` call — release-canary probe.
1700|    static WITGEN_ELIDE_HITS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
1701|
1702|    #[cfg(test)]
1703|    pub(crate) fn last_elide_hits() -> u8 {
1704|        WITGEN_ELIDE_HITS.load(std::sync::atomic::Ordering::Relaxed)
1705|    }
1706|
1707|    /// Constant-region probe set for a staged release: sampled blocks'
1708|    /// zero tails (and, for b, MAX prefixes) that `give_f128` re-verifies
1709|    /// before attaching the token.
1710|    fn elide_probes(n_total: usize, b_flavor: bool) -> Vec<flock_core::scratch::ReleaseProbe> {
1711|        use flock_core::scratch::ReleaseProbe;
1712|        let mut blocks = [0, 1, n_total / 2, n_total - 2, n_total - 1];
1713|        blocks.sort_unstable();
1714|        let mut probes = Vec::with_capacity(if b_flavor { 4 } else { 1 } * blocks.len());
1715|        let mut last = usize::MAX;
1716|        for &blk in &blocks {
1717|            if blk == last {
1718|                continue;
1719|            }
1720|            last = blk;
1721|            if b_flavor {
1722|                probes.push(ReleaseProbe {
1723|                    byte_off: blk * BLOCK_BYTES,
1724|                    len: B_PREFIX_BYTES,
1725|                    value: 0xFF,
1726|                });
1727|                probes.push(ReleaseProbe {
1728|                    byte_off: blk * BLOCK_BYTES + B_TAIL_BYTE,
1729|                    len: B_FULL_ONES_END_BYTE - B_TAIL_BYTE,
1730|                    value: 0xFF,
1731|                });
1732|                probes.push(ReleaseProbe {
1733|                    byte_off: blk * BLOCK_BYTES + B_FULL_ONES_END_BYTE,
1734|                    len: 1,
1735|                    value: B_LAST_BYTE_VALUE,
1736|                });
1737|                probes.push(ReleaseProbe {
1738|                    byte_off: blk * BLOCK_BYTES + B_ZERO_START_BYTE,
1739|                    len: BLOCK_BYTES - B_ZERO_START_BYTE,
1740|                    value: 0x00,
1741|                });
1742|            } else {
1743|                probes.push(ReleaseProbe {
1744|                    byte_off: blk * BLOCK_BYTES + ZERO_TAIL_BYTE,
1745|                    len: BLOCK_BYTES - ZERO_TAIL_BYTE,
1746|                    value: 0x00,
1747|                });
1748|            }
1749|        }
1750|        probes
1751|    }
1752|
1753|    /// Debug-build gate (i): before a group's dump runs with elision, its
1754|    /// destination ranges that will be SKIPPED must already hold the
1755|    /// constants the builder would have written. A failure here means the
1756|    /// provenance token vouched for bytes that are not there — a
1757|    /// silently-wrong witness in release — so this asserts, not warns.
1758|    #[cfg(debug_assertions)]
1759|    fn debug_verify_elided_group(z: &[F128], a: &[F128], b: &[F128], elide: [bool; 3]) {
1760|        let bytes = |v: &[F128]| unsafe {
1761|            core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), core::mem::size_of_val(v))
1762|        };
1763|        for blk in 0..8 {
1764|            let block = blk * BLOCK_BYTES;
1765|            for (i, (buf, on)) in [(z, elide[0]), (a, elide[1]), (b, elide[2])]
1766|                .into_iter()
1767|                .enumerate()
1768|            {
1769|                if !on {
1770|                    continue;
1771|                }
1772|                let zero_start = if i == 2 {
1773|                    B_ZERO_START_BYTE
1774|                } else {
1775|                    ZERO_TAIL_BYTE
1776|                };
1777|                assert!(
1778|                    bytes(buf)[block + zero_start..block + BLOCK_BYTES]
1779|                        .iter()
1780|                        .all(|&x| x == 0),
1781|                    "elide zero-tail mismatch buf={i} blk={blk}"
1782|                );
1783|            }
1784|            if elide[2] {
1785|                let prefix = block..block + B_PREFIX_BYTES;
1786|                assert!(
1787|                    bytes(b)[prefix].iter().all(|&x| x == 0xFF),
1788|                    "elide b-prefix mismatch blk={blk}"
1789|                );
1790|                assert!(
1791|                    bytes(b)[block + B_TAIL_BYTE..block + B_FULL_ONES_END_BYTE]
1792|                        .iter()
1793|                        .all(|&x| x == 0xFF),
1794|                    "elide b-one-tail mismatch blk={blk}"
1795|                );
1796|                assert_eq!(
1797|                    bytes(b)[block + B_FULL_ONES_END_BYTE],
1798|                    B_LAST_BYTE_VALUE,
1799|                    "elide b-last-byte mismatch blk={blk}"
1800|                );
1801|            }
1802|        }
1803|    }
1804|
1805|    pub(crate) fn enabled() -> bool {
1806|        static ON: LazyLock<bool> =
1807|            LazyLock::new(|| std::env::var_os("FLOCK_NO_WITGEN_SIMD").is_none());
1808|        *ON
1809|    }
1810|
1811|    fn nt_enabled() -> bool {
1812|        // Global same-binary kill switch for all SIMD NT drain stores.
1813|        static NT: LazyLock<bool> =
1814|            LazyLock::new(|| std::env::var_os("FLOCK_WITGEN_SIMD_PLAIN_STORES").is_none());
1815|        *NT
1816|    }
1817|
1818|    fn z_nt_enabled() -> bool {
1819|        static ON: LazyLock<bool> =
1820|            LazyLock::new(|| std::env::var_os("FLOCK_NO_WITGEN_Z_NT").is_none());
1821|        *ON
1822|    }
1823|
1824|    #[inline(always)]
1825|    pub(super) const fn select_z_nt(
1826|        nt_enabled: bool,
1827|        defer_ranked_stripe: bool,
1828|        z_nt_enabled: bool,
1829|    ) -> bool {
1830|        nt_enabled && defer_ranked_stripe && z_nt_enabled
1831|    }
1832|
1833|    type V4 = uint32x4_t;
1834|
1835|    pub(crate) enum QuadInput<'a> {
1836|        Blocks([&'a Compression; 4]),
1837|        Seeded(&'a CompressionQuadSoa),
1838|    }
1839|
1840|    /// Fixed 4x4 u32 transpose. Both orientations use the same network:
1841|    /// (word w across 4 blocks) <-> (block j's 4 consecutive words). Pure
1842|    /// data movement — exact.
1843|    #[inline(always)]
1844|    fn tr4(w0: V4, w1: V4, w2: V4, w3: V4) -> (V4, V4, V4, V4) {
1845|        unsafe {
1846|            let t0 = vtrn1q_u32(w0, w1);
1847|            let t1 = vtrn2q_u32(w0, w1);
1848|            let t2 = vtrn1q_u32(w2, w3);
1849|            let t3 = vtrn2q_u32(w2, w3);
1850|            (
1851|                vreinterpretq_u32_u64(vtrn1q_u64(
1852|                    vreinterpretq_u64_u32(t0),
1853|                    vreinterpretq_u64_u32(t2),
1854|                )),
1855|                vreinterpretq_u32_u64(vtrn1q_u64(
1856|                    vreinterpretq_u64_u32(t1),
1857|                    vreinterpretq_u64_u32(t3),
1858|                )),
1859|                vreinterpretq_u32_u64(vtrn2q_u64(
1860|                    vreinterpretq_u64_u32(t0),
1861|                    vreinterpretq_u64_u32(t2),
1862|                )),
1863|                vreinterpretq_u32_u64(vtrn2q_u64(
1864|                    vreinterpretq_u64_u32(t1),
1865|                    vreinterpretq_u64_u32(t3),
1866|                )),
1867|            )
1868|        }
1869|    }
1870|
1871|    /// NT 32-byte store pair (a/b pass the failed.md §14 never-read test:
1872|    /// their next readers are a proof later, from DRAM).
1873|    #[inline(always)]
1874|    unsafe fn store_nt_pair(x: V4, y: V4, p: *mut u32) {
1875|        unsafe {
1876|            core::arch::asm!(
1877|                "stnp {0:q}, {1:q}, [{2}]",
1878|                in(vreg) x,
1879|                in(vreg) y,
1880|                in(reg) p,
1881|                options(nostack)
1882|            );
1883|        }
1884|    }
1885|
1886|    /// Last useful word (bit 15408 → word 481, 17 bits used).
1887|    const LAST_WORD: usize = (USEFUL_BITS - 1) / 32; // 481
1888|
1889|    /// NT 64-byte stripe chunk store (via an L1 stack bounce): the lincheck
1890|    /// stripe passes the failed.md §14 never-read test (read ~85 ms later,
1891|    /// 512 MiB ≫ SLC), so it stores non-temporally like a/b.
1892|    #[inline(always)]
1893|    unsafe fn stripe_store_nt(src: *const u8, dst: *mut u8) {
1894|        unsafe {
1895|            core::arch::asm!(
1896|                "ldp {t0:q}, {t1:q}, [{s}]",
1897|                "stnp {t0:q}, {t1:q}, [{d}]",
1898|                "ldp {t0:q}, {t1:q}, [{s}, #32]",
1899|                "stnp {t0:q}, {t1:q}, [{d}, #32]",
1900|                s = in(reg) src,
1901|                d = in(reg) dst,
1902|                t0 = out(vreg) _,
1903|                t1 = out(vreg) _,
1904|                options(nostack)
1905|            );
1906|        }
1907|    }
1908|
1909|    /// u32-granular lane-wise `PackedWordWriter`: `pending` plus the
1910|    /// absolute-word L1 stage. Every push site is monomorphized with its
1911|    /// stream offset (USED), the straddle back-shift (BACK), and — when it
1912|    /// completes a word — the ABSOLUTE word index (WORD), so completed words
1913|    /// go straight to the stage with immediate store offsets. There is no
1914|    /// runtime writer state besides `pending` — the vector analogue of the
1915|    /// scalar builder's fully-unrolled writer.
1916|    struct W32 {
1917|        pending: V4,
1918|        stage: *mut V4, // 512 block-lane words for this buffer's quad
1919|    }
1920|
1921|    impl W32 {
1922|        #[inline(always)]
1923|        fn at(stage: *mut V4, pending: V4) -> Self {
1924|            Self { pending, stage }
1925|        }
1926|
1927|        /// Push the low WIDTH bits of `v` at stream offset ≡ USED (mod 32).
1928|        /// WIDTH ∈ {31, 32}. Carry values deliberately retain an arbitrary
1929|        /// bit 31: `vsli` preserves only the already-final low `USED` bits and
1930|        /// overwrites every following bit with the new field, so the dirty bit
1931|        /// just above a 31-bit field is overwritten by the next push instead
1932|        /// of requiring an eager mask. The fixed stream ends in full-width
1933|        /// lin-id fields, hence no dirty carry bit can reach `finish`.
1934|        ///
1935|        /// BACK is the straddle back-shift `room = 32 − USED`; WORD is the
1936|        /// absolute index of the completed word (iff this push completes one).
1937|        /// All consts are spelled out at the call site (stable Rust cannot
1938|        /// derive const arguments from const parameters).
1939|        #[inline(always)]
1940|        unsafe fn push<const USED: i32, const WIDTH: i32, const BACK: i32, const WORD: usize>(
1941|            &mut self,
1942|            v: V4,
1943|        ) {
1944|            const {
1945|                assert!(USED >= 0 && USED < 32);
1946|                assert!(WIDTH == 31 || WIDTH == 32);
1947|                assert!(BACK >= 1 && BACK < 32);
1948|                assert!(WORD < U32_PER_BLOCK);
1949|            }
1950|            debug_assert!(USED + WIDTH <= 32 || BACK == 32 - USED);
1951|            unsafe {
1952|                // The USED == 0 arm avoids instantiating `vsliq_n::<0>`
1953|                // (illegal immediate) — no insert is needed at word-aligned
1954|                // positions. A width-31 value may leave bit 31 dirty here;
1955|                // the next `vsli #31` overwrites it exactly.
1956|                if USED == 0 {
1957|                    if WIDTH == 32 {
1958|                        vst1q_u32(self.stage.add(WORD) as *mut u32, v);
1959|                        self.pending = vdupq_n_u32(0);
1960|                    } else {
1961|                        self.pending = v;
1962|                    }
1963|                } else if USED + WIDTH < 32 {
1964|                    self.pending = vsliq_n_u32::<USED>(self.pending, v);
1965|                } else {
1966|                    let out = vsliq_n_u32::<USED>(self.pending, v);
1967|                    vst1q_u32(self.stage.add(WORD) as *mut u32, out);
1968|                    if USED + WIDTH == 32 {
1969|                        self.pending = vdupq_n_u32(0);
1970|                    } else {
1971|                        self.pending = vshrq_n_u32::<BACK>(v);
1972|                    }
1973|                }
1974|            }
1975|        }
1976|
1977|        /// `PackedWordWriter::finish` semantics: the partial final word 481
1978|        /// (upper bits zero by construction) joins the stage.
1979|        #[inline(always)]
1980|        unsafe fn finish(&mut self) {
1981|            unsafe {
1982|                vst1q_u32(self.stage.add(LAST_WORD) as *mut u32, self.pending);
1983|            }
1984|        }
1985|    }
1986|
1987|    /// Drain a 512-word block-lane stage to the four row-major block
1988|    /// destinations. `ld4` deinterleaves four block-lane words into
1989|    /// per-block 16-B runs (the register transpose the batch-major layout
1990|    /// dodged), so each block's 2 KiB drains as ONE long ascending burst:
1991|    /// stnp pairs for the §14-passing buffers (a/b), plain stores for z
1992|    /// (§16 in-closure stripe re-read). Drains dump-chunk range `g0..g1`
1993|    /// only (a dump chunk `g` covers u32 words `8g..8g+8` of every block in
1994|    /// the quad — 32 bytes per block; the full block is `0..DUMP_CHUNKS`).
1995|    /// The recycled-scratch constant-region elision narrows the range to
1996|    /// skip chunks whose destination bytes are token-verified to already
1997|    /// hold the per-block constants the builder would rewrite.
1998|    #[inline(always)]
1999|    unsafe fn dump_range<const NT: bool>(stage: *const V4, dst: *mut u32, g0: usize, g1: usize) {
2000|        unsafe {
2001|            for g in g0..g1 {
2002|                let w = 8 * g;
2003|                let x = vld4q_u32(stage.add(w) as *const u32);
2004|                let y = vld4q_u32(stage.add(w + 4) as *const u32);
2005|                let p0 = dst.add(w);
2006|                let p1 = dst.add(U32_PER_BLOCK + w);
2007|                let p2 = dst.add(2 * U32_PER_BLOCK + w);
2008|                let p3 = dst.add(3 * U32_PER_BLOCK + w);
2009|                if NT {
2010|                    store_nt_pair(x.0, y.0, p0);
2011|                    store_nt_pair(x.1, y.1, p1);
2012|                    store_nt_pair(x.2, y.2, p2);
2013|                    store_nt_pair(x.3, y.3, p3);
2014|                } else {
2015|                    vst1q_u32(p0, x.0);
2016|                    vst1q_u32(p0.add(4), y.0);
2017|                    vst1q_u32(p1, x.1);
2018|                    vst1q_u32(p1.add(4), y.1);
2019|                    vst1q_u32(p2, x.2);
2020|                    vst1q_u32(p2.add(4), y.2);
2021|                    vst1q_u32(p3, x.3);
2022|                    vst1q_u32(p3.add(4), y.3);
2023|                }
2024|            }
2025|        }
2026|    }
2027|
2028|    /// Stream-sequential field push at absolute bit position `$pos`: computes
2029|    /// all four monomorphization consts at the call site. BACK is the
2030|    /// straddle back-shift `room = 32 − USED` (clamped to the legal immediate
2031|    /// range for the dead-branch instantiation); WORD = `pos/32` is the
2032|    /// completed word's absolute index.
2033|    macro_rules! pushf {
2034|        ($w:ident, $pos:expr, $width:literal, $v:expr) => {{
2035|            $w.push::<{ ($pos % 32) as i32 }, $width, {
2036|                let u = ($pos % 32) as i32;
2037|                if u == 0 { 1 } else { 32 - u }
2038|            }, { $pos / 32 }>($v);
2039|        }};
2040|    }
2041|
2042|    /// Lane-wise `add_carry_parts`: `(sum, left, right, carry_aux)`.
2043|    /// `vaddq_u32` wraps mod 2^32 per lane — bit-identical to scalar
2044|    /// `wrapping_add` for each independent block; carries never cross lanes.
2045|    /// The three row values retain their irrelevant bit 31. [`W32::push`]
2046|    /// consumes only the low 31 bits and overwrites that dirty boundary bit,
2047|    /// removing two vector masks from every one of the 336 additions.
2048|    #[inline(always)]
2049|    fn add_carry_parts_v(x: V4, y: V4) -> (V4, V4, V4, V4) {
2050|        unsafe {
2051|            let sum = vaddq_u32(x, y);
2052|            let cin = veorq_u32(veorq_u32(sum, x), y);
2053|            let left = veorq_u32(x, cin);
2054|            let right = veorq_u32(y, cin);
2055|            let carry = vandq_u32(left, right);
2056|            (sum, left, right, carry)
2057|        }
2058|    }
2059|
2060|    /// `(x ^ y).rotate_right(N)` — NEON has no vector ROR; shr/shl/or is
2061|    /// exact bitwise. M = 32 − N is spelled out at the call site (stable
2062|    /// Rust cannot derive const arguments from const parameters).
2063|    #[inline(always)]
2064|    fn xor_rotr<const N: i32, const M: i32>(x: V4, y: V4) -> V4 {
2065|        debug_assert_eq!(N + M, 32);
2066|        unsafe {
2067|            let v = veorq_u32(x, y);
2068|            vorrq_u32(vshrq_n_u32::<N>(v), vshlq_n_u32::<M>(v))
2069|        }
2070|    }
2071|
2072|    /// Build the (z, a, b) blocks for FOUR compressions in u32-lane lockstep,
2073|    /// fully writing every word (stale scratch). `z`/`a`/`b` point at the
2074|    /// quad's first block; block j occupies `dst + j*512 .. +512` u32 words.
2075|    /// `z_nt` and `ab_nt` independently select non-temporal drain stores for
2076|    /// z and for the a/b pair, respectively.
2077|    /// Bit-exact with [`super::build_block_witness_ab_stream_into`] x4.
2078|    #[cfg_attr(not(test), allow(dead_code))]
2079|    pub(crate) unsafe fn build_quad_witness_ab_stream_neon(
2080|        inputs: [&Compression; 4],
2081|        z: *mut u32,
2082|        a: *mut u32,
2083|        b: *mut u32,
2084|        z_nt: bool,
2085|        ab_nt: bool,
2086|    ) {
2087|        unsafe {
2088|            build_quad_witness_ab_stream_neon_elide(
2089|                QuadInput::Blocks(inputs),
2090|                z,
2091|                a,
2092|                b,
2093|                z_nt,
2094|                ab_nt,
2095|                [false; 3],
2096|            )
2097|        }
2098|    }
2099|
2100|