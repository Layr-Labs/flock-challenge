//! Lincheck PIOP for **block-diagonal** R1CS over GF(2).
//!
//! Reduces three MLE evaluation claims (`â(x)=v`, `b̂(x')=v'`, `ĉ(x'')=v''`)
//! plus the linear constraints (`a = Az`, `b = Bz`, `c = Cz`) to three MLE
//! evaluation claims on `z`, all sharing a fresh random inner coord.
//!
//! ## Matrix structure (the assumption we exploit)
//!
//! `A = I_{2^n_log} ⊗ A_0` (block-diagonal with `A_0` repeated `2^n_log`
//! times along the diagonal). Same for B, C. Storage is `O(k²)` for the
//! small base matrices, not `O(N²)`.
//!
//! With the row/col index decomposed as `(i_inner, i_outer)` with `k_log`
//! inner bits and `n_log` outer bits (`m = k_log + n_log`), the bilinear MLE
//! factors:
//!
//!   `Â(i, x)  =  Â_0(i_inner, x_inner) · eq(i_outer, x_outer)`
//!
//! So for the claim `v = â(x) = Σ_i z(i) · Â(i, x)` the outer summation
//! collapses by the eq-MLE identity:
//!
//!   `v  =  Σ_{i_inner}  Â_0(i_inner, x_inner) · ẑ(i_inner, x_outer)`
//!
//! — a sum over only `2^k_log` terms, with `ẑ(·, x_outer)` being the
//! partial fold of `z` at the outer half of the claim point.
//!
//! ## Protocol shape (circuit R1CS: C = I, A & B share a claim point)
//!
//! For R1CS coming from circuits, `C = I` (identity), so `c = Cz = z` and
//! the zerocheck's c-claim `ĉ(point_c) = v_c` IS a direct `z`-claim
//! `ẑ(point_c) = v_c` — handled by the PCS without going through lincheck.
//! Likewise the zerocheck's `â` and `b̂` claims live at the **same** point
//! `(z, ρ-values)`, so lincheck only needs to fold `z` **once** at that
//! shared point.
//!
//! 1. **Prover sends** one length-`k = 2^k_log` F128 vector
//!    `z_vec[i_inner] = ẑ(i_inner, x_ab.x_outer)`.
//! 2. **Verifier checks** *two* consistency equations against the same
//!    `z_vec`:
//!    ```text
//!    Σ_{i_inner}  Â_0_quirky(z_skip, x_inner_rest, i_inner) · z_vec[i_inner]  ==  v_a
//!    Σ_{i_inner}  B̂_0_quirky(z_skip, x_inner_rest, i_inner) · z_vec[i_inner]  ==  v_b
//!    ```
//! 3. **Verifier samples** quirky `(r_inner_skip, r_inner_rest)` after
//!    observing `z_vec`.
//! 4. **Verifier derives** one z-claim at the shared output point:
//!    ```text
//!    w = ẑ((r_inner_skip, r_inner_rest), x_ab.x_outer)
//!      = Σ_{i_inner} quirky_eq(r_inner_skip, r_inner_rest, i_inner) · z_vec[i_inner]
//!    ```
//!
//! The lincheck output is one `(point, value)` z-claim; combined with the
//! c-claim handed in directly by the caller, the PCS sees **two** z-openings.
//!
//! ## Soundness
//!
//! - The two scalar checks tie `z_vec` to `v_a` and `v_b` from the upstream
//!   layer — without them a malicious prover could send any vector.
//! - The post-vector random `(r_inner_skip, r_inner_rest)` plus Schwartz-Zippel
//!   ensures that if `z_vec_claimed` differs from the true partial fold of `z`,
//!   the derived `w` differs from the true `ẑ((r_inner_skip, r_inner_rest), x_outer)`
//!   with probability `≈ 1 − 2⁻¹²⁸`. The PCS opening catches that downstream.
//!
//! ## Quirky (univariate-skip) claim points
//!
//! To compose with the **zerocheck's univariate skip** for the first `k_skip`
//! variables, claim points use the [`QuirkyPoint`] representation:
//!
//!   `x = (z_skip ∈ F_{2^128},  x_inner_rest ∈ F_{2^128}^{k_log − k_skip},  x_outer ∈ F_{2^128}^{n_log})`
//!
//! - `z_skip` is the univariate-skip challenge; it represents all `k_skip`
//!   skip variables collapsed via the polynomial extension with Lagrange
//!   basis on `φ_8(0), …, φ_8(2^{k_skip} − 1)`.
//! - The remaining `k_log − k_skip` inner coords plus the `n_log` outer
//!   coords are standard multilinear.
//!
//! When evaluating the bilinear matrix MLE at a quirky claim point, the
//! eq factor for the inner row index becomes the **outer product of**:
//! `L_{i_skip}(z_skip) · eq(x_inner_rest, i_inner_rest)`, where `L_*` are
//! Lagrange weights at `z_skip` for the `k_skip` skip dims (see
//! [`build_quirky_eq_table`]).
//!
//! The prover's partial fold `ẑ(·, x_outer)` is unchanged — it only depends
//! on `x_outer` (still pure multilinear). The verifier-side eq tables and
//! the final-sample reduction are the only changes.
//!
//! ## Conventions
//!
//! - **Point ordering inside `QuirkyPoint`.** `x_inner_rest[0..k_log − k_skip]`
//!   bind to inner variables `i_inner_rest[0..k_log − k_skip]`. `x_outer[0..n_log]`
//!   to outer vars.
//! - **Eq table layout.** `eq_table[i]` where `i = Σ b_j · 2^j` is
//!   `Π_j eq(point[j], b_j) = Π_j (1 + point[j] + b_j)`.
//! - **`z_packed` byte layout (specific to lincheck — enables column-scan
//!   lookup tables without an explicit transpose).** Writing `i_outer = 8·byte_idx + r`
//!   with `r ∈ {0,..,7}` and `byte_idx ∈ {0,..,n_outer/8 − 1}`, the bit
//!   `z[i_inner, i_outer]` lives at:
//!     - **byte position** `byte_idx · k + i_inner`,
//!     - **bit-within-byte** `r`.
//!
//!   Equivalently: `z_packed` is organized in `n_outer/8` *stripes* of `k`
//!   contiguous bytes each. Stripe `byte_idx` covers all `i_inner ∈ {0,..,k}`
//!   for the same outer batch `i_outer ∈ {8·byte_idx, …, 8·byte_idx + 7}`.
//!   Each byte holds 8 outer bits for one i_inner.
//!
//!   In bit-position terms, the bit-index decomposes as:
//!   ```text
//!   LSB:  3 bits = r           (= low 3 bits of i_outer, = bit-within-byte)
//!         k_log bits = i_inner
//!   MSB:  (n_log − 3) bits = byte_idx (= upper bits of i_outer)
//!   ```
//!
//!   This layout makes the partial-fold column scan sequential — for each
//!   `byte_idx`, all `k` per-i_inner bytes are at consecutive byte positions,
//!   so we build a 256-entry sum table for the 8 outer values once per
//!   `byte_idx` and apply it across all `i_inner` with one lookup + one XOR
//!   per byte.

use crate::challenger::Challenger;
use crate::field::F128;
use crate::r1cs::SparseBinaryMatrix;
use crate::zerocheck::multilinear::lagrange_weights_naive;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::atomic::AtomicBool;
use std::thread::JoinHandle;

mod kernels;

#[cfg(target_arch = "x86_64")]
pub use kernels::partial_fold_packed_z_x86_tiled_padded;
#[cfg(target_arch = "aarch64")]
pub use kernels::{
    partial_fold_packed_z_neon_iblock_padded, partial_fold_packed_z_neon_oblock_padded,
    partial_fold_packed_z_neon_single, partial_fold_packed_z_neon_single_padded,
};

/// Bench-only A/B toggle: when set, [`partial_fold_packed_z_best`] uses the legacy
/// `i_inner`-partitioned `partial_fold_packed_z_neon_iblock_padded` instead of the
/// default outer(tile)-partitioned `partial_fold_packed_z_neon_oblock_padded`. The
/// two are bit-identical (GF(2¹²⁸) add is XOR — associative + commutative), so one
/// process can time both back-to-back and cancel thermal drift. The oblock default
/// builds each tile's sum-tables once instead of once per worker, scaling the fold
/// ~8.5× vs iblock's ~6.5× on 10 P-cores at m=32. See `benches/lincheck.rs` (FOLD_AB=1).
pub static FOLD_IBLOCK: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// LincheckCircuit: the per-block linear structure lincheck consumes
// ---------------------------------------------------------------------------
//
// Lincheck's hot path computes a single length-`k = 2^k_log` vector
//
//   `comb_vec[c] = α · ξ_A(c) + ξ_B(c)`
//
// where `ξ_M(c) = Σ_r eq_inner[r] · M[r, c]` is the eq-weighted column
// marginal of base matrix `M ∈ {A_0, B_0}`. Today's `sparse_row_fold_alpha_batched`
// computes it by scattering `eq_inner[r]` to every column in row r's nonzero
// set — cost ∝ NNZ.
//
// For circuit-shaped R1CS (Keccak, BLAKE3, SHA-256) the same `comb_vec` can be
// produced by walking the constraint graph in round order — same operations
// the witness gen already does, just with eq-weights instead of bit values.
// Per-hash impls can also avoid materializing matrices entirely (relevant for
// encodings where intermediate state slots are dropped and substitution would
// otherwise blow up A/B density).
//
// `LincheckCircuit` is the seam: `lincheck::prove`/`verify` take
// `&dyn LincheckCircuit` instead of a pair of matrices. The default impl
// `SparseMatrixCircuit` wraps the existing fused sparse kernel so callers
// that haven't ported get identical behavior.

/// Per-block linear structure consumed by lincheck. Implementations produce
/// the α-batched column marginal `comb_vec[c] = α · ξ_A(c) + ξ_B(c)` either
/// by sparse-matrix iteration (default) or by walking the circuit directly.
pub trait LincheckCircuit: Sync {
    /// Number of columns in the per-block matrices A_0, B_0 (= k = 2^k_log).
    fn n_cols(&self) -> usize;

    /// Compute `comb_vec[c] = α · (eq^T · A_0)[c] + (eq^T · B_0)[c]` over
    /// `c ∈ [0, n_cols())`. `eq_inner.len() == n_cols()`.
    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128>;

    /// Column index of a constant-one wire to pin, or `None` if the circuit has
    /// no such wire. When `Some(col)`, lincheck folds one extra `β`-term into the
    /// comb so the sumcheck also proves that the committed constant column is the
    /// all-ones vector (whose MLE is the constant `1`), closing the all-zero
    /// witness soundness gap. This REQUIRES the witness to set that wire to `1`
    /// in *every* batched instance — padding included. See
    /// `docs/const-wire-pin.md`. Default `None` keeps the transcript unchanged
    /// for circuits without a constant wire.
    fn const_pin_col(&self) -> Option<usize> {
        None
    }
}

/// Default `LincheckCircuit` over a pair of sparse binary matrices. Delegates
/// to the existing fused row-fold kernel. Callers that haven't migrated to a
/// per-hash circuit walker use this wrapper.
pub struct SparseMatrixCircuit<'a> {
    pub a_0: &'a SparseBinaryMatrix,
    pub b_0: &'a SparseBinaryMatrix,
    /// Constant-wire pin column (see [`LincheckCircuit::const_pin_col`]).
    const_pin: Option<usize>,
}

impl<'a> SparseMatrixCircuit<'a> {
    pub fn new(a_0: &'a SparseBinaryMatrix, b_0: &'a SparseBinaryMatrix) -> Self {
        debug_assert_eq!(a_0.num_rows, b_0.num_rows);
        debug_assert_eq!(a_0.num_cols, b_0.num_cols);
        Self {
            a_0,
            b_0,
            const_pin: None,
        }
    }

    /// Set the constant-wire pin column (see `docs/const-wire-pin.md`).
    pub fn with_const_pin(mut self, const_pin: Option<usize>) -> Self {
        self.const_pin = const_pin;
        self
    }
}

impl<'a> LincheckCircuit for SparseMatrixCircuit<'a> {
    fn n_cols(&self) -> usize {
        self.a_0.num_cols
    }
    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        sparse_row_fold_alpha_batched(alpha, self.a_0, self.b_0, eq_inner)
    }
    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }
}

/// Column-major (CSC) `LincheckCircuit`: `(A_0, B_0)` transposed once into
/// flat `col_ptr`/`row_idx` arrays. `fold_alpha_batched` becomes a gather —
/// each column reads its own row list and sums `eq_inner[r]`, so columns are
/// independent (parallel with no per-thread accumulator copies and no write
/// scatter) and the α-mul amortizes to one per column:
///
///   `comb[c] = α · Σ_{r ∈ colA(c)} eq_inner[r] + Σ_{r ∈ colB(c)} eq_inner[r]`
///
/// On the SHA-256 hybrid matrices (k = 2^15, ~1.3M nonzeros) this measures
/// ~7× faster than the row-scatter fold and ~100× faster than the symbolic
/// per-hash walkers; on BLAKE3 (~21M nonzeros) ~1.7× faster than row-scatter.
/// Construction costs one pass over the nonzeros (~4 ms / ~40 ms for the
/// above) — do it once at setup, e.g. via
/// [`crate::r1cs::BlockR1cs::csc_lincheck_circuit`].
#[derive(Clone)]
pub struct CscCircuit {
    n_cols: usize,
    /// Row-index bound: every entry of `a_rows` / `b_rows` is `< n_rows`
    /// (guaranteed by [`csc_from_rows`]). Justifies the unchecked
    /// `eq_inner[r]` gather in [`LincheckCircuit::fold_alpha_batched`].
    n_rows: usize,
    a_col_ptr: Vec<u32>,
    a_rows: Vec<u32>,
    b_col_ptr: Vec<u32>,
    b_rows: Vec<u32>,
    /// Narrow (`u16`) copies of `a_rows`/`b_rows`, used when every row index
    /// fits in 16 bits (`n_rows <= 65536`). The gather in
    /// [`LincheckCircuit::fold_alpha_batched`] streams the whole index array
    /// once per proof — at the BLAKE3 shape that is ~84 MB of `u32` against a
    /// 256 KiB `eq_inner` that stays in L2, so the pass is bound by the index
    /// stream, and halving it halves the DRAM traffic. When these are
    /// populated `a_rows`/`b_rows` are emptied (never both representations).
    a_rows16: Vec<u16>,
    b_rows16: Vec<u16>,
    /// True iff `a_rows16`/`b_rows16` hold the row indices.
    narrow: bool,
    /// Constant-wire pin column (see [`LincheckCircuit::const_pin_col`]).
    const_pin: Option<usize>,
}

/// `FLOCK_NO_LC_CSC_U16=1` restores the 32-bit CSC row-index arrays in
/// [`CscCircuit`] (exact A/B control: same sums, twice the index bytes).
fn csc_u16_rows_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_CSC_U16").is_none());
    *ON
}

/// Flatten one sparse matrix into CSC arrays: rows with a 1 in column `c` are
/// `rows_flat[col_ptr[c] as usize .. col_ptr[c+1] as usize]`.
fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    assert!(m.num_rows <= u32::MAX as usize);
    assert!(m.num_cols <= u32::MAX as usize);
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in &m.rows {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

// Compact Debug — the row arrays run to millions of entries.
impl std::fmt::Debug for CscCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (nnz_a, nnz_b) = if self.narrow {
            (self.a_rows16.len(), self.b_rows16.len())
        } else {
            (self.a_rows.len(), self.b_rows.len())
        };
        f.debug_struct("CscCircuit")
            .field("n_cols", &self.n_cols)
            .field("nnz_a", &nnz_a)
            .field("nnz_b", &nnz_b)
            .finish()
    }
}

impl CscCircuit {
    pub fn from_matrices(a_0: &SparseBinaryMatrix, b_0: &SparseBinaryMatrix) -> Self {
        Self::from_matrices_narrow(a_0, b_0, csc_u16_rows_enabled())
    }

    /// [`Self::from_matrices`] with the u16 row-index narrowing forced on or
    /// off — the A/B oracle hook for tests (production goes through the
    /// `FLOCK_NO_LC_CSC_U16` switch).
    #[doc(hidden)]
    pub fn from_matrices_narrow(
        a_0: &SparseBinaryMatrix,
        b_0: &SparseBinaryMatrix,
        want_narrow: bool,
    ) -> Self {
        assert_eq!(a_0.num_rows, b_0.num_rows);
        assert_eq!(a_0.num_cols, b_0.num_cols);
        let (a_col_ptr, mut a_rows) = csc_from_rows(a_0);
        let (b_col_ptr, mut b_rows) = csc_from_rows(b_0);
        // Narrow the row indices to u16 when the base matrix has at most
        // 2^16 rows (the BLAKE3/SHA-2 shapes have 2^14). Same values, half
        // the bytes streamed by `fold_alpha_batched`.
        let narrow = a_0.num_rows <= (1usize << 16) && want_narrow;
        let (a_rows16, b_rows16) = if narrow {
            let a16: Vec<u16> = a_rows.iter().map(|&r| r as u16).collect();
            let b16: Vec<u16> = b_rows.iter().map(|&r| r as u16).collect();
            a_rows = Vec::new();
            b_rows = Vec::new();
            (a16, b16)
        } else {
            (Vec::new(), Vec::new())
        };
        Self {
            n_cols: a_0.num_cols,
            n_rows: a_0.num_rows,
            a_col_ptr,
            a_rows,
            b_col_ptr,
            b_rows,
            a_rows16,
            b_rows16,
            narrow,
            const_pin: None,
        }
    }

    /// Set the constant-wire pin column (see `docs/const-wire-pin.md`).
    pub fn with_const_pin(mut self, const_pin: Option<usize>) -> Self {
        self.const_pin = const_pin;
        self
    }

    /// Evaluate `one_col` for every column — sequential below the rayon
    /// threshold, one `par_iter_mut` dispatch above it. Shared by the u16 and
    /// u32 gather bodies so both keep exactly the same dispatch structure.
    #[inline]
    fn map_cols(&self, one_col: impl Fn(usize) -> F128 + Sync + Send) -> Vec<F128> {
        use rayon::prelude::*;
        if self.n_cols < SUMCHECK_PAR_THRESHOLD {
            return (0..self.n_cols).map(one_col).collect();
        }
        let mut out = vec![F128::ZERO; self.n_cols];
        out.par_iter_mut()
            .enumerate()
            .for_each(|(c, slot)| *slot = one_col(c));
        out
    }
}

impl LincheckCircuit for CscCircuit {
    fn n_cols(&self) -> usize {
        self.n_cols
    }
    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }
    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        assert_eq!(eq_inner.len(), self.n_cols);
        // Row indices are `< n_rows` by construction ([`csc_from_rows`]);
        // checking here (once) instead of per nonzero drops the two-branch
        // bounds check from the ~7-instruction gather loop body.
        assert!(self.n_rows <= eq_inner.len());
        if self.narrow {
            // u16 index stream: identical row order, identical XOR order,
            // half the bytes read per nonzero.
            return self.map_cols(|c| {
                let mut sa = F128::ZERO;
                for &r in &self.a_rows16[self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize]
                {
                    // SAFETY: r < n_rows ≤ eq_inner.len(), asserted above.
                    sa += *unsafe { eq_inner.get_unchecked(r as usize) };
                }
                let mut sb = F128::ZERO;
                for &r in &self.b_rows16[self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize]
                {
                    // SAFETY: as above.
                    sb += *unsafe { eq_inner.get_unchecked(r as usize) };
                }
                alpha * sa + sb
            });
        }
        self.map_cols(|c| {
            let mut sa = F128::ZERO;
            for &r in &self.a_rows[self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize] {
                // SAFETY: r < n_rows ≤ eq_inner.len(), asserted above.
                sa += *unsafe { eq_inner.get_unchecked(r as usize) };
            }
            let mut sb = F128::ZERO;
            for &r in &self.b_rows[self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize] {
                // SAFETY: as above.
                sb += *unsafe { eq_inner.get_unchecked(r as usize) };
            }
            alpha * sa + sb
        })
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A "quirky" claim point: one univariate-skip coord (`z_skip`) representing
/// the first `k_skip` variables via the polynomial extension with the φ_8 basis,
/// followed by multilinear coords for the rest of inner and for outer.
///
/// Total "elements" = `1 + (k_log − k_skip) + n_log`, which is the shape the
/// zerocheck's extract_c output uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuirkyPoint {
    /// Univariate-skip challenge ∈ F₁₂₈. Binds all `k_skip` skip variables.
    pub z_skip: F128,
    /// Multilinear coords for the inner dims *after* the skip block. Length
    /// `k_log − k_skip`.
    pub x_inner_rest: Vec<F128>,
    /// Multilinear coords for the outer dims. Length `n_log = m − k_log`.
    pub x_outer: Vec<F128>,
}

/// Lincheck prover message: a partial product-sumcheck that proves the two
/// scalar consistency equations against `z` partially folded at the shared
/// outer half `x_ab.x_outer`, without sending the full length-`2^k_log`
/// `z_vec`. Sumcheck binds the high `k_log − k_skip` multilinear inner dims;
/// the low `k_skip` (φ8 univariate-skip) dims are handled by sending
/// `z_partial` (the post-sumcheck length-`2^k_skip` collapsed vector) and
/// applying a fresh-`z_skip` φ8 Lagrange combination at verify time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LincheckProof {
    /// Per-round messages `(q(1), q(∞))` of the `k_log − k_skip`-round
    /// product-sumcheck. `q(0)` is recovered from the running claim
    /// (`q(0) = T_r + q(1)` in char 2). Standard multilinear binding.
    pub rounds: Vec<(F128, F128)>,
    /// The length-`2^k_skip` collapse of the prover's `z_vec` over the
    /// sumcheck-bound `r_rest` dims. Folded against φ8 Lagrange weights at a
    /// fresh `z_skip` to yield the output claim's value.
    pub z_partial: Vec<F128>,
}

/// Lincheck output: one MLE evaluation claim on `z`, at the quirky inner
/// point `(r_inner_skip, r_inner_rest)` combined with `x_ab.x_outer`
/// (publicly known to the caller).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LincheckClaim {
    /// Univariate-skip post-vector random sample.
    pub r_inner_skip: F128,
    /// Multilinear post-vector random sample, length `k_log − k_skip`.
    pub r_inner_rest: Vec<F128>,
    /// `ẑ((r_inner_skip, r_inner_rest), x_ab.x_outer)` — the single
    /// `z`-claim derived from the A and B consistency checks.
    pub w: F128,
}

/// Reasons the verifier may reject.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    /// One of the proof vectors has the wrong length (expected `2^k_log`).
    BadVectorLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },
    /// One of the input quirky points has wrong `x_inner_rest` length
    /// (expected `k_log − k_skip`).
    BadInnerRestLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },
    /// One of the input quirky points has wrong `x_outer` length
    /// (expected `n_log = m − k_log`).
    BadOuterLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },
    /// One of the base matrices isn't `2^k_log × 2^k_log`.
    BadMatrixShape {
        which: &'static str,
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },
    /// `k_skip` exceeds `k_log` (the matrix inner dimension).
    KSkipExceedsKLog { k_skip: usize, k_log: usize },
    /// The scalar consistency check failed for one of (A, B, C).
    /// Detected: `Σ_{i_inner} M̂_0_quirky(z_skip, x_inner_rest, i_inner) · z_x_vec[i_inner] ≠ v`.
    ConsistencyFailed { which: &'static str },
}

// ---------------------------------------------------------------------------
// Core kernels
// ---------------------------------------------------------------------------

/// Build the eq-MLE table at `point ∈ F^d`. Returns a length-`2^d` vector
/// where `output[i] = Π_j (1 + point[j] + bit_j(i)) = Π_j eq(point[j], bit_j(i))`.
///
/// Standard "doubling-in-half" construction: `O(2^d)` F128 muls, no
/// inversions. Indexing is LSB-first — `bit_j(i)` is the `j`-th LSB of `i`.
pub fn build_eq_table(point: &[F128]) -> Vec<F128> {
    let d = point.len();
    let mut out: Vec<F128> = Vec::with_capacity(1usize << d);
    out.push(F128::ONE);
    for j in 0..d {
        let r_j = point[j];
        let len = 1usize << j;
        out.resize(2 * len, F128::ZERO);
        // Char-2: v*(1+r) = v + v*r. One GHASH plus an XOR per old entry.
        //   out[i]       = v + v*r_j      ← new bit_j = 0
        //   out[i + len] = v * r_j        ← new bit_j = 1
        // Forward iteration is safe: the [i] and [i+len] slots are disjoint.
        for i in 0..len {
            let v = out[i];
            let hi = v * r_j;
            out[i + len] = hi;
            out[i] = v + hi;
        }
    }
    out
}

/// Fold a sparse boolean matrix's rows against an eq table at the row
/// coords. Computes the **transposed** matrix-vector product:
///
///   `output[col] = Σ_{row: M[row, col] = 1} eq_table[row]`
///
/// This is the row-MLE `M̂_0(x_inner, ·)` evaluated at all boolean column
/// indices — the length-`k` vector the verifier needs for the consistency
/// check. Cost: `nnz(M)` F128 adds.
/// Below this matrix row count, the sequential path beats rayon dispatch
/// overhead. Tuned for `k = 2^14` (BLAKE3) — small matrices stay scalar,
/// big ones parallelize.
const SPARSE_ROW_FOLD_PAR_THRESHOLD: usize = 1usize << 12;

pub fn sparse_row_fold(matrix: &SparseBinaryMatrix, eq_table: &[F128]) -> Vec<F128> {
    assert_eq!(
        eq_table.len(),
        matrix.num_rows,
        "eq_table length must match matrix row count"
    );
    let n_cols = matrix.num_cols;
    if matrix.rows.len() < SPARSE_ROW_FOLD_PAR_THRESHOLD {
        let mut out = vec![F128::ZERO; n_cols];
        for (row_idx, row) in matrix.rows.iter().enumerate() {
            let e = eq_table[row_idx];
            for &col in row {
                out[col] += e;
            }
        }
        out
    } else {
        // Scatter-reduce: per-thread accumulator, XOR-merge at the end. Each
        // thread allocates a length-n_cols buffer (~256 KB at k=16384) — fine
        // vs the witness-scale buffers already in flight.
        use rayon::prelude::*;
        matrix
            .rows
            .par_iter()
            .enumerate()
            .fold(
                || vec![F128::ZERO; n_cols],
                |mut acc, (row_idx, row)| {
                    let e = eq_table[row_idx];
                    for &col in row {
                        acc[col] += e;
                    }
                    acc
                },
            )
            .reduce(
                || vec![F128::ZERO; n_cols],
                |mut a, b| {
                    for i in 0..n_cols {
                        a[i] += b[i];
                    }
                    a
                },
            )
    }
}

/// Partial fold of `z` at the outer half of a claim point — single-matrix,
/// **scalar reference**. Uses the lincheck `z_packed` stripe layout
/// (see module docs).
///
///   `output[i_inner] = Σ_{i_outer ∈ {0,1}^n_log}  z[i_inner, i_outer] · eq_outer[i_outer]`
///
/// Equivalently, `output[i_inner] = ẑ(i_inner_as_F128, x_outer)` for boolean
/// `i_inner`. Used as the cross-check oracle for the production
/// `partial_fold_packed_z_triple`.
pub fn partial_fold_packed_z(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 3, "need n_outer ≥ 8 for byte stripes");
    let n_stripes = n_outer / 8;

    let mut out = vec![F128::ZERO; k];
    for byte_idx in 0..n_stripes {
        let stripe = &z_packed[byte_idx * k..(byte_idx + 1) * k];
        for (i_inner, &byte) in stripe.iter().enumerate() {
            if byte == 0 {
                continue;
            }
            let mut bits = byte;
            while bits != 0 {
                let r = bits.trailing_zeros() as usize;
                let i_outer = 8 * byte_idx + r;
                out[i_inner] += eq_outer[i_outer];
                bits &= bits - 1;
            }
        }
    }
    out
}

/// **Optimized single-matrix partial fold.** Same shape as
/// [`partial_fold_packed_z`] but uses 256-entry **sum-table lookups** and is
/// parallelized via rayon. The hot inner kernel does just **1 byte load +
/// 1 table lookup + 1 XOR** per `(byte_idx, i_inner)` pair.
///
/// At m=29 multi-thread this is ~3× faster than the naive scalar
/// `partial_fold_packed_z` (which we keep as the cross-check reference).
///
/// Iteration:
/// 1. For each `byte_idx ∈ 0..n_outer/8`, build a 256-entry F128 table
///    where `table[b] = Σ_{r: bit r set in b} eq_outer[8·byte_idx + r]`.
///    Cost: 255 F128 XORs (doubling construction).
/// 2. Sweep the `k`-byte stripe at `z_packed[byte_idx·k .. (byte_idx+1)·k]`.
///    For each `i_inner`, do `out[i_inner] ^= table[z_byte]`.
///
/// Parallel: each worker owns a contiguous range of stripes and a private
/// length-`k` accumulator; results XOR-reduced.
pub fn partial_fold_packed_z_fast(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let k = 1usize << k_log;
    partial_fold_packed_z_fast_padded(z_packed, m, k_log, k, eq_outer)
}

/// Padding-aware variant of [`partial_fold_packed_z_fast`]. Skips rows
/// `i_inner ∈ [useful_bits, k)` — those rows hold zero in every block of an
/// honestly padded witness, so the fold over the outer dim is zero. Output
/// is byte-identical to the dense path on such witnesses.
pub fn partial_fold_packed_z_fast_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 3, "need n_outer ≥ 8 for byte stripes");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;

    let stripes_per_chunk = (n_stripes / 256).max(1);
    let bytes_per_chunk = stripes_per_chunk * k;

    // fold(): one length-k accumulator per WORKER rather than per chunk —
    // at large k the per-chunk accumulators of map().reduce() dominate MT
    // time with allocation + tree-reduce XOR traffic (keccak3: k = 2^17
    // means 2 MB per chunk across ~128 chunks).
    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![F128::ZERO; k],
            |mut acc, (chunk_idx, chunk_bytes)| {
                let stripe_start = chunk_idx * stripes_per_chunk;
                let mut table = vec![F128::ZERO; 256];
                for (rel_stripe, stripe) in chunk_bytes.chunks(k).enumerate() {
                    let byte_idx = stripe_start + rel_stripe;
                    build_sum_table(&eq_outer[8 * byte_idx..8 * byte_idx + 8], &mut table);
                    for (i_inner, &z_byte) in stripe[..useful_bits].iter().enumerate() {
                        acc[i_inner] += table[z_byte as usize];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Outer stripes processed together by the direct block-major fold. Eight
/// stripes cover 64 consecutive outer blocks: their 32 KiB of sum tables fit
/// in L1, while one 128-entry output row-group stays hot across all 8 tables.
const DIRECT_FOLD_TILE_STRIPES: usize = 8;

/// Ranked BLAKE3 has `m=32`, `k_log=14`, hence 18 outer variables. Splitting
/// those variables 9+9 replaces the 4 MiB full equality tensor with two 8 KiB
/// factors. Reconstructing each of the 2^18 weights once in the fold plus the
/// two factor builds saves exactly 260,098 F128 multiplications versus the
/// full doubling construction.
const BLOCK_MAJOR_FACTORED_EQ_N_LOG: usize = 18;
const BLOCK_MAJOR_FACTORED_EQ_LO_LOG: usize = 9;

/// Transpose one F128 row-group from 8 consecutive outer blocks into the byte
/// shape consumed by a lincheck sum table. Output byte `b` has bit `r` equal
/// to bit `b` of `lanes[r]`.
#[inline(always)]
fn transpose_8_f128s_to_128_bytes(lanes: &[F128; 8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), 128);
    let lo: [u64; 8] = std::array::from_fn(|r| lanes[r].lo);
    let hi: [u64; 8] = std::array::from_fn(|r| lanes[r].hi);
    let (out_lo, out_hi) = out.split_at_mut(64);
    crate::bits::transpose_8_u64s_to_64_bytes(&lo, out_lo);
    crate::bits::transpose_8_u64s_to_64_bytes(&hi, out_hi);
}

/// Byte selectors for the ranked gather/transpose `VPERMT2B` fusion.
///
/// A 512-bit `VPERMT2B` uses index bits 0..=5 for the byte offset and bit 6
/// to choose its second 64-byte table; bit 7 is ignored. Rows 0..4 live in
/// the first input ZMM and rows 4..8 in the second, with each F128 row laid
/// out as eight low-limb bytes followed by eight high-limb bytes. The prior
/// failed candidate used `128 + offset` for rows 4..8, which leaves bit 6
/// clear and therefore selects the first table again. `64 + offset` is the
/// required second-table encoding.
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    )
))]
const fn gather_transpose_vpermt2b_indices(high: bool) -> [u8; 64] {
    let mut indices = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        // The following GFNI affine transpose reads matrix byte `7 - row`,
        // so feed the rows in reverse order to cancel that reversal.
        let row = 7 - i % 8;
        let byte_in_limb = i / 8;
        let limb_offset = if high { 8 } else { 0 };
        indices[i] = if row < 4 {
            (16 * row + limb_offset + byte_in_limb) as u8
        } else {
            (64 + 16 * (row - 4) + limb_offset + byte_in_limb) as u8
        };
        i += 1;
    }
    indices
}

/// Direct partial fold from the canonical block-major F128 witness packing.
/// This avoids materializing the equally-sized byte-stripe copy used by
/// [`partial_fold_packed_z_fast`].
///
/// Each outer block contains `chunks_per_block = k / 128` F128 values, so
/// `z_packed[i_outer * chunks_per_block + q]` holds inner columns
/// `128*q..128*q+128`. For each group of 8 outer blocks, the corresponding 8
/// F128 values are transposed into 128 bytes; byte `b` then indexes the same
/// 256-entry sum table as the stripe fold.
pub fn partial_fold_packed_z_block_major(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    partial_fold_packed_z_block_major_padded(z_packed, m, k_log, 1usize << k_log, eq_outer)
}

/// Padding-aware variant of [`partial_fold_packed_z_block_major`]. Inner
/// columns `[useful_bits, k)` remain zero and are not read from the witness.
pub fn partial_fold_packed_z_block_major_padded(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    assert!(m >= k_log);
    let n_outer = 1usize << (m - k_log);
    assert_eq!(eq_outer.len(), n_outer);
    partial_fold_packed_z_block_major_padded_with_tables(
        z_packed,
        m,
        k_log,
        useful_bits,
        |outer_base| std::array::from_fn(|r| eq_outer[outer_base + r]),
        None,
    )
}

/// Block-major partial fold with an exactly factorized outer equality tensor.
///
/// `eq_outer[i] = eq_lo[i & (B - 1)] * eq_hi[i >> log2(B)]`, where
/// `B = eq_lo.len()`. The complete tensor is never materialized: each outer
/// weight is reconstructed exactly once while building its 8-bit sum table.
/// This preserves the existing witness sweep and private-accumulator schedule.
fn partial_fold_packed_z_block_major_factorized_padded(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_lo: &[F128],
    eq_hi: &[F128],
) -> Vec<F128> {
    partial_fold_packed_z_block_major_factorized_padded_with_top_bind(
        z_packed,
        m,
        k_log,
        useful_bits,
        eq_lo,
        eq_hi,
        None,
    )
}

/// Factorized block-major fold with an optional immediate bind of the top
/// remaining inner coordinate. The GFNI arm fuses that bind into its existing
/// cross-worker plane reduce; other targets bind the completed vector.
fn partial_fold_packed_z_block_major_factorized_padded_with_top_bind(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_lo: &[F128],
    eq_hi: &[F128],
    top_bind: Option<F128>,
) -> Vec<F128> {
    assert!(m >= k_log);
    assert!(eq_lo.len().is_power_of_two());
    assert!(eq_hi.len().is_power_of_two());
    let n_outer = 1usize << (m - k_log);
    assert_eq!(eq_lo.len() * eq_hi.len(), n_outer);
    let log_b = eq_lo.len().trailing_zeros() as usize;
    let lo_mask = eq_lo.len() - 1;

    partial_fold_packed_z_block_major_padded_with_tables(
        z_packed,
        m,
        k_log,
        useful_bits,
        |outer_base| {
            std::array::from_fn(|r| {
                let outer = outer_base + r;
                eq_lo[outer & lo_mask] * eq_hi[outer >> log_b]
            })
        },
        top_bind,
    )
}

/// Today's one-shot block-major fold at `x_outer`. Same dispatch as
/// [`prove_padded_inner`]: factorized eq when `n_log == 18`, else a
/// materialized outer table. Used by the last-ρ kick and the sequential
/// fallback so both produce the same `ẑ`, and by zerocheck's identity-C
/// round-one fold (see `univariate_skip_optimized::identity_c_inner_fold`).
pub(crate) fn fold_block_major_one_shot(
    z: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    x_outer: &[F128],
) -> Vec<F128> {
    let n_log = m - k_log;
    debug_assert_eq!(x_outer.len(), n_log);
    if n_log == BLOCK_MAJOR_FACTORED_EQ_N_LOG {
        let (outer_lo, outer_hi) = x_outer.split_at(BLOCK_MAJOR_FACTORED_EQ_LO_LOG);
        let eq_lo = build_eq_table(outer_lo);
        let eq_hi = build_eq_table(outer_hi);
        partial_fold_packed_z_block_major_factorized_padded(
            z,
            m,
            k_log,
            useful_bits,
            &eq_lo,
            &eq_hi,
        )
    } else {
        let eq_x_outer = build_eq_table(x_outer);
        partial_fold_packed_z_block_major_padded(z, m, k_log, useful_bits, &eq_x_outer)
    }
}

/// One-shot block-major outer fold followed immediately by binding the top
/// remaining inner coordinate. On the ranked GFNI path the bind is fused into
/// the worker-plane reduce, so the full inner vector is never materialized.
pub(crate) fn fold_block_major_one_shot_bind_top(
    z: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    x_outer: &[F128],
    r_top: F128,
) -> Vec<F128> {
    let n_log = m - k_log;
    debug_assert_eq!(x_outer.len(), n_log);
    if n_log == BLOCK_MAJOR_FACTORED_EQ_N_LOG {
        let (outer_lo, outer_hi) = x_outer.split_at(BLOCK_MAJOR_FACTORED_EQ_LO_LOG);
        let eq_lo = build_eq_table(outer_lo);
        let eq_hi = build_eq_table(outer_hi);
        partial_fold_packed_z_block_major_factorized_padded_with_top_bind(
            z,
            m,
            k_log,
            useful_bits,
            &eq_lo,
            &eq_hi,
            Some(r_top),
        )
    } else {
        let eq_x_outer = build_eq_table(x_outer);
        partial_fold_packed_z_block_major_padded_with_tables(
            z,
            m,
            k_log,
            useful_bits,
            |outer_base| std::array::from_fn(|lane| eq_x_outer[outer_base + lane]),
            Some(r_top),
        )
    }
}

/// Ranked identity-C fold plus the contribution of complete K-rows whose B
/// multilinear extension is one.
pub(crate) fn fold_block_major_one_shot_bind_top_ranked_one_rows(
    z: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    x_outer: &[F128],
    r_top: F128,
) -> (Vec<F128>, Vec<F128>) {
    assert_eq!(m, 32);
    assert_eq!(k_log, 14);
    assert_eq!(useful_bits, 15409);
    assert_eq!(x_outer.len(), BLOCK_MAJOR_FACTORED_EQ_N_LOG);
    let (outer_lo, outer_hi) = x_outer.split_at(BLOCK_MAJOR_FACTORED_EQ_LO_LOG);
    let eq_lo = build_eq_table(outer_lo);
    let eq_hi = build_eq_table(outer_hi);
    let log_b = eq_lo.len().trailing_zeros() as usize;
    let lo_mask = eq_lo.len() - 1;
    let (full, one) = partial_fold_packed_z_block_major_padded_with_tables_result(
        z,
        m,
        k_log,
        useful_bits,
        |outer_base| {
            std::array::from_fn(|lane| {
                let outer = outer_base + lane;
                eq_lo[outer & lo_mask] * eq_hi[outer >> log_b]
            })
        },
        Some(r_top),
        true,
    );
    (full, one.expect("ranked one-row fold requested"))
}

/// `FLOCK_NO_LC_NIBBLE_FOLD=1` disables the AVX-512 nibble-table accumulate
/// of the block-major sweep (exact A/B control: the scalar 256-entry
/// byte-table loop runs instead). Resolved once per process.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
fn lincheck_nibble_fold_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_NIBBLE_FOLD").is_none());
    *ON
}

/// Test-only latch forcing the block-major GFNI arm OFF (it ships on), so
/// both arms can be compared in one process (the env switches resolve once).
#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
pub(crate) static BM_GFNI_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// `FLOCK_NO_LC_BM_GFNI=1` restores the nibble-table accumulate in the
/// block-major sweep (exact same-binary A/B). Distinct from
/// `FLOCK_NO_LC_GFNI`, which guards only the byte-stripe dispatcher — the
/// block-major path never reaches it. Resolved once per process.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
fn block_major_gfni_enabled() -> bool {
    #[cfg(test)]
    if BM_GFNI_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_BM_GFNI").is_none());
    *ON
}

/// `FLOCK_NO_LC_GATHER_TR=1` restores the scalar stripe gather + staging
/// transpose in the GFNI block-major arm (exact same-binary A/B).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
/// `FLOCK_NO_LC_GATHER4=1` restores single-column gather+transpose visits in
/// the block-major GFNI fold (exact A/B control; the fused single-column arm
/// then serves every chunk). Resolved once per process.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi"))]
fn lc_gather4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_GATHER4").is_none());
    *ON
}

/// `FLOCK_NO_LC_ALPHA_OVERLAP=1` restores the park-first order: join the
/// kicked z-fold before `fold_alpha_batched` instead of after (exact
/// same-binary A/B; the overlap changes scheduling only).
fn lc_alpha_overlap_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_ALPHA_OVERLAP").is_none());
    *ON
}

/// Chunks of look-ahead for the grouped block-major gather prefetch. One
/// grouped visit consumes four F128 chunks = one 64-byte line per row, so
/// `4` is the next visit's line on the SAME eight rows.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi"))]
const LC_ZFOLD_PF_CHUNKS: usize = 4;

/// `FLOCK_NO_LC_ZFOLD_PF=1` restores the incumbent one-stripe-ahead prefetch
/// in the grouped block-major gather (exact same-binary A/B).
///
/// The incumbent asks for the next stripe's eight rows at the column the loop
/// is on now. This arm asks instead for the lines the SAME eight rows will
/// need on a later grouped visit. Same eight prefetch instructions per stripe,
/// different address: no work is added, and prefetches have no architectural
/// effect, so the folded values are bit-identical either way.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi"))]
fn lc_zfold_pf_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_ZFOLD_PF").is_none());
    *ON
}

/// `FLOCK_NO_LC_ZFOLD_PF_NEAR=1` restores the incumbent two-visit look-ahead
/// in the grouped gather prefetch above (exact same-binary A/B). Same eight
/// prefetch instructions per stripe, one line further out; a prefetch has no
/// architectural effect, so the folded values are identical either way.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi"))]
fn lc_zfold_pf_near_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_ZFOLD_PF_NEAR").is_none());
    *ON
}

/// `FLOCK_NO_LC_ZFOLD_PF_SPREAD=1` restores the incumbent delivery of the
/// grouped gather prefetch: every stripe's hint block issued inside the
/// gather loop, next to that stripe's own strided reads. The default arm
/// issues the same hints for the same lines from the fold loop that follows,
/// two stripes' worth per fold call. Exact same-binary A/B: a prefetch has
/// no architectural effect, so the folded values are identical either way.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi"))]
fn lc_zfold_pf_spread_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_ZFOLD_PF_SPREAD").is_none());
    *ON
}

/// Block-major row stride, in `F128`, at the ranked shape: `k = 2^14`, so
/// `chunks_per_block = k / 128 = 128`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi"))]
const LC_RANKED_CHUNKS_PER_BLOCK: usize = 128;

/// Sixteen T0 hints on the consecutive block-major rows `base .. base + 16`
/// (row stride `chunks_per_block` F128). The `ranked` arm exists only so the
/// stride is an immediate: identical addresses, identical order.
///
/// The sweep's row stride is `chunks_per_block = k / 128`, a value the
/// compiler only sees at run time, so each of the sixteen hints a spread
/// block issues rebuilds its own address with `lea` + `imul` + `shl`. At the
/// ranked shape that value is the constant 128 and the sixteen rows are
/// consecutive: handing the constant to the addressing collapses the whole
/// block to one base register plus fifteen `disp32`. Same sixteen lines, same
/// order, same look-ahead — a prefetch has no architectural effect and none
/// of the folded values are touched, so `ẑ` is bit-identical either way.
///
/// # Safety
/// `base .. base + 15 * chunks_per_block` must lie inside the witness
/// allocation. A prefetch never dereferences, but the pointer arithmetic
/// itself must stay in bounds.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi"))]
#[inline(always)]
unsafe fn lc_prefetch_rows16(base: *const F128, chunks_per_block: usize, ranked: bool) {
    use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
    // SAFETY: bounds per the contract.
    unsafe {
        if ranked {
            for i in 0..16 {
                _mm_prefetch(
                    base.add(i * LC_RANKED_CHUNKS_PER_BLOCK).cast::<i8>(),
                    _MM_HINT_T0,
                );
            }
        } else {
            let mut p = base;
            for _ in 0..16 {
                _mm_prefetch(p.cast::<i8>(), _MM_HINT_T0);
                p = p.add(chunks_per_block);
            }
        }
    }
}

#[allow(dead_code)] // Retained same-binary rollback selector.
fn lc_gather_tr_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_GATHER_TR").is_none());
    *ON
}

/// `FLOCK_NO_LINCHECK_GT_FUSE=1` restores the two-instruction
/// gather/transpose composition (`VPERMT2Q` lo/hi split plus `VPERMB` byte
/// transpose). The default uses one `VPERMT2B` per limb with the static
/// selectors from [`gather_transpose_vpermt2b_indices`].
#[cfg(any(
    test,
    all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    )
))]
fn lincheck_gt_fuse_disabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
fn lincheck_gt_fuse_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !lincheck_gt_fuse_disabled_value(std::env::var_os("FLOCK_NO_LINCHECK_GT_FUSE").as_deref())
    });
    *ON
}

/// `FLOCK_NO_LC_DYNAMIC_TILES=1` restores the fixed contiguous tile range
/// per worker in the block-major sweep (exact A/B control).
fn dynamic_tiles_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_DYNAMIC_TILES").is_none());
    *ON
}

/// `FLOCK_NO_LC_REDUCE_SINGLE_PASS=1` restores the per-worker sequence of
/// rayon reductions of the block-major partials (exact A/B control).
fn reduce_single_pass_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_REDUCE_SINGLE_PASS").is_none());
    *ON
}

/// `FLOCK_NO_LC_FOLD_UNTIMED=1` restores the *unconditional* per-tile /
/// per-chunk `Instant::now()` probes inside the block-major sweep (the
/// incumbent behaviour, and the way to get the tables / transpose+read /
/// accumulate split back under `LINCHECK_TRACE`). Left on (default), the
/// probes are skipped entirely: at the ranked shape the incumbent takes
/// 4 clock reads per (tile, 128-column chunk) — 256 tiles × 121 chunks ×
/// 4 ≈ 124 K `clock_gettime` calls per worker — in the middle of the
/// witness sweep. Pure instrumentation: the folded values are identical
/// either way.
fn fold_untimed_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_FOLD_UNTIMED").is_none());
    *ON
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[inline(always)]
fn gather_transpose_tile_scalar(
    z_packed: &[F128],
    chunks_per_block: usize,
    stripe_base: usize,
    q: usize,
    transposed: &mut [u8; 4 * DIRECT_FOLD_TILE_STRIPES * 128],
) {
    for t in 0..DIRECT_FOLD_TILE_STRIPES {
        let outer_base = 8 * (stripe_base + t);
        let lanes: [F128; 8] =
            std::array::from_fn(|r| z_packed[(outer_base + r) * chunks_per_block + q]);
        transpose_8_f128s_to_128_bytes(&lanes, &mut transposed[t * 128..(t + 1) * 128]);
    }
}

/// Gather and transpose one eight-stripe, four-column batch. `FUSE` makes
/// the VPERMT2B/incumbent choice compile-time inside the small leaf while the
/// caller keeps the runtime kill switch outside this eight-iteration loop.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn gather_transpose_group4_x86<const FUSE: bool>(
    z_packed: &[F128],
    chunks_per_block: usize,
    stripe_base: usize,
    q: usize,
    full_chunks: usize,
    pf_far: bool,
    pf_spread: bool,
    pf_chunks: usize,
    transposed: &mut [u8; 4 * DIRECT_FOLD_TILE_STRIPES * 128],
) {
    for t in 0..DIRECT_FOLD_TILE_STRIPES {
        let outer_base = 8 * (stripe_base + t);
        if pf_far && !pf_spread {
            // These eight rows, pf_chunks chunks on: the lines this stripe
            // demand-loads on a later grouped visit. Bounds keep every hint
            // within a chunk the sweep will actually demand.
            let qn = q + pf_chunks;
            if qn <= full_chunks && qn < chunks_per_block {
                unsafe {
                    for r in 0..8 {
                        core::arch::x86_64::_mm_prefetch(
                            z_packed
                                .as_ptr()
                                .add((outer_base + r) * chunks_per_block + qn)
                                .cast::<i8>(),
                            core::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }
        } else if pf_far {
        } else if t + 1 < DIRECT_FOLD_TILE_STRIPES {
            let next_base = 8 * (stripe_base + t + 1);
            // One line per row covers all four columns.
            unsafe {
                for r in 0..8 {
                    core::arch::x86_64::_mm_prefetch(
                        z_packed
                            .as_ptr()
                            .add((next_base + r) * chunks_per_block + q)
                            .cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }
        // SAFETY: rows (outer_base + r) * chunks_per_block + q + c for
        // r in 0..8, c in 0..4 are the four full chunks selected by the
        // caller; every column slab is 1024 writable bytes.
        unsafe {
            kernels::gather_transpose_stripe4_x86::<FUSE>(
                z_packed.as_ptr().add(outer_base * chunks_per_block + q),
                chunks_per_block,
                transposed.as_mut_ptr().add(t * 128),
                1024,
            );
        }
    }
}

/// Single-column counterpart of [`gather_transpose_group4_x86`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[inline(always)]
fn gather_transpose_tile_x86<const FUSE: bool>(
    z_packed: &[F128],
    chunks_per_block: usize,
    stripe_base: usize,
    q: usize,
    transposed: &mut [u8; 4 * DIRECT_FOLD_TILE_STRIPES * 128],
) {
    for t in 0..DIRECT_FOLD_TILE_STRIPES {
        let outer_base = 8 * (stripe_base + t);
        // Row-strided F128 loads defeat sequential hardware prefetch. Pull
        // the next stripe into L1, without crossing a dynamically scheduled
        // tile boundary.
        if t + 1 < DIRECT_FOLD_TILE_STRIPES {
            let next_base = 8 * (stripe_base + t + 1);
            unsafe {
                for r in 0..8 {
                    core::arch::x86_64::_mm_prefetch(
                        z_packed
                            .as_ptr()
                            .add((next_base + r) * chunks_per_block + q)
                            .cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }
        // SAFETY: the caller's q is a live chunk, all eight row-strided
        // F128 values are in bounds, and the destination stripe is 128 B.
        unsafe {
            kernels::gather_transpose_stripe_x86::<FUSE>(
                z_packed.as_ptr().add(outer_base * chunks_per_block + q),
                chunks_per_block,
                transposed.as_mut_ptr().add(t * 128),
            );
        }
    }
}

/// GFNI plane-major arm of the block-major sweep: per (tile, 128-column
/// chunk) the eight gathered+transposed stripe rows drain through
/// [`kernels::gfni_fold_tile`] into per-worker byte-plane accumulators
/// (16 planes x 64 columns per 64-column block); one transpose back to F128
/// happens inside the cross-worker reduce. Bit-identical to the
/// scalar/nibble arms: F128 addition IS bitwise XOR, so plane-domain
/// accumulation commutes with the transpose, and the whole-block writes
/// past `useful_bits` land on index bytes that are ZERO in memory (r1cs
/// zero padding) through a linear map with no constant — contributing
/// nothing, exactly like the masked stores they replace.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[inline(always)]
fn transpose_8x8_bytes(mut rows: [u64; 8]) -> [u64; 8] {
    const M1: u64 = 0x00FF_00FF_00FF_00FF;
    for i in (0..8).step_by(2) {
        let t = ((rows[i] >> 8) ^ rows[i + 1]) & M1;
        rows[i + 1] ^= t;
        rows[i] ^= t << 8;
    }

    const M2: u64 = 0x0000_FFFF_0000_FFFF;
    for i in [0, 1, 4, 5] {
        let t = ((rows[i] >> 16) ^ rows[i + 2]) & M2;
        rows[i + 2] ^= t;
        rows[i] ^= t << 16;
    }

    const M4: u64 = 0x0000_0000_FFFF_FFFF;
    for i in 0..4 {
        let t = ((rows[i] >> 32) ^ rows[i + 4]) & M4;
        rows[i + 4] ^= t;
        rows[i] ^= t << 32;
    }
    rows
}

/// Reduce one 64-column GFNI plane block across workers and transpose it
/// back to the canonical F128 column layout.
///
/// # Safety
/// For every entry in `active_workers`, all 1024 bytes of plane block `blk`
/// must have been initialized, and every worker index/block range must lie in
/// `planes`. The producer join must happen-before this call.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[inline(always)]
unsafe fn reduce_worker_plane_block(
    planes: &[core::mem::MaybeUninit<u8>],
    worker_stride: usize,
    active_workers: &[usize],
    blk: usize,
    out: &mut [F128],
) {
    debug_assert_eq!(out.len(), 64);
    let Some((&first_worker, rest_workers)) = active_workers.split_first() else {
        out.fill(F128::ZERO);
        return;
    };
    let base = blk * 1024;
    let mut acc = [0u8; 1024];
    let first = first_worker * worker_stride + base;
    let first_block = &planes[first..first + 1024];
    // SAFETY: callers reduce only `blk < live_blocks`, and `first_worker` is
    // active. Its first claimed tile called `gfni_fold_tile(seed_zero=true)`,
    // which stores all 16 x 64 bytes of every live block. The Rayon producer
    // join completed before reduction starts, so this exact block is fully
    // initialized and may now be viewed as bytes.
    let first_block = unsafe {
        core::slice::from_raw_parts(first_block.as_ptr().cast::<u8>(), first_block.len())
    };
    acc.copy_from_slice(first_block);
    for &w in rest_workers {
        let src = &planes[w * worker_stride + base..w * worker_stride + base + 1024];
        // SAFETY: the same active-worker/live-block proof as `first_block`.
        let src = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<u8>(), src.len()) };
        // SAFETY: both slices are 1024 bytes (16 x 64); XOR is bitwise so
        // VPXORD equals the scalar `*a ^= *b` loop byte-for-byte.
        unsafe {
            kernels::xor_bytes_avx512(acc.as_mut_ptr(), src.as_ptr(), 1024);
        }
    }
    // The plane rows are contiguous, while the old per-column loop made 16
    // strided byte loads for every F128. Transpose eight 8-byte groups with
    // GPR delta-swaps so every load is a contiguous u64.
    for group in 0..8 {
        let mut lo_rows = [0u64; 8];
        let mut hi_rows = [0u64; 8];
        for byte in 0..8 {
            let lo = byte * 64 + group * 8;
            let hi = (byte + 8) * 64 + group * 8;
            lo_rows[byte] = u64::from_le_bytes(acc[lo..lo + 8].try_into().unwrap());
            hi_rows[byte] = u64::from_le_bytes(acc[hi..hi + 8].try_into().unwrap());
        }
        let lo_cols = transpose_8x8_bytes(lo_rows);
        let hi_cols = transpose_8x8_bytes(hi_rows);
        for col in 0..8 {
            out[group * 8 + col] = F128 {
                lo: lo_cols[col],
                hi: hi_cols[col],
            };
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
#[allow(clippy::too_many_arguments)]
fn fold_block_major_gfni(
    z_packed: &[F128],
    k: usize,
    chunks_per_block: usize,
    useful_bits: usize,
    useful_chunks: usize,
    n_workers: usize,
    tiles_per_worker: usize,
    n_tiles: usize,
    dynamic: bool,
    eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync),
    top_bind: Option<F128>,
    ranked_one_rows: bool,
) -> (Vec<F128>, Option<Vec<F128>>) {
    use rayon::prelude::*;
    const TILE_GRAB: usize = 4;
    let next_tile = std::sync::atomic::AtomicUsize::new(0);
    // Same total footprint as the F128 partials (k*16 bytes per worker). The
    // first claimed tile seeds every live plane block from register zero, so
    // eager zeroing is dead work. Dynamic claiming can leave a Rayon chunk
    // with no tile, however; `active` keeps those stale chunks out of reduce.
    // MaybeUninit keeps the whole Rayon slice valid while only active/live
    // subranges are initialized; it is never reinterpreted wholesale as u8.
    let mut planes = crate::alloc_uninit_vec::<core::mem::MaybeUninit<u8>>(n_workers * k * 16);
    let mut active = vec![0u8; n_workers];
    planes
        .par_chunks_mut(k * 16)
        .zip(active.par_iter_mut())
        .enumerate()
        .for_each(|(worker, (wplanes, worker_active))| {
            let tile_lo = worker * tiles_per_worker;
            let tile_hi = ((worker + 1) * tiles_per_worker).min(n_tiles);
            let (mut claim_lo, mut claim_hi) = if dynamic {
                (0usize, 0usize)
            } else {
                (tile_lo, tile_hi)
            };
            let mut mats = [0u64; 128];
            debug_assert_eq!(DIRECT_FOLD_TILE_STRIPES * 16, mats.len());
            // Four column-slabs of 8×128 bytes: the grouped gather writes
            // column c at slab c; the single-column arms use slab 0 only.
            let mut transposed = [0u8; 4 * DIRECT_FOLD_TILE_STRIPES * 128];
            // First tile this worker writes into an uninitialized plane
            // buffer: seed the GFNI acc from a register zero idiom. Later
            // tiles load only blocks that this first tile initialized.
            let mut first_tile = true;
            // Fused register gather+transpose (no staging arrays); the
            // scalar path stays as the kill-switch arm.
            #[cfg(target_feature = "avx512vbmi")]
            let gather_tr_fused = lc_gather_tr_enabled();
            // Resolve the corrected VPERMT2B-vs-incumbent gate once per
            // worker. The hot q loops branch once per eight-stripe batch;
            // their const-generic leaf has no per-gather gate branch.
            #[cfg(target_feature = "avx512vbmi")]
            let gather_tr_vpermt2b = lincheck_gt_fuse_enabled();
            // Grouped-gather prefetch distance, resolved once per worker
            // (never inside the tile / chunk / stripe loops).
            #[cfg(target_feature = "avx512vbmi")]
            let pf_far = lc_zfold_pf_enabled();
            #[cfg(target_feature = "avx512vbmi")]
            let pf_chunks = if lc_zfold_pf_near_enabled() {
                LC_ZFOLD_PF_CHUNKS
            } else {
                2 * LC_ZFOLD_PF_CHUNKS
            };
            #[cfg(target_feature = "avx512vbmi")]
            let pf_spread = lc_zfold_pf_spread_enabled();
            // Ranked-stride addressing, resolved once per worker (never
            // inside the tile / chunk / stripe loops).
            #[cfg(target_feature = "avx512vbmi")]
            let cpb_ranked = chunks_per_block == LC_RANKED_CHUNKS_PER_BLOCK;
            loop {
                let tile = if claim_lo < claim_hi {
                    claim_lo += 1;
                    claim_lo - 1
                } else if dynamic {
                    let lo = next_tile.fetch_add(TILE_GRAB, std::sync::atomic::Ordering::Relaxed);
                    if lo >= n_tiles {
                        break;
                    }
                    claim_lo = lo + 1;
                    claim_hi = (lo + TILE_GRAB).min(n_tiles);
                    lo
                } else {
                    break;
                };
                let stripe_base = tile * DIRECT_FOLD_TILE_STRIPES;
                for t in 0..DIRECT_FOLD_TILE_STRIPES {
                    let eq8 = eq8_at(8 * (stripe_base + t));
                    kernels::fold_mats_from_basis(&eq8, &mut mats[t * 16..(t + 1) * 16]);
                }
                let mut q = 0usize;
                // Grouped arm: four full 128-bit chunks per gather visit.
                // The row stride is 2048 bytes, so a tile's 64 live rows
                // land in two L1 sets; one wide load per row per FOUR
                // columns quarters the residency each row line needs there.
                // Full chunks only (chunk_bits == 128 ⇔ q < useful_bits/128);
                // the ragged final chunk takes the single-column arm below.
                #[cfg(target_feature = "avx512vbmi")]
                if gather_tr_fused && lc_gather4_enabled() {
                    let full_chunks = useful_bits / 128;
                    while q + 4 <= full_chunks {
                        if gather_tr_vpermt2b {
                            gather_transpose_group4_x86::<true>(
                                z_packed,
                                chunks_per_block,
                                stripe_base,
                                q,
                                full_chunks,
                                pf_far,
                                pf_spread,
                                pf_chunks,
                                &mut transposed,
                            );
                        } else {
                            gather_transpose_group4_x86::<false>(
                                z_packed,
                                chunks_per_block,
                                stripe_base,
                                q,
                                full_chunks,
                                pf_far,
                                pf_spread,
                                pf_chunks,
                                &mut transposed,
                            );
                        }
                        for c in 0..4 {
                            // Spread delivery: the same eight-hints-per-stripe
                            // block, issued from the fold that follows the
                            // gather instead of from the gather itself, two
                            // stripes at a time. Same lines, same look-ahead.
                            if pf_far && pf_spread {
                                let qn = q + pf_chunks;
                                if qn <= full_chunks && qn < chunks_per_block {
                                    // Stripes 2c and 2c+1 are the SIXTEEN
                                    // consecutive rows 8*stripe_base + 16c ..
                                    // + 16, so one base pointer and a fixed
                                    // row stride reach every hint the two
                                    // eight-row blocks used to address one at
                                    // a time. Same sixteen lines, same order.
                                    // SAFETY: those rows are inside this
                                    // tile's 64 and `qn < chunks_per_block`
                                    // keeps the column inside the block, so
                                    // every address the helper forms is in
                                    // bounds; a prefetch never dereferences.
                                    unsafe {
                                        lc_prefetch_rows16(
                                            z_packed.as_ptr().add(
                                                (8 * stripe_base + 16 * c) * chunks_per_block + qn,
                                            ),
                                            chunks_per_block,
                                            cpb_ranked,
                                        );
                                    }
                                }
                            }
                            // SAFETY: as for the single-column call below;
                            // every grouped chunk is full (2 blocks of 64).
                            unsafe {
                                kernels::gfni_fold_tile(
                                    transposed.as_ptr().add(c * 1024),
                                    128,
                                    2,
                                    &mats,
                                    wplanes.as_mut_ptr().cast::<u8>().add(2 * (q + c) * 1024),
                                    first_tile,
                                );
                            }
                        }
                        q += 4;
                    }
                }
                while q < useful_chunks {
                    let inner_base = q * 128;
                    let chunk_bits = (useful_bits - inner_base).min(128);
                    #[cfg(target_feature = "avx512vbmi")]
                    if gather_tr_fused {
                        if gather_tr_vpermt2b {
                            gather_transpose_tile_x86::<true>(
                                z_packed,
                                chunks_per_block,
                                stripe_base,
                                q,
                                &mut transposed,
                            );
                        } else {
                            gather_transpose_tile_x86::<false>(
                                z_packed,
                                chunks_per_block,
                                stripe_base,
                                q,
                                &mut transposed,
                            );
                        }
                    } else {
                        gather_transpose_tile_scalar(
                            z_packed,
                            chunks_per_block,
                            stripe_base,
                            q,
                            &mut transposed,
                        );
                    }
                    #[cfg(not(target_feature = "avx512vbmi"))]
                    gather_transpose_tile_scalar(
                        z_packed,
                        chunks_per_block,
                        stripe_base,
                        q,
                        &mut transposed,
                    );
                    // SAFETY: `transposed` holds 8 stripes x 128 bytes at
                    // stride 128 (max read 7*128 + 2*64 = 1024 = its size);
                    // the worker planes cover (2q + chunk blocks) * 1024
                    // bytes for every q < useful_chunks <= k/128. first_tile
                    // is true iff this worker has not yet stored into wplanes.
                    unsafe {
                        kernels::gfni_fold_tile(
                            transposed.as_ptr(),
                            128,
                            chunk_bits.div_ceil(64),
                            &mats,
                            wplanes.as_mut_ptr().cast::<u8>().add(2 * q * 1024),
                            first_tile,
                        );
                    }
                    q += 1;
                }
                first_tile = false;
            }
            // This byte is exclusively owned by the zipped Rayon chunk. A
            // true `first_tile` means dynamic claiming assigned it no work,
            // so none of its uninitialized plane bytes may enter reduction.
            *worker_active = u8::from(!first_tile);
        });

    // Cross-worker reduce + transpose-back in ONE parallel pass over
    // 64-column blocks (a standalone transpose would be a full-buffer
    // read+write pass — the class this arm deletes). When the caller will
    // immediately bind the top coordinate, pair each low/high plane block
    // here and write only the already-bound low half. This preserves the
    // incumbent reduction and transpose leaves while avoiding a length-k
    // intermediate allocation plus its complete readback.
    let worker_stride = k * 16;
    // Iteration over the marker vector preserves the incumbent ascending
    // worker reduction order while deleting inactive zero contributions.
    let active_workers: Vec<usize> = active
        .into_iter()
        .enumerate()
        .filter_map(|(worker, active)| (active != 0).then_some(worker))
        .collect();
    let live_blocks = useful_bits.div_ceil(64);
    let out_len = if top_bind.is_some() { k / 2 } else { k };
    // Keep output initialized as F128 throughout. The much larger plane
    // buffer carries the dead zero-fill; avoiding this comparatively small
    // clear would require a separate MaybeUninit ownership conversion.
    let mut out = vec![F128::ZERO; out_len];
    let mut one = ranked_one_rows.then(|| vec![F128::ZERO; out_len]);
    debug_assert!(!ranked_one_rows || (k == 1 << 14 && top_bind.is_some()));
    let reduce_block = |blk: usize, o: &mut [F128], one_o: Option<&mut [F128]>| {
        if blk < live_blocks {
            // SAFETY: `active_workers` contains exactly producer chunks that
            // claimed a tile. Their first tile used `seed_zero=true` and
            // stored every byte of each `blk < live_blocks`; the producer
            // parallel iterator joined before this reduction starts.
            unsafe {
                reduce_worker_plane_block(&planes, worker_stride, &active_workers, blk, o);
            }
        } else {
            // Padding has no backing worker-plane writes in the uninitialized
            // allocation, but its canonical folded value is algebraic zero.
            o.fill(F128::ZERO);
        }
        if let Some(r) = top_bind {
            let mut hi = [F128::ZERO; 64];
            let hi_blk = blk + k / 128;
            if hi_blk < live_blocks {
                // SAFETY: identical to the low-block call above; `hi_blk` is
                // explicitly constrained to the fully stored live range.
                unsafe {
                    reduce_worker_plane_block(
                        &planes,
                        worker_stride,
                        &active_workers,
                        hi_blk,
                        &mut hi,
                    );
                }
            }
            if let Some(one_o) = one_o {
                if blk < 18 {
                    let scale = F128::ONE + r;
                    for (dst, src) in one_o.iter_mut().zip(o.iter()) {
                        *dst = scale * *src;
                    }
                } else if (108..112).contains(&blk) {
                    for (dst, src) in one_o.iter_mut().zip(hi.iter()) {
                        *dst = r * *src;
                    }
                } else {
                    one_o.fill(F128::ZERO);
                }
            }
            crate::field::f128_slice::bind_split_half(o, &hi, r);
        }
    };
    if let Some(one) = one.as_mut() {
        out.par_chunks_mut(64)
            .zip(one.par_chunks_mut(64))
            .enumerate()
            .for_each(|(blk, (o, one_o))| reduce_block(blk, o, Some(one_o)));
    } else {
        out.par_chunks_mut(64)
            .enumerate()
            .for_each(|(blk, o)| reduce_block(blk, o, None));
    }
    (out, one)
}

/// Shared block-major witness sweep. `eq8_at(outer_base)` returns the eight
/// outer weights for the stripe beginning at `outer_base`; the sweep builds
/// whichever subset tables its accumulate kernel needs from them (the
/// 256-entry byte table for the scalar loop, or the AVX-512 kernel's two
/// 16-entry nibble tables — both are exact splits of the same subset sums).
fn partial_fold_packed_z_block_major_padded_with_tables(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq8_at: impl Fn(usize) -> [F128; 8] + Sync,
    top_bind: Option<F128>,
) -> Vec<F128> {
    partial_fold_packed_z_block_major_padded_with_tables_result(
        z_packed,
        m,
        k_log,
        useful_bits,
        eq8_at,
        top_bind,
        false,
    )
    .0
}

fn partial_fold_packed_z_block_major_padded_with_tables_result(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq8_at: impl Fn(usize) -> [F128; 8] + Sync,
    top_bind: Option<F128>,
    ranked_one_rows: bool,
) -> (Vec<F128>, Option<Vec<F128>>) {
    use rayon::prelude::*;

    assert!(m >= k_log);
    assert!(k_log >= 7, "block-major F128 fold requires k >= 128");
    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    let chunks_per_block = k / 128;
    assert_eq!(z_packed.len(), n_outer * chunks_per_block);
    assert!(n_log >= 3, "need n_outer >= 8 for byte groups");
    assert!(useful_bits <= k);

    let n_stripes = n_outer / 8;
    let n_tiles = n_stripes.div_ceil(DIRECT_FOLD_TILE_STRIPES);
    let p = rayon::current_num_threads().max(1);
    let tiles_per_worker = n_tiles.div_ceil(p);
    let n_workers = n_tiles.div_ceil(tiles_per_worker);
    let useful_chunks = useful_bits.div_ceil(128);
    // Dynamic tile claiming: workers grab `TILE_GRAB` tiles at a time from a
    // shared counter instead of a fixed contiguous range, so a worker whose
    // thread was descheduled (or shares a core) does not gate the join.
    // Every tile is still swept exactly once by exactly one worker; the
    // per-column XOR association changes but the sum does not.
    const TILE_GRAB: usize = 4;
    let next_tile = std::sync::atomic::AtomicUsize::new(0);
    let dynamic = dynamic_tiles_enabled();
    // Per-tile/per-chunk probe clocks: only with LINCHECK_TRACE (or the
    // kill switch, which restores the always-on probes). Resolved once here
    // so the worker loop reads a register, not the environment.
    let trace_fold = std::env::var_os("LINCHECK_TRACE").is_some();
    // NB: the split probes follow the kill switch alone, NOT `LINCHECK_TRACE`,
    // so `LINCHECK_TRACE=1` measures the production sweep. Set
    // `FLOCK_NO_LC_FOLD_UNTIMED=1` together with the trace to get the
    // tables / transpose+read / accumulate split back (it costs ~1 ms/worker).
    let timing = !fold_untimed_enabled();

    // GFNI plane-major arm: exact-tile shapes only (no ragged last tile),
    // 64-column blocks available. Everything else falls through to the
    // incumbent nibble/scalar sweep unchanged.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "gfni"
    ))]
    if block_major_gfni_enabled() && n_stripes % DIRECT_FOLD_TILE_STRIPES == 0 && k_log >= 6 {
        let _ = (trace_fold, timing);
        return fold_block_major_gfni(
            z_packed,
            k,
            chunks_per_block,
            useful_bits,
            useful_chunks,
            n_workers,
            tiles_per_worker,
            n_tiles,
            dynamic,
            &eq8_at,
            top_bind,
            ranked_one_rows,
        );
    }

    // Outer-tile partitioning reads every useful z chunk exactly once and
    // builds every sum table once. Each worker owns a private length-k partial;
    // the final XOR reduction is small relative to the witness pass.
    let probe_t0 = std::time::Instant::now();
    let mut partials = vec![F128::ZERO; n_workers * k];
    let probe_t1 = std::time::Instant::now();
    partials
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(worker, partial)| {
            let wt0 = std::time::Instant::now();
            let mut t_tables = std::time::Duration::ZERO;
            let mut t_tr = std::time::Duration::ZERO;
            let mut t_acc = std::time::Duration::ZERO;
            let tile_lo = worker * tiles_per_worker;
            let tile_hi = ((worker + 1) * tiles_per_worker).min(n_tiles);
            // Static range (kill switch) or dynamic claims from the counter.
            let mut claim_lo = tile_lo;
            let mut claim_hi = tile_hi;
            if dynamic {
                claim_lo = 0;
                claim_hi = 0;
            }
            let mut tables = vec![F128::ZERO; DIRECT_FOLD_TILE_STRIPES * 256];
            let mut transposed = [0u8; DIRECT_FOLD_TILE_STRIPES * 128];
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            ))]
            let mut nib_tables = [[0u64; 64]; DIRECT_FOLD_TILE_STRIPES];
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "avx512bw"
            ))]
            let nibble_ok = lincheck_nibble_fold_enabled();

            loop {
                let tile = if claim_lo < claim_hi {
                    claim_lo += 1;
                    claim_lo - 1
                } else if dynamic {
                    let lo = next_tile.fetch_add(TILE_GRAB, std::sync::atomic::Ordering::Relaxed);
                    if lo >= n_tiles {
                        break;
                    }
                    claim_lo = lo + 1;
                    claim_hi = (lo + TILE_GRAB).min(n_tiles);
                    lo
                } else {
                    break;
                };
                let stripe_base = tile * DIRECT_FOLD_TILE_STRIPES;
                let tile_stripes = (n_stripes - stripe_base).min(DIRECT_FOLD_TILE_STRIPES);
                // Full tiles on AVX-512 take the nibble-table kernel and never
                // build the 256-entry tables; every other tile builds them for
                // the scalar loop.
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512bw"
                ))]
                let use_nibble = nibble_ok && tile_stripes == DIRECT_FOLD_TILE_STRIPES;
                #[cfg(not(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "avx512bw"
                )))]
                let use_nibble = false;
                let tt0 = if timing {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                for t in 0..tile_stripes {
                    let outer_base = 8 * (stripe_base + t);
                    let eq8 = eq8_at(outer_base);
                    #[cfg(all(
                        target_arch = "x86_64",
                        target_feature = "avx512f",
                        target_feature = "avx512bw"
                    ))]
                    if use_nibble {
                        kernels::build_nibble_tables(&eq8, &mut nib_tables[t]);
                        continue;
                    }
                    build_sum_table(&eq8, &mut tables[t * 256..(t + 1) * 256]);
                }

                if let Some(tt0) = tt0 {
                    t_tables += tt0.elapsed();
                }
                // Keep one 128-column (2 KiB) output group hot while applying
                // all tables in this outer tile.
                for q in 0..useful_chunks {
                    let inner_base = q * 128;
                    let chunk_bits = (useful_bits - inner_base).min(128);
                    let tq0 = if timing {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    // Full tiles take the all-NEON gather + bit-transpose
                    // kernel (q-loads, uzp lo/hi split, tbl + vector-resident
                    // swap rounds, no bounds checks) — byte-identical output
                    // to the scalar-formed gather below, which remains as the
                    // partial-tile fallback and `FLOCK_NO_LINCHECK_QFORM`
                    // kill-switch path.
                    #[cfg(target_arch = "aarch64")]
                    let transposed_done = if tile_stripes == DIRECT_FOLD_TILE_STRIPES
                        && kernels::lincheck_qform_enabled()
                    {
                        // SAFETY: lane (t, r) is read at index
                        // `(8·(stripe_base + t) + r) · chunks_per_block + q`
                        // — exactly the indices the scalar path reads; the
                        // full-tile guard plus `q < useful_chunks ≤
                        // chunks_per_block` keep all 64 in bounds. The output
                        // is the whole 8×128 `transposed` buffer.
                        unsafe {
                            kernels::gather_transpose_tile_neon(
                                z_packed
                                    .as_ptr()
                                    .add(8 * stripe_base * chunks_per_block + q),
                                chunks_per_block,
                                transposed.as_mut_ptr(),
                            );
                        }
                        true
                    } else {
                        false
                    };
                    #[cfg(not(target_arch = "aarch64"))]
                    let transposed_done = false;
                    if !transposed_done {
                        for t in 0..tile_stripes {
                            let outer_base = 8 * (stripe_base + t);
                            let lanes: [F128; 8] = std::array::from_fn(|r| {
                                z_packed[(outer_base + r) * chunks_per_block + q]
                            });
                            transpose_8_f128s_to_128_bytes(
                                &lanes,
                                &mut transposed[t * 128..(t + 1) * 128],
                            );
                        }
                    }
                    if let Some(tq0) = tq0 {
                        t_tr += tq0.elapsed();
                    }
                    let ta0 = if timing {
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    let group = &mut partial[inner_base..inner_base + chunk_bits];
                    // Full tiles take the two-stream NEON wavefront leaf
                    // (paired 8-column blocks, 16 register accumulators —
                    // bit-identical XOR order, ~2× the independent lookup
                    // chains in flight). Partial last tiles and the
                    // `chunk_bits % 8` remainder use the scalar chain below.
                    #[cfg(target_arch = "aarch64")]
                    let b_done = if tile_stripes == DIRECT_FOLD_TILE_STRIPES {
                        kernels::fold_block_major_chunk_neon_x2(
                            &transposed,
                            &tables,
                            group,
                            chunk_bits,
                        )
                    } else {
                        0
                    };
                    #[cfg(all(
                        target_arch = "x86_64",
                        target_feature = "avx512f",
                        target_feature = "avx512bw"
                    ))]
                    let b_done = if use_nibble {
                        // SAFETY: cfg guarantees AVX-512F/BW; `transposed`
                        // has 8×128 bytes, `nib_tables` 8 stripes, `group`
                        // exactly `chunk_bits` F128.
                        unsafe {
                            kernels::fold_block_major_chunk_x86_avx512(
                                &transposed,
                                &nib_tables,
                                group,
                                chunk_bits,
                            );
                        }
                        chunk_bits
                    } else {
                        0
                    };
                    #[cfg(not(any(
                        target_arch = "aarch64",
                        all(
                            target_arch = "x86_64",
                            target_feature = "avx512f",
                            target_feature = "avx512bw"
                        )
                    )))]
                    let b_done = 0;
                    let _ = use_nibble;
                    for b in b_done..chunk_bits {
                        let mut acc = group[b];
                        for t in 0..tile_stripes {
                            let byte = transposed[t * 128 + b] as usize;
                            acc += tables[t * 256 + byte];
                        }
                        group[b] = acc;
                    }
                    if let Some(ta0) = ta0 {
                        t_acc += ta0.elapsed();
                    }
                }
            }
            if trace_fold {
                eprintln!(
                    "[lc] fold worker {worker}: total {:.2} ms tables {:.2} transpose+read {:.2} acc {:.2}",
                    wt0.elapsed().as_secs_f64() * 1e3,
                    t_tables.as_secs_f64() * 1e3,
                    t_tr.as_secs_f64() * 1e3,
                    t_acc.as_secs_f64() * 1e3
                );
            }
        });
    let probe_t2 = std::time::Instant::now();

    // Reduce the per-worker partials in ONE parallel pass over column ranges
    // (each task XORs every worker's slice of its range), instead of one
    // rayon dispatch per worker over the whole length-k vector: on a 16-way
    // pool the latter is fifteen back-to-back fork/joins over 16 K tiny
    // items and measured ~30 ms at the ranked shape — more than the sweep
    // itself. Same XORs, same association order per column (worker 0 first,
    // then 1..n_workers), so the result is bit-identical.
    let mut out = vec![F128::ZERO; k];
    if reduce_single_pass_enabled() {
        let cols_per_task = k.div_ceil(4 * p).max(64);
        out.par_chunks_mut(cols_per_task)
            .enumerate()
            .for_each(|(ti, o)| {
                let base = ti * cols_per_task;
                let len = o.len();
                o.copy_from_slice(&partials[base..base + len]);
                for w in 1..n_workers {
                    let src = &partials[w * k + base..w * k + base + len];
                    for (a, b) in o.iter_mut().zip(src) {
                        *a += *b;
                    }
                }
            });
    } else {
        let (first, rest) = partials.split_at(k);
        out.copy_from_slice(first);
        for partial in rest.chunks(k) {
            out.par_iter_mut()
                .zip(partial.par_iter())
                .for_each(|(o, p)| *o += *p);
        }
    }
    if std::env::var_os("LINCHECK_TRACE").is_some() {
        eprintln!(
            "[lc] fold alloc {:.2} ms par {:.2} ms reduce {:.2} ms",
            (probe_t1 - probe_t0).as_secs_f64() * 1e3,
            (probe_t2 - probe_t1).as_secs_f64() * 1e3,
            probe_t2.elapsed().as_secs_f64() * 1e3
        );
    }
    let one = if ranked_one_rows {
        let r = top_bind.expect("ranked one-row contribution needs top bind");
        assert_eq!(k, 1 << 14);
        let half = k / 2;
        let mut one = vec![F128::ZERO; half];
        let scale_lo = F128::ONE + r;
        for i in 0..1152 {
            one[i] = scale_lo * out[i];
        }
        for i in 15104..15360 {
            one[i - half] = r * out[i];
        }
        Some(one)
    } else {
        None
    };
    if let Some(r) = top_bind {
        let half = out.len() / 2;
        let (lo, hi) = out.split_at_mut(half);
        crate::field::f128_slice::bind_split_half(lo, hi, r);
        out.truncate(half);
    }
    (out, one)
}

/// Stripes swept per accumulator touch in the NEON tiled partial fold.
/// Larger ⇒ the length-`k` accumulator is re-streamed fewer times
/// (`n_stripes / NEON_TILE_T`), but the per-tile sum tables grow
/// `NEON_TILE_T × 4 KB` and must stay L1-resident.
const NEON_TILE_T: usize = 8;

/// Dispatch helper: pick the fastest single-matrix partial fold available
/// for the given (m, k_log). Threads `useful_bits` through so the kernel
/// can skip blocks past the useful region of each block (byte-identical to
/// the dense path on honestly-padded witnesses).
fn partial_fold_packed_z_best(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    if n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
        #[cfg(target_arch = "aarch64")]
        {
            // Pick the partition that wins for this size. The outer(tile)-partitioned
            // `oblock` builds each tile's sum-tables once instead of once per worker,
            // so it scales the fold far better (≈8.5× vs iblock's ≈6.5× on 10 P-cores
            // at m=32) — BUT its private-partial alloc + XOR-reduce overhead makes it
            // up to ~1.7× SLOWER on small folds. Empirically (M4 Max, 10 P-cores) the
            // crossover sits at n_log ≈ 15–16 across k_log ∈ {11,14}, so gate oblock at
            // n_log ≥ 16; below that the L1-resident `iblock` wins. `FOLD_IBLOCK` forces
            // iblock everywhere (bench A/B).
            let n_log = m - k_log;
            if n_log >= OBLOCK_MIN_N_LOG && !FOLD_IBLOCK.load(std::sync::atomic::Ordering::Relaxed)
            {
                return partial_fold_packed_z_neon_oblock_padded(
                    z_packed,
                    m,
                    k_log,
                    useful_bits,
                    eq_outer,
                );
            }
            partial_fold_packed_z_neon_iblock_padded(z_packed, m, k_log, useful_bits, eq_outer)
        }
        #[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
        {
            // GFNI plane fold (`FLOCK_NO_LC_GFNI=1` restores the table-gather
            // tile kernel; output is bit-identical either way).
            #[cfg(all(target_feature = "avx512f", target_feature = "gfni"))]
            if k_log >= 6 && lincheck_gfni_enabled() {
                return kernels::partial_fold_packed_z_x86_gfni_padded(
                    z_packed,
                    m,
                    k_log,
                    useful_bits,
                    eq_outer,
                );
            }
            partial_fold_packed_z_x86_tiled_padded(z_packed, m, k_log, useful_bits, eq_outer)
        }
        #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
        {
            partial_fold_packed_z_fast_padded(z_packed, m, k_log, useful_bits, eq_outer)
        }
    } else {
        partial_fold_packed_z_fast_padded(z_packed, m, k_log, useful_bits, eq_outer)
    }
}

/// GFNI stripe-fold kill switch: exactly `FLOCK_NO_LC_GFNI=1` restores the
/// table-gather tile kernel as a same-binary control.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
fn lincheck_gfni_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_LC_GFNI").as_deref() != Some(std::ffi::OsStr::new("1"))
    })
}

/// Outer-dimension threshold (`n_log = m − k_log`) at/above which the
/// outer(tile)-partitioned fold beats the i_inner-partitioned one. See
/// [`partial_fold_packed_z_best`] for the crossover calibration.
#[cfg(target_arch = "aarch64")]
const OBLOCK_MIN_N_LOG: usize = 16;

/// Quick test for "can we use the tiled fast path?". Tile uses `TILE_T`
/// stripes; we need `n_stripes` divisible by TILE_T and enough outer dim.
fn n_log_ok_for_tile(m: usize, k_log: usize, tile_t: usize) -> bool {
    if k_log < 3 {
        return false;
    }
    let n_log = m - k_log;
    if n_log < 3 + (tile_t.trailing_zeros() as usize) {
        return false;
    }
    let n_stripes = 1usize << (n_log - 3);
    n_stripes.is_multiple_of(tile_t)
}

/// Build a 256-entry sum table over 8 F128 values:
///   `table[b] = Σ_{r: bit r of b is set}  eq8[r]`
///
/// Doubling construction (255 XORs): for each new bit position `i ∈ 0..8`,
/// extend the table by XORing `eq8[i]` into each existing entry. This
/// avoids the naive 8·256 = 2048 operations.
#[inline]
fn build_sum_table(eq8: &[F128], table: &mut [F128]) {
    debug_assert_eq!(eq8.len(), 8);
    debug_assert_eq!(table.len(), 256);
    table[0] = F128::ZERO;
    for i in 0..8 {
        let e = eq8[i];
        let len = 1usize << i;
        for j in 0..len {
            table[len + j] = table[j] + e;
        }
    }
}

/// Pack a logical Boolean witness vector into the lincheck `z_packed`
/// stripe layout. The input `z_logical` is indexed linearly with
/// `z_logical[i_inner + i_outer · k]` = z's value at `(i_inner, i_outer)`.
/// The output `z_packed[byte_idx · k + i_inner]` holds 8 outer bits
/// `z[i_inner, 8·byte_idx + r]` for `r ∈ 0..8`, with bit `r` within the byte.
///
/// See the module-level docs for the full bit-position decomposition.
pub fn pack_z_lincheck(z_logical: &[bool], m: usize, k_log: usize) -> Vec<u8> {
    let k = 1usize << k_log;
    let n_total = 1usize << m;
    assert_eq!(z_logical.len(), n_total);
    let n_outer = n_total / k;
    assert_eq!(n_outer % 8, 0, "need n_outer ≥ 8 for byte stripes");
    let n_stripes = n_outer / 8;

    // Uninit alloc — every byte is written exactly once in the loop below.
    let mut z_packed: Vec<u8> = crate::alloc_uninit_vec(n_total / 8);
    for byte_idx in 0..n_stripes {
        for i_inner in 0..k {
            let mut byte = 0u8;
            for r in 0..8 {
                let i_outer = 8 * byte_idx + r;
                let logical_idx = i_inner + i_outer * k;
                if z_logical[logical_idx] {
                    byte |= 1u8 << r;
                }
            }
            z_packed[byte_idx * k + i_inner] = byte;
        }
    }
    z_packed
}

/// Same output as [`pack_z_lincheck`] but reads bits from an F_{2^128}-packed
/// witness (polynomial basis: bit `i` of logical = bit `i % 128` of
/// `z_packed_f128[i / 128]`).
pub fn pack_z_lincheck_from_packed(
    z_packed_f128: &[crate::field::F128],
    m: usize,
    k_log: usize,
) -> Vec<u8> {
    use rayon::prelude::*;
    let k = 1usize << k_log;
    let n_total = 1usize << m;
    assert_eq!(z_packed_f128.len(), n_total / 128);
    let n_outer = n_total / k;
    assert_eq!(n_outer % 8, 0, "need n_outer ≥ 8 for byte stripes");

    // Uninit alloc — the par_chunks_mut loop below writes every byte of
    // every k-byte stripe exactly once. Saves ~10 ms of sequential
    // zero-fill at m=29 (64 MB byte buffer) on the main thread.
    let mut z_packed: Vec<u8> = crate::alloc_uninit_vec(n_total / 8);
    // Each stripe (byte_idx) writes a disjoint k-byte chunk — process them in
    // parallel. Inside one stripe, k independent output bytes.
    z_packed
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(byte_idx, chunk)| {
            for i_inner in 0..k {
                let mut byte = 0u8;
                for r in 0..8 {
                    let i_outer = 8 * byte_idx + r;
                    let logical_idx = i_inner + i_outer * k;
                    let f128_idx = logical_idx / 128;
                    let local_bit = logical_idx % 128;
                    let bit = if local_bit < 64 {
                        (z_packed_f128[f128_idx].lo >> local_bit) & 1 == 1
                    } else {
                        (z_packed_f128[f128_idx].hi >> (local_bit - 64)) & 1 == 1
                    };
                    if bit {
                        byte |= 1u8 << r;
                    }
                }
                chunk[i_inner] = byte;
            }
        });
    z_packed
}

/// Build the **quirky eq table** for a claim point on the inner half:
///
///   `out[i_skip + i_inner_rest · 2^k_skip]
///     = L_{i_skip}(z_skip)  ·  eq(x_inner_rest, i_inner_rest)`
///
/// where `L_{i_skip}` are Lagrange weights at `z_skip` for the φ_8 basis
/// over `{0, …, 2^k_skip − 1}`. Length: `2^k_log`.
///
/// Encoding: the skip dim occupies the **low** `k_skip` bits of the table
/// index (matches z_packed's stripe layout / zerocheck's LSB-first
/// univariate-skip variable ordering). The `k_log − k_skip` multilinear
/// inner-rest dims occupy the next bits.
///
/// Cost: 64 (Lagrange) + 32 (eq) + 2048 outer products ≈ tiny.
pub fn build_quirky_eq_table(z_skip: F128, x_inner_rest: &[F128], k_skip: usize) -> Vec<F128> {
    let ell_skip = 1usize << k_skip;
    let ell_rest = 1usize << x_inner_rest.len();
    let lambda_skip = lagrange_weights_naive(k_skip, z_skip);
    let eq_rest = build_eq_table(x_inner_rest);
    let total = ell_skip * ell_rest;
    let mut out = Vec::with_capacity(total);
    // Layout: index = i_skip + i_inner_rest · 2^k_skip  ⇒  i_skip is low bits.
    for &er in &eq_rest {
        for &ls in &lambda_skip {
            out.push(ls * er);
        }
    }
    debug_assert_eq!(out.len(), total);
    out
}

/// Dot product of two equal-length F128 slices.
fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    assert_eq!(a.len(), b.len());
    let mut acc = F128::ZERO;
    for (x, y) in a.iter().zip(b.iter()) {
        acc += *x * *y;
    }
    acc
}

/// Length above which the inner product / element-wise kernels split via
/// rayon. Below it, sequential beats dispatch overhead.
const SUMCHECK_PAR_THRESHOLD: usize = 1usize << 12;

/// Fused `sparse_row_fold(A) + α-batch + sparse_row_fold(B)`: produces the
/// `comb_vec[c] = α · (A^T·eq)[c] + (B^T·eq)[c]` in a single pass, halving the
/// allocations and reduction phases vs. two separate sparse_row_folds + an
/// α-batch step. Both matrices must be `k × k` and `eq_table.len() == k`.
fn sparse_row_fold_alpha_batched(
    alpha: F128,
    a_0: &SparseBinaryMatrix,
    b_0: &SparseBinaryMatrix,
    eq_table: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;
    let n_cols = a_0.num_cols;
    debug_assert_eq!(b_0.num_cols, n_cols);
    debug_assert_eq!(eq_table.len(), a_0.num_rows);
    debug_assert_eq!(eq_table.len(), b_0.num_rows);

    let total_rows = a_0.num_rows + b_0.num_rows;
    if total_rows < SPARSE_ROW_FOLD_PAR_THRESHOLD {
        // Scalar fused path.
        let mut out = vec![F128::ZERO; n_cols];
        for (r, row) in a_0.rows.iter().enumerate() {
            let e = alpha * eq_table[r];
            for &c in row {
                out[c] += e;
            }
        }
        for (r, row) in b_0.rows.iter().enumerate() {
            let e = eq_table[r];
            for &c in row {
                out[c] += e;
            }
        }
        return out;
    }

    // Parallel fused path with a BOUNDED number of accumulators. These base
    // matrices are dense (e.g. BLAKE3: ~21M nonzeros over 16384 rows), so the
    // fold is ~21M F128 adds. The natural `par_iter().fold()` form spawns a
    // fresh length-`n_cols` (256 KB) accumulator per work-steal split and then
    // tree-reduces all of them — O(n_cols × num_splits) of pure overhead that
    // doesn't shrink with useful work, which capped scaling at ~1.5×. Here we
    // split the *rows* into a fixed number of contiguous chunks (rows are
    // evenly sized, so this load-balances), give each chunk one private
    // accumulator, then reduce. Overhead is O(n_cols × num_chunks) with
    // num_chunks ≈ 4× the thread count — negligible vs. the 21M-add body.
    let n_rows = a_0.num_rows;
    let p = rayon::current_num_threads().max(1);
    // ~4 chunks per worker for work-stealing balance, ≥256 rows each to keep
    // accumulator alloc/reduce overhead amortized.
    let chunk_rows = (n_rows.div_ceil(p * 4)).max(256);
    let n_chunks = n_rows.div_ceil(chunk_rows);

    let partials: Vec<Vec<F128>> = (0..n_chunks)
        .into_par_iter()
        .map(|ci| {
            let lo = ci * chunk_rows;
            let hi = ((ci + 1) * chunk_rows).min(n_rows);
            let mut acc = vec![F128::ZERO; n_cols];
            for r in lo..hi {
                let ea = alpha * eq_table[r];
                let eb = eq_table[r];
                for &c in &a_0.rows[r] {
                    acc[c] += ea;
                }
                for &c in &b_0.rows[r] {
                    acc[c] += eb;
                }
            }
            acc
        })
        .collect();

    let mut out = vec![F128::ZERO; n_cols];
    for acc in &partials {
        for i in 0..n_cols {
            out[i] += acc[i];
        }
    }
    out
}

/// One round of product-sumcheck on `(c, z)`: compute `(q(1), q(∞))` =
/// `(Σ c_hi·z_hi, Σ (c_hi+c_lo)·(z_hi+z_lo))` over the top-bit split. The
/// `len()` of `c` and `z` is even; `half = len/2`.
fn sumcheck_round_eval_par(c: &[F128], z: &[F128]) -> (F128, F128) {
    use rayon::prelude::*;
    let half = c.len() / 2;
    debug_assert_eq!(z.len(), c.len());
    let (clo, chi) = c.split_at(half);
    let (zlo, zhi) = z.split_at(half);
    if !sumcheck_x4_enabled() {
        if half < SUMCHECK_PAR_THRESHOLD {
            let mut e1 = F128::ZERO;
            let mut einf = F128::ZERO;
            for i in 0..half {
                e1 += chi[i] * zhi[i];
                einf += (chi[i] + clo[i]) * (zhi[i] + zlo[i]);
            }
            return (e1, einf);
        }
        return (0..half)
            .into_par_iter()
            .map(|i| {
                let e1_i = chi[i] * zhi[i];
                let einf_i = (chi[i] + clo[i]) * (zhi[i] + zlo[i]);
                (e1_i, einf_i)
            })
            .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1));
    }
    if half < SUMCHECK_PAR_THRESHOLD {
        return crate::field::f128_slice::msg_split_half(chi, clo, zhi, zlo, half);
    }
    // Chunked: the per-chunk sums are XORed, and XOR is associative and
    // commutative, so any chunking yields the same field element.
    let chunk = sumcheck_chunk(half);
    chi.par_chunks(chunk)
        .zip(clo.par_chunks(chunk))
        .zip(zhi.par_chunks(chunk))
        .zip(zlo.par_chunks(chunk))
        .map(|(((chi_c, clo_c), zhi_c), zlo_c)| {
            crate::field::f128_slice::msg_split_half(chi_c, clo_c, zhi_c, zlo_c, chi_c.len())
        })
        .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// Ranked default routes the lincheck product-sumcheck rounds through the
/// four-lane split-half leaves in `field::f128_slice`, with chunk-granular
/// rayon instead of one item per element. `FLOCK_NO_LC_SUMCHECK_X4=1` restores
/// the per-element scalar rounds. Read once per process; default ON.
fn sumcheck_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_SUMCHECK_X4").is_none());
    *ON
}

/// Ranked default gates the FUSED bind+eval pass's parallelism on `half =
/// len/2` — the same granularity the standalone bind and eval passes use —
/// instead of `half2 = len/4`. The incumbent `half2` comparison sent the
/// fused rounds serial at twice the table size the unfused passes would
/// have, despite the fused pass doing strictly more work per index: at the
/// ranked lincheck shape (2^14 tables) rounds 1..6 of the product-sumcheck
/// ran on one core. Width cannot change wire bytes: bound values are pure
/// per-slot functions and the message terms are XOR sums, so chunked and
/// serial evaluation give the same field elements (pinned by
/// `split_half_chunking_is_exact` and
/// `fused_bind_eval_threshold_arms_agree`).
/// `FLOCK_NO_LC_SUMCHECK_PAR_FIX=1` restores the incumbent `half2`
/// comparison for exact same-binary A/B. Read once per process; default ON.
fn sumcheck_par_fix_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LC_SUMCHECK_PAR_FIX").is_none());
    *ON
}

/// Rayon chunk width for one sumcheck round over `n` slots: enough chunks to
/// fill the pool, never finer than a whole cache line of `F128`.
fn sumcheck_chunk(n: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    n.div_ceil(threads).max(4)
}

/// Bind the top remaining variable of `v` at challenge `r`: `v[i] ← v[i] +
/// r·(v[i+half] + v[i])` for `i ∈ [0, half)`, then truncate to `half`. In-place.
fn sumcheck_bind_top_in_place_par(v: &mut Vec<F128>, r: F128) {
    use rayon::prelude::*;
    let half = v.len() / 2;
    if !sumcheck_x4_enabled() {
        if half < SUMCHECK_PAR_THRESHOLD {
            for i in 0..half {
                v[i] = v[i] + r * (v[i + half] + v[i]);
            }
        } else {
            let (lo, hi) = v.split_at_mut(half);
            let hi = &hi[..half];
            lo.par_iter_mut()
                .zip(hi.par_iter())
                .for_each(|(lo_i, &hi_i)| {
                    *lo_i = *lo_i + r * (hi_i + *lo_i);
                });
        }
        v.truncate(half);
        return;
    }
    let (lo, hi) = v.split_at_mut(half);
    let hi = &hi[..half];
    if half < SUMCHECK_PAR_THRESHOLD {
        crate::field::f128_slice::bind_split_half(lo, hi, r);
    } else {
        let chunk = sumcheck_chunk(half);
        lo.par_chunks_mut(chunk)
            .zip(hi.par_chunks(chunk))
            .for_each(|(lo_c, hi_c)| {
                crate::field::f128_slice::bind_split_half(lo_c, hi_c, r);
            });
    }
    v.truncate(half);
}

/// **Fused fold + next-round evaluation.** Binds the top variable of *both*
/// `comb` and `z` at `r` (in place, each length halves) AND returns the next
/// product-sumcheck round's message `(q(1), q(∞))` over the just-bound tables —
/// all in a single pass over the data.
///
/// Why it fuses: round `t`'s message must be sent before `r_t` is sampled, so
/// eval(t) and bind(t) can't share a pass. But binding at `r_t` produces
/// exactly the table eval(t+1) reads, and `r_t` is known by then. The bound
/// values `new[i]` and `new[i+half2]` are precisely the `lo`/`hi` halves the
/// next round's eval pairs up, so we form each product the moment both bound
/// values exist. This replaces eval + two binds (3 passes) with 1.
///
/// Operates on quarters of each array (`half2 = len/4`). For `i ∈ 0..half2`:
/// ```text
///   lo' = q0[i] + r·(q2[i] + q0[i])   (= new[i],        next round's lo)
///   hi' = q1[i] + r·(q3[i] + q1[i])   (= new[i+half2],  next round's hi)
///   q0[i] ← lo';  q1[i] ← hi'
///   e1   += hi'·zhi';   einf += (hi'+lo')·(zhi'+zlo')
/// ```
/// In-place is safe: each `i` reads its 4 quarter-entries before writing the 2
/// low-half slots, and writes across distinct `i` are disjoint. Requires
/// `comb.len() == z.len()`, a power of two ≥ 4 (so the bound length ≥ 2 has a
/// well-defined next round — the caller guarantees this by only fusing when a
/// later round exists). The returned message is bit-identical to
/// `sumcheck_round_eval_par` run on the bound tables.
fn sumcheck_bind_both_and_eval_next(
    comb: &mut Vec<F128>,
    z: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    use rayon::prelude::*;
    let len = comb.len();
    debug_assert_eq!(z.len(), len);
    let half = len / 2;
    let half2 = half / 2;
    debug_assert!(half2 >= 1, "fused step needs a well-defined next round");
    // Parallelism granularity: `half` matches the standalone bind/eval passes
    // this fusion replaced; the incumbent compared `half2` (see
    // [`sumcheck_par_fix_enabled`]). Same field elements either way — the
    // choice only picks serial vs chunked execution.
    let par_gate = if sumcheck_par_fix_enabled() {
        half
    } else {
        half2
    };

    // q0,q1 = low half (written); q2,q3 = high half (read-only).
    let (c_lo, c_hi) = comb.split_at_mut(half);
    let (cq0, cq1) = c_lo.split_at_mut(half2);
    let (cq2, cq3) = c_hi.split_at(half2);
    let (z_lo, z_hi) = z.split_at_mut(half);
    let (zq0, zq1) = z_lo.split_at_mut(half2);
    let (zq2, zq3) = z_hi.split_at(half2);

    let (e1, einf) = if sumcheck_x4_enabled() {
        if par_gate < SUMCHECK_PAR_THRESHOLD {
            crate::field::f128_slice::bind_both_and_msg_split(
                cq0, cq1, cq2, cq3, zq0, zq1, zq2, zq3, r, half2,
            )
        } else {
            // Chunked: bound values depend only on their own slot, and the
            // message terms are XOR-summed, so any chunking is exact.
            let chunk = sumcheck_chunk(half2);
            cq0.par_chunks_mut(chunk)
                .zip(cq1.par_chunks_mut(chunk))
                .zip(cq2.par_chunks(chunk))
                .zip(cq3.par_chunks(chunk))
                .zip(zq0.par_chunks_mut(chunk))
                .zip(zq1.par_chunks_mut(chunk))
                .zip(zq2.par_chunks(chunk))
                .zip(zq3.par_chunks(chunk))
                .map(|(((((((c0, c1), c2), c3), z0), z1), z2), z3)| {
                    let n = c0.len();
                    crate::field::f128_slice::bind_both_and_msg_split(
                        c0, c1, c2, c3, z0, z1, z2, z3, r, n,
                    )
                })
                .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1))
        }
    } else if par_gate < SUMCHECK_PAR_THRESHOLD {
        let mut e1 = F128::ZERO;
        let mut einf = F128::ZERO;
        for i in 0..half2 {
            let lo = cq0[i] + r * (cq2[i] + cq0[i]);
            let hi = cq1[i] + r * (cq3[i] + cq1[i]);
            let zlo = zq0[i] + r * (zq2[i] + zq0[i]);
            let zhi = zq1[i] + r * (zq3[i] + zq1[i]);
            cq0[i] = lo;
            cq1[i] = hi;
            zq0[i] = zlo;
            zq1[i] = zhi;
            e1 += hi * zhi;
            einf += (hi + lo) * (zhi + zlo);
        }
        (e1, einf)
    } else {
        cq0.par_iter_mut()
            .zip(cq1.par_iter_mut())
            .zip(cq2.par_iter())
            .zip(cq3.par_iter())
            .zip(zq0.par_iter_mut())
            .zip(zq1.par_iter_mut())
            .zip(zq2.par_iter())
            .zip(zq3.par_iter())
            .map(|(((((((c0, c1), c2), c3), z0), z1), z2), z3)| {
                let lo = *c0 + r * (*c2 + *c0);
                let hi = *c1 + r * (*c3 + *c1);
                let zlo = *z0 + r * (*z2 + *z0);
                let zhi = *z1 + r * (*z3 + *z1);
                *c0 = lo;
                *c1 = hi;
                *z0 = zlo;
                *z1 = zhi;
                (hi * zhi, (hi + lo) * (zhi + zlo))
            })
            .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1))
    };

    comb.truncate(half);
    z.truncate(half);
    (e1, einf)
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

enum PackedZ<'a> {
    LincheckStripe(&'a [u8]),
    BlockMajor(&'a [F128]),
}

// ---------------------------------------------------------------------------
// Last-ρ leftover z-fold (wait-not-join)
// ---------------------------------------------------------------------------
//
// Ranked BlockMajor path: start today's one-shot `ẑ(·, x_outer)` at the last
// zerocheck ML ρ (`x_outer = mlv[inner_rest_len..]` is complete there). The
// fold runs on a dedicated OS thread so it can own the full global rayon
// pool while main finishes serial FS (final bind, observe â/b̂, lincheck
// label, sample α, build eq_inner).
//
// When lincheck needs the pool for `fold_alpha_batched`, WAIT the handle
// first, then run CSC at full width. Do not `rayon::join` / `rayon::scope`
// the leftover fold with fold_alpha (that split is HOLD and can lose).
// Residual join stays unimplemented. Incremental 18-pass packed fold stays
// unimplemented. Compute only — no observe/sample in the kick.
//
// Kick after zc returns is a no-op (final bind already done; the prepare
// slot is consumed or never armed). Kick with a short `mlv` (rounds 2–27)
// is a no-op: `x_outer` is incomplete. Do not reuse URM `r` as `x_outer`.
// `z` is read-only (`C` aliases it).

struct LastRhoPrepared {
    z_ptr: *const F128,
    z_len: usize,
    m: usize,
    k_log: usize,
    useful_bits: usize,
    inner_rest_len: usize,
}

// SAFETY: the pointer is only read, and only while the registering thread
// keeps `z` alive and unmutated (see [`LastRhoZFoldGuard`]).
unsafe impl Send for LastRhoPrepared {}

enum LastRhoSlot {
    Empty,
    Prepared(LastRhoPrepared),
    Running(JoinHandle<Vec<F128>>),
}

thread_local! {
    static LAST_RHO: RefCell<LastRhoSlot> = const { RefCell::new(LastRhoSlot::Empty) };
}

/// Keeps packed `z` alive until a kicked leftover fold is waited. Drop
/// joins any in-flight handle so the raw pointer cannot dangle.
pub struct LastRhoZFoldGuard;

impl Drop for LastRhoZFoldGuard {
    fn drop(&mut self) {
        let _ = wait_last_rho_z_fold();
    }
}

/// Register packed `z` for a last-ρ leftover fold. Call from the ranked
/// BlockMajor prover **before** zerocheck. `z` must stay live and unmutated
/// until [`wait_last_rho_z_fold`] (or drop of the returned guard).
///
/// `inner_rest_len = k_log − k_skip` so the kick can slice
/// `x_outer = mlv[inner_rest_len..]` without using URM `r`.
pub fn prepare_last_rho_z_fold(
    z: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    inner_rest_len: usize,
) -> LastRhoZFoldGuard {
    // A previous prepare on this thread that was never waited would leave a
    // live handle holding a pointer into a now-dead buffer. Join it first.
    let _ = wait_last_rho_z_fold();
    LAST_RHO.with(|slot| {
        *slot.borrow_mut() = LastRhoSlot::Prepared(LastRhoPrepared {
            z_ptr: z.as_ptr(),
            z_len: z.len(),
            m,
            k_log,
            useful_bits,
            inner_rest_len,
        });
    });
    LastRhoZFoldGuard
}

/// Start today's one-shot z-fold. Compute only — no observe/sample.
///
/// No-op unless this thread prepared a BlockMajor fold **and** `mlv` is
/// complete (`len == inner_rest_len + n_log`). A short `mlv` means the
/// caller is still in zerocheck rounds 2–27 (`x_outer` incomplete) — that
/// kick is REJECT, so we refuse to start. A second kick, or a kick after
/// wait / after zc with no prepare, is a no-op.
pub fn kick_last_rho_z_fold(mlv: &[F128]) {
    let prepared = LAST_RHO.with(|slot| {
        let mut slot = slot.borrow_mut();
        match std::mem::replace(&mut *slot, LastRhoSlot::Empty) {
            LastRhoSlot::Prepared(p) if mlv.len() == p.inner_rest_len + (p.m - p.k_log) => Some(p),
            LastRhoSlot::Prepared(p) => {
                *slot = LastRhoSlot::Prepared(p);
                None
            }
            other => {
                *slot = other;
                None
            }
        }
    });
    let Some(p) = prepared else {
        return;
    };
    let x_outer = mlv[p.inner_rest_len..].to_vec();
    // Carry the address as usize so the spawn closure is Send without
    // relying on `*const F128: Send`.
    let z_addr = p.z_ptr as usize;
    let z_len = p.z_len;
    let m = p.m;
    let k_log = p.k_log;
    let useful_bits = p.useful_bits;
    let handle = std::thread::Builder::new()
        .name("flock-last-rho-z-fold".into())
        .spawn(move || {
            // SAFETY: [`prepare_last_rho_z_fold`] contract — `z` is live,
            // unmutated, and not aliased for writes (`C` aliases `z` but
            // the leftover fold starts after the last ρ, when C is no
            // longer written).
            let z = unsafe { std::slice::from_raw_parts(z_addr as *const F128, z_len) };
            let trace = std::env::var_os("LINCHECK_TRACE").is_some();
            let t0 = std::time::Instant::now();
            let out = fold_block_major_one_shot(z, m, k_log, useful_bits, &x_outer);
            if trace {
                eprintln!(
                    "[lc] {:<26} {:>7.2} ms",
                    "kicked z-fold (thread)",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            out
        })
        .expect("spawn last-ρ z-fold");
    LAST_RHO.with(|slot| {
        *slot.borrow_mut() = LastRhoSlot::Running(handle);
    });
}

/// Join a kicked leftover fold. Call **after** serial FS and **before**
/// `fold_alpha_batched` so CSC keeps the full pool. Returns `Some(ẑ)` if
/// a kick was running; `None` if nothing was kicked (caller runs today's
/// sequential one-shot).
pub fn wait_last_rho_z_fold() -> Option<Vec<F128>> {
    let handle = LAST_RHO.with(|slot| {
        match std::mem::replace(&mut *slot.borrow_mut(), LastRhoSlot::Empty) {
            LastRhoSlot::Running(h) => Some(h),
            LastRhoSlot::Prepared(_) | LastRhoSlot::Empty => None,
        }
    });
    handle.map(|h| h.join().expect("last-ρ z-fold thread"))
}

/// Prove the lincheck statement for the block-diagonal R1CS instance
/// `A = I_{2^n_log} ⊗ a_0`, `B = I ⊗ b_0`, `C = I ⊗ c_0`.
///
/// Preconditions:
/// - `m ≥ k_log`, `m = k_log + n_log` (caller's responsibility).
/// - `a_0, b_0, c_0` are each `k × k` where `k = 2^k_log`.
/// - `x.len() == x_prime.len() == x_pprime.len() == m`.
/// - `z_packed.len() == 2^m / 8`.
///
/// Returns `(LincheckProof, LincheckClaim)`. The claim's `r_inner` is
/// sampled from the challenger after the proof vectors are observed.
pub fn prove<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim) {
    prove_padded(
        z_packed,
        m,
        k_log,
        k_skip,
        1usize << k_log,
        circuit,
        x_ab,
        challenger,
    )
}

/// Padding-aware variant of [`prove`]. `useful_bits ≤ 2^k_log` declares how
/// many rows of each block carry real witness data; rows
/// `[useful_bits, 2^k_log)` are honest zero padding. The partial-fold over
/// the outer dimension skips work for those padding rows — byte-identical
/// proof on a witness with zero-padded blocks.
pub fn prove_padded<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim) {
    let (proof, claim, _, _) = prove_padded_inner(
        PackedZ::LincheckStripe(z_packed),
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        ZCaptureMode::None,
        challenger,
    );
    (proof, claim)
}

/// Which form of lincheck's z table the caller wants back.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ZCaptureMode {
    /// Return nothing.
    None,
    /// Return the pre-sumcheck table.
    PreSumcheck,
    /// Return the table after the first top bind. At the ranked shape this is
    /// already the Fold8 statistic consumed by the PCS opening.
    RankedFold8Tail,
}

/// Capture lincheck's z table in the requested form, reporting the form that
/// the shape actually allowed.
#[allow(clippy::too_many_arguments)]
pub fn prove_padded_capture_z_vec_mode<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    capture: ZCaptureMode,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>, ZCaptureMode) {
    assert!(
        capture != ZCaptureMode::None,
        "capture mode must not be None"
    );
    let (proof, claim, captured, actual) = prove_padded_inner(
        PackedZ::LincheckStripe(z_packed),
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        capture,
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("a capture mode must produce z_vec"),
        actual,
    )
}

/// Block-major counterpart of [`prove_padded_capture_z_vec_mode`].
#[allow(clippy::too_many_arguments)]
pub fn prove_padded_capture_z_vec_block_major_mode<Ch: Challenger>(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    capture: ZCaptureMode,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>, ZCaptureMode) {
    assert!(
        capture != ZCaptureMode::None,
        "capture mode must not be None"
    );
    let (proof, claim, captured, actual) = prove_padded_inner(
        PackedZ::BlockMajor(z_packed),
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        capture,
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("a capture mode must produce z_vec"),
        actual,
    )
}

/// Variant of [`prove_padded`] that also returns the **pre-sumcheck** z_vec
/// (`output[i_inner] = ẑ(i_inner, x_ab.x_outer)`, length `2^k_log`). The
/// downstream PCS reuses this vector to compute the AB-claim's ring-switch
/// `s_hat_v` via [`crate::pcs::ring_switch::s_hat_v_from_z_vec`], skipping a
/// `fold_1b_rows` pass at open time.
///
/// Pays one extra `2^k_log` F128 clone (~2 MB at k_log=17) before the
/// sumcheck loop; callers that don't need the reuse should keep using
/// [`prove_padded`] to avoid that clone.
pub fn prove_padded_capture_z_vec<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    let (proof, claim, captured, _actual) = prove_padded_inner(
        PackedZ::LincheckStripe(z_packed),
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        ZCaptureMode::PreSumcheck,
        challenger,
    );
    (proof, claim, captured.expect("capture must produce z_vec"))
}

/// Direct block-major counterpart of [`prove_padded_capture_z_vec`]. The
/// canonical F128-packed witness is folded after `x_ab.x_outer` is known, so
/// callers do not need to allocate or populate a lincheck byte stripe.
#[allow(clippy::too_many_arguments)]
pub fn prove_padded_capture_z_vec_block_major<Ch: Challenger>(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    let (proof, claim, captured, _actual) = prove_padded_inner(
        PackedZ::BlockMajor(z_packed),
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        ZCaptureMode::PreSumcheck,
        challenger,
    );
    (proof, claim, captured.expect("capture must produce z_vec"))
}

#[allow(clippy::too_many_arguments)]
fn prove_padded_inner<Ch: Challenger>(
    z_packed: PackedZ<'_>,
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    capture: ZCaptureMode,
    challenger: &mut Ch,
) -> (
    LincheckProof,
    LincheckClaim,
    Option<Vec<F128>>,
    ZCaptureMode,
) {
    let k = 1usize << k_log;
    let n_log = m - k_log;
    assert!(m >= k_log);
    assert!(k_skip <= k_log, "k_skip must be ≤ k_log");
    assert!(useful_bits <= k, "useful_bits ({useful_bits}) > k ({k})");
    let inner_rest_len = k_log - k_skip;
    assert_eq!(circuit.n_cols(), k);
    assert_eq!(x_ab.x_inner_rest.len(), inner_rest_len);
    assert_eq!(x_ab.x_outer.len(), n_log);

    challenger.observe_label(b"flock-lincheck-v0");
    let trace = std::env::var("LINCHECK_TRACE").is_ok();

    // 1. Sample α (matches verifier's order). Used to batch the two scalar
    //    consistency checks v_a, v_b into a single sumcheck.
    let alpha = challenger.sample_f128();

    // 2. Build the α-batched comb_vec via the circuit's per-block fold. For
    //    the sparse-matrix default this is the fused single-pass row-fold;
    //    per-hash circuit walkers compute the same `comb_vec` directly from
    //    the constraint graph.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "build_quirky_eq",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // Overlap-not-park: the kicked z-fold is a pure DRAM stream running on
    // its own OS thread against the global pool, while `fold_alpha_batched`
    // is a load-latency-bound sparse gather into an L2-resident eq table —
    // complementary profiles that want each other's stall slots. Run the
    // gather NOW and join the fold only where its output is first consumed
    // (the z_vec match below). fold_alpha never touches z, and α/β keep
    // their transcript positions, so the proof bytes are identical. This is
    // NOT the rayon::join-with-CSC pattern the earlier HOLD comment warns
    // about (that pulled this thread into the fold and split the pool); the
    // caller merely queues a second pool-wide job instead of parking.
    // `FLOCK_NO_LC_ALPHA_OVERLAP=1` restores the park-first order.
    let overlap = lc_alpha_overlap_enabled();
    let mut kicked_z_vec = None;
    if !overlap {
        let t_wait = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };
        kicked_z_vec = wait_last_rho_z_fold();
        if let Some(t) = t_wait {
            eprintln!(
                "[lc] {:<26} {:>7.2} ms",
                "wait kicked z-fold",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut comb_vec = circuit.fold_alpha_batched(alpha, &eq_inner);
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "fold_alpha_batched",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    if overlap {
        let t_wait = if trace {
            Some(std::time::Instant::now())
        } else {
            None
        };
        kicked_z_vec = wait_last_rho_z_fold();
        if let Some(t) = t_wait {
            eprintln!(
                "[lc] {:<26} {:>7.2} ms",
                "wait kicked z-fold (overlapped)",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    // 2b. Constant-wire pin. Fold β·eq(j*, ·) into the comb so the same sumcheck
    //     also proves z_vec[j*] = 1 (the all-ones constant column). Since j* is a
    //     boolean index, eq(j*, ·) is the one-hot vector and this is a single
    //     entry update. β is sampled after α; the verifier mirrors both. See
    //     docs/const-wire-pin.md.
    if let Some(col) = circuit.const_pin_col() {
        let beta = challenger.sample_f128();
        comb_vec[col] += beta;
    }

    // 3. Partial fold of z at the shared outer half (length-k F128 vector).
    //    A last-ρ kick already produced the same one-shot ẑ; reuse it.
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut z_vec = match (kicked_z_vec, z_packed) {
        (Some(z), PackedZ::BlockMajor(_)) => z,
        (kicked, z_packed) => {
            debug_assert!(
                kicked.is_none(),
                "last-ρ kick is BlockMajor-only; stripe path must not be prepared"
            );
            match z_packed {
                PackedZ::LincheckStripe(z) => {
                    let eq_x_outer = build_eq_table(&x_ab.x_outer);
                    partial_fold_packed_z_best(z, m, k_log, useful_bits, &eq_x_outer)
                }
                PackedZ::BlockMajor(z) => {
                    fold_block_major_one_shot(z, m, k_log, useful_bits, &x_ab.x_outer)
                }
            }
        }
    };
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "partial_fold_z",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    // At the ranked shape, the first lincheck bind and the downstream Fold8
    // map use the same halves and challenge. Capture that output directly.
    let fold8_ready = capture == ZCaptureMode::RankedFold8Tail
        && inner_rest_len == 8
        && z_vec.len() == (1usize << crate::pcs::LOG_PACKING) << 7;
    let actual_capture = if fold8_ready {
        ZCaptureMode::RankedFold8Tail
    } else if capture == ZCaptureMode::None {
        ZCaptureMode::None
    } else {
        ZCaptureMode::PreSumcheck
    };
    let mut captured_z_vec = if capture == ZCaptureMode::None || fold8_ready {
        None
    } else {
        Some(z_vec.clone())
    };
    let t_sumcheck_start = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // 5. Standard multilinear product-sumcheck over the high `inner_rest_len`
    //    bits of `i`. Each round binds the TOP remaining bit (mirrors
    //    chain::prove_chain_shift). After `inner_rest_len` rounds, both
    //    tables collapse to length `2^k_skip`. Per-round work is parallel via
    //    rayon when the residual table is large enough.
    let mut rounds = Vec::with_capacity(inner_rest_len);
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    if inner_rest_len > 0 {
        // Round 0's message is the only standalone evaluation pass; every later
        // round's message falls out of binding the previous round (fold +
        // next-eval fused into one pass — see `sumcheck_bind_both_and_eval_next`).
        let (mut e1, mut einf) = sumcheck_round_eval_par(&comb_vec, &z_vec);
        for t in 0..inner_rest_len {
            challenger.observe_f128(e1);
            challenger.observe_f128(einf);
            let r = challenger.sample_f128();
            rounds.push((e1, einf));
            r_rounds.push(r);
            if t + 1 < inner_rest_len {
                // Fused: bind both tables at r AND compute round (t+1)'s message.
                let (ne1, neinf) = sumcheck_bind_both_and_eval_next(&mut comb_vec, &mut z_vec, r);
                e1 = ne1;
                einf = neinf;
            } else {
                // Final round: just fold; z_vec collapses to z_partial.
                sumcheck_bind_top_in_place_par(&mut comb_vec, r);
                sumcheck_bind_top_in_place_par(&mut z_vec, r);
            }
            if fold8_ready && t == 0 {
                debug_assert_eq!(z_vec.len(), 64usize << crate::pcs::LOG_PACKING);
                captured_z_vec = Some(z_vec.clone());
            }
        }
    }
    if let Some(t) = t_sumcheck_start {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "sumcheck (all rounds)",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    // 6. Send `z_partial` (the post-sumcheck collapsed z_vec). Length 2^k_skip.
    let z_partial = z_vec.clone();
    challenger.observe_f128_slice(&z_partial);

    // 7. Sample fresh z_skip AFTER observing z_partial — gives Schwartz-Zippel
    //    soundness on the φ8 (univariate-skip) dim.
    let r_inner_skip = challenger.sample_f128();

    // 8. Output claim's value: φ8 Lagrange combination of z_partial at z_skip.
    //    Equals ẑ_φ8(z_skip, r_rest, x_outer) when z_partial is honest; the
    //    PCS catches mismatches downstream.
    let lambda = lagrange_weights_naive(k_skip, r_inner_skip);
    let w = inner_product(&lambda, &z_partial);

    // 9. Convert sumcheck challenges to LSB-first `x_inner_rest` order. The
    //    loop binds the TOP bit each round, so r_rounds[0] bound bit
    //    (inner_rest_len − 1) of the i_rest part (= bit (k_log − 1) of i).
    //    LSB-first: x_inner_rest[j] binds bit (k_skip + j) of i — i.e.,
    //    r_inner_rest[j] = r_rounds[inner_rest_len − 1 − j].
    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();

    let proof = LincheckProof { rounds, z_partial };
    let claim = LincheckClaim {
        r_inner_skip,
        r_inner_rest,
        w,
    };
    (proof, claim, captured_z_vec, actual_capture)
}

/// Verify a lincheck proof. Walks the challenger in lockstep with `prove`,
/// performs the three scalar consistency checks against `v, v', v''`, and
/// derives the three output z claims.
pub fn verify<Ch: Challenger>(
    m: usize,
    k_log: usize,
    k_skip: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    challenger: &mut Ch,
) -> Result<LincheckClaim, VerifyError> {
    let k = 1usize << k_log;
    let n_log = m - k_log;

    if k_skip > k_log {
        return Err(VerifyError::KSkipExceedsKLog { k_skip, k_log });
    }
    let inner_rest_len = k_log - k_skip;
    let n_skip = 1usize << k_skip;

    if x_ab.x_inner_rest.len() != inner_rest_len {
        return Err(VerifyError::BadInnerRestLength {
            which: "x_ab",
            expected: inner_rest_len,
            got: x_ab.x_inner_rest.len(),
        });
    }
    if x_ab.x_outer.len() != n_log {
        return Err(VerifyError::BadOuterLength {
            which: "x_ab",
            expected: n_log,
            got: x_ab.x_outer.len(),
        });
    }
    if circuit.n_cols() != k {
        return Err(VerifyError::BadMatrixShape {
            which: "circuit",
            expected: k,
            got_rows: k,
            got_cols: circuit.n_cols(),
        });
    }
    if proof.rounds.len() != inner_rest_len {
        return Err(VerifyError::BadVectorLength {
            which: "rounds",
            expected: inner_rest_len,
            got: proof.rounds.len(),
        });
    }
    if proof.z_partial.len() != n_skip {
        return Err(VerifyError::BadVectorLength {
            which: "z_partial",
            expected: n_skip,
            got: proof.z_partial.len(),
        });
    }

    challenger.observe_label(b"flock-lincheck-v0");

    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    // 1. Sample α (matches prover's order).
    let alpha = challenger.sample_f128();

    // 2. Build α-batched comb_vec via the circuit's per-block fold (same call
    //    the prover made — sparse default delegates to the fused row-fold;
    //    per-hash impls walk the constraint graph directly).
    let t = std::time::Instant::now();
    let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
    if trace {
        eprintln!(
            "        [lcv] build_quirky_eq_table (2^{k_log}): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }
    let t = std::time::Instant::now();
    let mut comb_vec = circuit.fold_alpha_batched(alpha, &eq_inner);
    if trace {
        eprintln!(
            "        [lcv] circuit.fold_alpha_batched: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // 3. Replay the multilinear product-sumcheck (inner_rest_len rounds),
    //    folding comb_vec in lockstep so we end up with the "comb_partial"
    //    vector of length 2^k_skip. Parallel fold for the early (large) rounds.
    let t = std::time::Instant::now();
    // Constant-wire pin (mirror of prove): β sampled after α, comb gains +β at
    // the constant column, and the initial target gains +β·1 — the honest
    // all-ones constant column folds to 1. See docs/const-wire-pin.md.
    let mut target = alpha * v_a + v_b;
    if let Some(col) = circuit.const_pin_col() {
        let beta = challenger.sample_f128();
        comb_vec[col] += beta;
        target += beta;
    }
    let mut running = target;
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    for &(e1, einf) in &proof.rounds {
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = challenger.sample_f128();
        // q(0) = claim + q(1) in char 2; q(X) = einf·X² + c1·X + e0.
        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;
        // Fold comb_vec at the same r (mirrors prover's fold).
        sumcheck_bind_top_in_place_par(&mut comb_vec, r);
        r_rounds.push(r);
    }
    debug_assert_eq!(comb_vec.len(), n_skip);
    if trace {
        eprintln!(
            "        [lcv] sumcheck replay + comb_vec fold ({} rounds): {}",
            inner_rest_len,
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // 4. Observe z_partial AFTER the sumcheck rounds (matches prover order).
    challenger.observe_f128_slice(&proof.z_partial);

    // 5. Final sumcheck consistency: Σ comb_partial[i_skip] · z_partial[i_skip]
    //    must equal the running claim. Ties z_partial to the upstream v_a, v_b.
    //    Small (length 2^k_skip = 64); sequential.
    let final_sum = inner_product(&comb_vec, &proof.z_partial);
    if running != final_sum {
        return Err(VerifyError::ConsistencyFailed {
            which: "sumcheck-final",
        });
    }

    // 6. Sample fresh z_skip AFTER z_partial — gives SZ on the φ8 dim.
    let r_inner_skip = challenger.sample_f128();

    // 7. Derive output claim value via φ8 Lagrange on z_partial at z_skip.
    //    Equals ẑ_φ8(z_skip, r_rest, x_outer) when z_partial is honest;
    //    PCS catches mismatches downstream.
    let t = std::time::Instant::now();
    let lambda = lagrange_weights_naive(k_skip, r_inner_skip);
    let w = inner_product(&lambda, &proof.z_partial);
    if trace {
        eprintln!(
            "        [lcv] final consistency + lagrange_weights_naive: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    // 8. Convert sumcheck challenges to LSB-first x_inner_rest order
    //    (same convention as prover).
    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();

    Ok(LincheckClaim {
        r_inner_skip,
        r_inner_rest,
        w,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

    /// On the ranked eight-round shape, the first top bind is precisely the
    /// incumbent Fold8 intake's only non-retained-coordinate fold. This is
    /// the equivalence that makes `RankedFold8Tail` safe to hand directly to
    /// the PCS opening.
    #[test]
    fn ranked_fold8_first_bind_matches_incumbent_intake() {
        let mut rng = Rng::new(0xF018_C4A7_0BE5); // deterministic, nonzero seed
        let n_packed = 1usize << crate::pcs::LOG_PACKING;
        let mut z_vec: Vec<F128> = (0..128 * n_packed).map(|_| rng.f128()).collect();
        let pre_sumcheck = z_vec.clone();
        let top = rng.f128();
        let mut inner_rest_tail = rng.f128_vec(7);
        inner_rest_tail[6] = top;

        let incumbent =
            crate::pcs::ring_switch::s_hat_v_fold8_from_z_vec(&pre_sumcheck, &inner_rest_tail);
        sumcheck_bind_top_in_place_par(&mut z_vec, top);

        assert_eq!(z_vec.len(), 64 * n_packed);
        assert_eq!(
            z_vec, incumbent,
            "first top bind must be the exact ranked Fold8 intake"
        );
    }

    /// The fused bind+eval must equal the unfused bind-bind-eval sequence —
    /// bound tables AND next message — at sizes on BOTH sides of the parallel
    /// threshold, including `len = 2^13` (serial under the incumbent `half2`
    /// comparison, parallel under the `half` fix) and `2^14` (parallel under
    /// both). Negative control: one corrupted table slot must change the
    /// message.
    #[test]
    fn fused_bind_eval_threshold_arms_agree() {
        let mut rng = Rng::new(0xF05E_D9A7);
        for &len in &[1usize << 12, 1 << 13, 1 << 14] {
            let comb: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
            let z: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
            let r = rng.f128();

            // Oracle: standalone binds, then the standalone eval.
            let mut want_comb = comb.clone();
            let mut want_z = z.clone();
            sumcheck_bind_top_in_place_par(&mut want_comb, r);
            sumcheck_bind_top_in_place_par(&mut want_z, r);
            let want_msg = sumcheck_round_eval_par(&want_comb, &want_z);

            let mut got_comb = comb.clone();
            let mut got_z = z.clone();
            let got_msg = sumcheck_bind_both_and_eval_next(&mut got_comb, &mut got_z, r);
            assert_eq!(got_msg, want_msg, "fused message len={len}");
            assert_eq!(got_comb, want_comb, "fused comb table len={len}");
            assert_eq!(got_z, want_z, "fused z table len={len}");

            // Negative control: one corrupted slot must not still match.
            let mut bad_comb = comb.clone();
            bad_comb[len / 3] += F128::ONE;
            let mut bad_z = z.clone();
            let bad_msg = sumcheck_bind_both_and_eval_next(&mut bad_comb, &mut bad_z, r);
            assert_ne!(
                bad_msg, want_msg,
                "corrupted table went undetected len={len}"
            );
        }
    }

    /// SplitMix64 PRNG, deterministic.
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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
    }

    /// Naive MLE evaluation: `f̂(point) = Σ_i eq(point, i) · f[i]` where i ∈
    /// {0,1}^d and f[i] is given as a bool slice.
    fn mle_eval_bool(f: &[bool], point: &[F128]) -> F128 {
        let d = point.len();
        assert_eq!(f.len(), 1 << d);
        let eq = build_eq_table(point);
        let mut acc = F128::ZERO;
        for (i, &b) in f.iter().enumerate() {
            if b {
                acc += eq[i];
            }
        }
        acc
    }

    /// Sample a random `QuirkyPoint` for testing: z_skip ∈ F₁₂₈,
    /// x_inner_rest of length `k_log − k_skip`, x_outer of length `n_log`.
    fn random_quirky_point(m: usize, k_log: usize, k_skip: usize, rng: &mut Rng) -> QuirkyPoint {
        QuirkyPoint {
            z_skip: rng.f128(),
            x_inner_rest: rng.f128_vec(k_log - k_skip),
            x_outer: rng.f128_vec(m - k_log),
        }
    }

    /// "Quirky MLE evaluation" of a Boolean vector `f` at a quirky point.
    ///
    /// `ã(z_skip, x_inner_rest, x_outer) = Σ_i  f[i] · L_{i_skip}(z_skip)
    ///                                          · eq(x_inner_rest, i_inner_rest)
    ///                                          · eq(x_outer, i_outer)`
    ///
    /// where `i = i_skip + 2^k_skip · i_inner_rest + 2^k_log · i_outer` (matches
    /// the linear-LSB indexing of `f`).
    fn mle_eval_bool_quirky(
        f: &[bool],
        m: usize,
        k_log: usize,
        k_skip: usize,
        point: &QuirkyPoint,
    ) -> F128 {
        let k_skip_dim = 1usize << k_skip;
        let inner_rest_len = k_log - k_skip;
        let inner_rest_dim = 1usize << inner_rest_len;
        let k = 1usize << k_log;
        let n_outer = 1usize << (m - k_log);
        assert_eq!(f.len(), 1 << m);

        let lambda = crate::zerocheck::multilinear::lagrange_weights_naive(k_skip, point.z_skip);
        let eq_rest = build_eq_table(&point.x_inner_rest);
        let eq_outer = build_eq_table(&point.x_outer);
        debug_assert_eq!(lambda.len(), k_skip_dim);
        debug_assert_eq!(eq_rest.len(), inner_rest_dim);
        debug_assert_eq!(eq_outer.len(), n_outer);

        let mut acc = F128::ZERO;
        for i in 0..(1 << m) {
            if !f[i] {
                continue;
            }
            let i_skip = i & (k_skip_dim - 1);
            let i_inner_rest = (i >> k_skip) & (inner_rest_dim - 1);
            let i_outer = i / k;
            acc += lambda[i_skip] * eq_rest[i_inner_rest] * eq_outer[i_outer];
        }
        acc
    }

    /// Naive sparse matrix · bool-vector product: `out[i] = ⊕_{j: M[i,j]=1} z[j]`.
    fn matrix_vector_product(m: &SparseBinaryMatrix, z: &[bool]) -> Vec<bool> {
        assert_eq!(z.len(), m.num_cols);
        m.rows
            .iter()
            .map(|row| {
                let mut acc = false;
                for &col in row {
                    acc ^= z[col];
                }
                acc
            })
            .collect()
    }

    /// Build a block-diagonal full witness vector from a base matrix and the
    /// outer dimension: full[i_inner + i_outer · k] for the i_outer-th block.
    /// Used to construct `a = (I_{2^n_log} ⊗ A_0) · z` directly for tests.
    fn apply_block_diag(m_0: &SparseBinaryMatrix, z: &[bool], k_log: usize) -> Vec<bool> {
        let k = 1usize << k_log;
        assert_eq!(m_0.num_rows, k);
        assert_eq!(m_0.num_cols, k);
        assert_eq!(z.len() % k, 0);
        let n_outer = z.len() / k;
        let mut out = vec![false; z.len()];
        for i_outer in 0..n_outer {
            let z_block = &z[i_outer * k..(i_outer + 1) * k];
            let a_block = matrix_vector_product(m_0, z_block);
            out[i_outer * k..(i_outer + 1) * k].copy_from_slice(&a_block);
        }
        out
    }

    /// Build a sparse boolean matrix with `nnz` random nonzero entries among
    /// `k × k` slots. Used for tests.
    fn random_sparse_matrix(k: usize, nnz: usize, rng: &mut Rng) -> SparseBinaryMatrix {
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut seen = std::collections::HashSet::new();
        let mut count = 0;
        while count < nnz {
            let r = (rng.next_u64() as usize) % k;
            let c = (rng.next_u64() as usize) % k;
            if seen.insert((r, c)) {
                rows[r].push(c);
                count += 1;
            }
        }
        for row in &mut rows {
            row.sort();
        }
        SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows,
        }
    }

    // ---- Unit tests for the kernels ----

    /// The u16-narrowed CSC row indices produce exactly the same
    /// `fold_alpha_batched` output as the u32 arrays (oracle A/B for
    /// `FLOCK_NO_LC_CSC_U16`), for both the sequential and the rayon branch.
    #[test]
    fn csc_u16_rows_match_u32_rows() {
        for &(k, nnz) in &[(64usize, 500usize), (1 << 12, 40_000), (1 << 13, 90_000)] {
            let mut rng = Rng::new(0xC5C1_6000 ^ k as u64);
            let a = random_sparse_matrix(k, nnz, &mut rng);
            let b = random_sparse_matrix(k, nnz, &mut rng);
            let wide = CscCircuit::from_matrices_narrow(&a, &b, false);
            let narrow = CscCircuit::from_matrices_narrow(&a, &b, true);
            assert!(!wide.narrow);
            assert!(narrow.narrow);
            let alpha = rng.f128();
            let eq: Vec<F128> = (0..k).map(|_| rng.f128()).collect();
            let want = wide.fold_alpha_batched(alpha, &eq);
            let got = narrow.fold_alpha_batched(alpha, &eq);
            assert_eq!(want, got, "k={k}");
            // …and both agree with the row-scatter reference.
            let reference = sparse_row_fold_alpha_batched(alpha, &a, &b, &eq);
            assert_eq!(want, reference, "k={k} reference");
        }
    }

    /// `build_eq_table` produces eq(point, i) for all boolean i.
    #[test]
    fn eq_table_matches_direct_formula() {
        for &d in &[1usize, 2, 3, 5, 8] {
            let mut rng = Rng::new(11 + d as u64);
            let point = rng.f128_vec(d);
            let table = build_eq_table(&point);
            assert_eq!(table.len(), 1 << d);
            for i in 0..(1 << d) {
                let mut expected = F128::ONE;
                for j in 0..d {
                    let bit = ((i >> j) & 1) as u64;
                    // eq(r, bit) = (1 + r) if bit = 0 else r
                    let factor = if bit == 0 {
                        F128::ONE + point[j]
                    } else {
                        point[j]
                    };
                    expected *= factor;
                }
                assert_eq!(table[i], expected, "mismatch at d={d}, i={i}");
            }
        }
    }

    /// `sparse_row_fold` matches a brute-force dense implementation.
    #[test]
    fn sparse_row_fold_matches_dense() {
        let mut rng = Rng::new(22);
        let k = 16;
        let nnz = 40;
        let matrix = random_sparse_matrix(k, nnz, &mut rng);
        let eq_table: Vec<F128> = rng.f128_vec(k);

        let got = sparse_row_fold(&matrix, &eq_table);

        // Brute force: for each col j, sum eq[i] over rows i where M[i,j] = 1.
        let mut expected = vec![F128::ZERO; k];
        for (i, row) in matrix.rows.iter().enumerate() {
            for &j in row {
                expected[j] += eq_table[i];
            }
        }
        assert_eq!(got, expected);
    }

    /// `partial_fold_packed_z` matches the direct sum.
    #[test]
    fn partial_fold_matches_direct() {
        for &(m, k_log) in &[(10usize, 3), (12, 4), (14, 5), (16, 8)] {
            let mut rng = Rng::new(33 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let outer_point = rng.f128_vec(n_log);
            let eq_outer = build_eq_table(&outer_point);

            let got = partial_fold_packed_z(&z_packed, m, k_log, &eq_outer);

            let k = 1usize << k_log;
            assert_eq!(got.len(), k);
            for i_inner in 0..k {
                let mut acc = F128::ZERO;
                for i_outer in 0..(1usize << n_log) {
                    let i = i_inner + i_outer * k;
                    if z[i] {
                        acc += eq_outer[i_outer];
                    }
                }
                assert_eq!(got[i_inner], acc, "mismatch at m={m}, i_inner={i_inner}");
            }
        }
    }

    /// `partial_fold_packed_z_fast` (parallel lookup-table) matches the scalar
    /// reference `partial_fold_packed_z`.
    #[test]
    fn partial_fold_fast_matches_serial() {
        for &(m, k_log) in &[(10usize, 3), (12, 4), (14, 5), (16, 8), (18, 10)] {
            let mut rng = Rng::new(800 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let p = rng.f128_vec(n_log);
            let eq = build_eq_table(&p);

            let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
            let fast = partial_fold_packed_z_fast(&z_packed, m, k_log, &eq);
            assert_eq!(serial, fast, "at m={m}, k_log={k_log}");
        }
    }

    /// The direct block-major F128 fold is exactly the existing stripe fold,
    /// including padded, non-128-aligned useful regions and partial outer
    /// tiles.
    /// The AVX-512 nibble-table accumulate must reproduce the scalar
    /// 256-entry byte-table loop bit-for-bit for every column count
    /// 1..=128 (exercises the masked tail store) on random weights, random
    /// index bytes and a random pre-filled `partial`.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn block_major_nibble_kernel_matches_scalar_tables() {
        let mut rng = Rng::new(0x51B_B1E5);
        for chunk_bits in (1..=128).chain([8, 16, 120, 121, 127, 128]) {
            let eq8s: Vec<[F128; 8]> = (0..DIRECT_FOLD_TILE_STRIPES)
                .map(|_| std::array::from_fn(|_| rng.f128()))
                .collect();
            let mut tables = vec![F128::ZERO; DIRECT_FOLD_TILE_STRIPES * 256];
            let mut nib = [[0u64; 64]; DIRECT_FOLD_TILE_STRIPES];
            for t in 0..DIRECT_FOLD_TILE_STRIPES {
                build_sum_table(&eq8s[t], &mut tables[t * 256..(t + 1) * 256]);
                kernels::build_nibble_tables(&eq8s[t], &mut nib[t]);
            }
            let transposed: Vec<u8> = (0..DIRECT_FOLD_TILE_STRIPES * 128)
                .map(|_| (rng.f128().lo & 0xFF) as u8)
                .collect();
            let base: Vec<F128> = (0..chunk_bits).map(|_| rng.f128()).collect();
            // scalar reference
            let mut want = base.clone();
            for b in 0..chunk_bits {
                let mut acc = want[b];
                for t in 0..DIRECT_FOLD_TILE_STRIPES {
                    acc += tables[t * 256 + transposed[t * 128 + b] as usize];
                }
                want[b] = acc;
            }
            // kernel, with a poison guard word past the end
            let mut got = base.clone();
            got.push(F128 {
                lo: 0xDEAD_BEEF,
                hi: 0xFEED_FACE,
            });
            unsafe {
                kernels::fold_block_major_chunk_x86_avx512(
                    &transposed,
                    &nib,
                    &mut got[..chunk_bits],
                    chunk_bits,
                );
            }
            assert_eq!(
                got.pop(),
                Some(F128 {
                    lo: 0xDEAD_BEEF,
                    hi: 0xFEED_FACE
                })
            );
            assert_eq!(got, want, "chunk_bits={chunk_bits}");
        }
    }

    #[test]
    fn partial_fold_block_major_matches_stripe() {
        let cases: &[(usize, usize, usize)] = &[
            (10, 7, 1 << 7),
            (13, 8, 233),
            (16, 10, 997),
            (16, 8, 241),
            (18, 12, 3_801),
            // The ranked k_log and useful_bits at a small n_log: 121 full
            // 128-column chunks plus the 49-bit tail block, exact tiles.
            (20, 14, 15_409),
        ];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xD1EC_7F01 + (m * 31 + k_log) as u64);
            let k = 1usize << k_log;
            let n_outer = 1usize << (m - k_log);
            let mut z = rng.bits(1usize << m);
            for outer in 0..n_outer {
                z[outer * k + useful_bits..(outer + 1) * k].fill(false);
            }

            let z_block_major: Vec<F128> = z
                .chunks_exact(128)
                .map(|bits| {
                    let mut packed = F128::ZERO;
                    for (b, &set) in bits.iter().enumerate() {
                        if set {
                            if b < 64 {
                                packed.lo |= 1u64 << b;
                            } else {
                                packed.hi |= 1u64 << (b - 64);
                            }
                        }
                    }
                    packed
                })
                .collect();
            let z_stripe = pack_z_lincheck(&z, m, k_log);
            let eq_outer = build_eq_table(&rng.f128_vec(m - k_log));

            let want = partial_fold_packed_z_best(&z_stripe, m, k_log, useful_bits, &eq_outer);
            let got = partial_fold_packed_z_block_major_padded(
                &z_block_major,
                m,
                k_log,
                useful_bits,
                &eq_outer,
            );
            assert_eq!(want, got, "m={m} k_log={k_log} useful={useful_bits}");

            // Same call with the GFNI plane arm forced off: the nibble/scalar
            // sweep must produce the identical vector (arm-vs-arm identity in
            // one process; the env switches resolve once so a latch is the
            // only way to cover both).
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "gfni"
            ))]
            {
                BM_GFNI_FORCED_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
                let got_off = partial_fold_packed_z_block_major_padded(
                    &z_block_major,
                    m,
                    k_log,
                    useful_bits,
                    &eq_outer,
                );
                BM_GFNI_FORCED_OFF.store(false, std::sync::atomic::Ordering::Relaxed);
                assert_eq!(
                    want, got_off,
                    "forced-off arm m={m} k_log={k_log} useful={useful_bits}"
                );
            }
        }
    }

    /// Scalar oracle for the exact `VPERMT2B` selector semantics used by the
    /// fused gather/transpose. This test runs without AVX-512: it independently
    /// models bits 0..=5 as the byte offset, bit 6 as the table selector, and
    /// bit 7 as ignored. It covers both limbs, random tables, selector
    /// boundaries, poisoned source tables, and four independent columns with
    /// poisoned output-stride gaps.
    #[test]
    fn gather_transpose_vpermt2b_selector_oracle() {
        fn permute2(a: &[u8; 64], indices: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
            std::array::from_fn(|i| {
                let index = indices[i];
                let table = if index & 0x40 == 0 { a } else { b };
                table[(index & 0x3f) as usize]
            })
        }

        fn direct(a: &[u8; 64], b: &[u8; 64], high: bool) -> [u8; 64] {
            std::array::from_fn(|i| {
                let row = 7 - i % 8;
                let offset = 16 * (row % 4) + usize::from(high) * 8 + i / 8;
                if row < 4 { a[offset] } else { b[offset] }
            })
        }

        let lo = gather_transpose_vpermt2b_indices(false);
        let hi = gather_transpose_vpermt2b_indices(true);
        assert_eq!(lo[3], 64, "first byte selected from the second table");
        assert_eq!(hi[56], 127, "largest legal 512-bit VPERMT2B index");
        for (high, indices) in [(false, &lo), (true, &hi)] {
            assert!(indices.iter().all(|&index| index & 0x80 == 0));
            assert_eq!(
                indices.iter().filter(|&&index| index & 0x40 != 0).count(),
                32
            );
            for (i, &index) in indices.iter().enumerate() {
                let row = 7 - i % 8;
                let want_table = if row < 4 { 0 } else { 0x40 };
                let want_offset = 16 * (row % 4) + usize::from(high) * 8 + i / 8;
                assert_eq!(index & 0x40, want_table, "half={high} i={i} row={row}");
                assert_eq!(
                    index & 0x3f,
                    want_offset as u8,
                    "half={high} i={i} row={row}"
                );
            }
        }

        // Source poison catches the exact #1445 failure: 128+offset leaves
        // selector bit 6 clear and incorrectly reads `a` for rows 4..8.
        let a_poison = [0xA5; 64];
        let b_poison = [0x5A; 64];
        for (high, indices) in [(false, &lo), (true, &hi)] {
            let want = direct(&a_poison, &b_poison, high);
            assert_eq!(permute2(&a_poison, indices, &b_poison), want);
            let mut broken = *indices;
            for (i, index) in broken.iter_mut().enumerate() {
                if 7 - i % 8 >= 4 {
                    *index += 64; // the rejected candidate's 128+offset form
                }
            }
            assert_ne!(permute2(&a_poison, &broken, &b_poison), want);
        }

        let mut rng = Rng::new(0x5650_4552_4D54_3242);
        for _ in 0..64 {
            let a: [u8; 64] = std::array::from_fn(|_| rng.next_u64() as u8);
            let b: [u8; 64] = std::array::from_fn(|_| rng.next_u64() as u8);
            assert_eq!(permute2(&a, &lo, &b), direct(&a, &b, false));
            assert_eq!(permute2(&a, &hi, &b), direct(&a, &b, true));
        }

        // Four-column form: the SIMD tr4 network produces four independent
        // (z0,z1) table pairs consumed by the same selectors. Model all four
        // and prove that each 128-byte result stays inside its output slab.
        const OUT_STRIDE: usize = 160;
        const POISON: u8 = 0xD3;
        let a4: [[u8; 64]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.next_u64() as u8));
        let b4: [[u8; 64]; 4] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.next_u64() as u8));
        let mut got = [POISON; 4 * OUT_STRIDE];
        let mut want = [POISON; 4 * OUT_STRIDE];
        for column in 0..4 {
            let base = column * OUT_STRIDE;
            got[base..base + 64].copy_from_slice(&permute2(&a4[column], &lo, &b4[column]));
            got[base + 64..base + 128].copy_from_slice(&permute2(&a4[column], &hi, &b4[column]));
            want[base..base + 64].copy_from_slice(&direct(&a4[column], &b4[column], false));
            want[base + 64..base + 128].copy_from_slice(&direct(&a4[column], &b4[column], true));
        }
        assert_eq!(got, want);
        for column in 0..4 {
            let gap = &got[column * OUT_STRIDE + 128..(column + 1) * OUT_STRIDE];
            assert!(gap.iter().all(|&byte| byte == POISON));
        }
    }

    #[test]
    fn lincheck_gt_fuse_kill_switch_parser() {
        use std::ffi::OsStr;

        assert!(lincheck_gt_fuse_disabled_value(Some(OsStr::new("1"))));
        for value in [
            None,
            Some(OsStr::new("")),
            Some(OsStr::new("0")),
            Some(OsStr::new("01")),
            Some(OsStr::new("true")),
        ] {
            assert!(!lincheck_gt_fuse_disabled_value(value));
        }
    }

    /// The fused register gather+transpose must equal the scalar
    /// gather + `transpose_8_f128s_to_128_bytes` byte-for-byte, at several
    /// strides.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    #[test]
    fn gather_transpose_stripe_matches_scalar() {
        let mut rng = Rng::new(0x6A77_37B1);
        for stride in [1usize, 2, 17, 128] {
            let z: Vec<F128> = (0..8 * stride).map(|_| rng.f128()).collect();
            let mut want = [0u8; 128];
            let lanes: [F128; 8] = std::array::from_fn(|r| z[r * stride]);
            transpose_8_f128s_to_128_bytes(&lanes, &mut want);
            for fuse in [false, true] {
                let mut got = [0xA5u8; 128];
                // SAFETY: 8 lanes at the given stride are in bounds; out is 128 B.
                unsafe {
                    if fuse {
                        kernels::gather_transpose_stripe_x86::<true>(
                            z.as_ptr(),
                            stride,
                            got.as_mut_ptr(),
                        );
                    } else {
                        kernels::gather_transpose_stripe_x86::<false>(
                            z.as_ptr(),
                            stride,
                            got.as_mut_ptr(),
                        );
                    }
                }
                assert_eq!(want, got, "stride {stride} fuse={fuse}");
            }
        }
        // Four-column twin: each arm must equal four independent scalar
        // transposes at q..q+4, with every output-stride gap untouched.
        for stride in [4usize, 17, 128] {
            let z: Vec<F128> = (0..8 * stride + 4).map(|_| rng.f128()).collect();
            for q in [0usize, stride.saturating_sub(4).min(3)] {
                let mut want = [0u8; 4 * 1024];
                for c in 0..4 {
                    let lanes: [F128; 8] = std::array::from_fn(|r| z[r * stride + q + c]);
                    transpose_8_f128s_to_128_bytes(&lanes, &mut want[c * 1024..c * 1024 + 128]);
                }
                for fuse in [false, true] {
                    let mut got = [0x5Au8; 4 * 1024];
                    // SAFETY: rows r*stride + q + c are in bounds; each
                    // output-stride slab covers 128 writable bytes.
                    unsafe {
                        if fuse {
                            kernels::gather_transpose_stripe4_x86::<true>(
                                z.as_ptr().add(q),
                                stride,
                                got.as_mut_ptr(),
                                1024,
                            );
                        } else {
                            kernels::gather_transpose_stripe4_x86::<false>(
                                z.as_ptr().add(q),
                                stride,
                                got.as_mut_ptr(),
                                1024,
                            );
                        }
                    }
                    for c in 0..4 {
                        assert_eq!(
                            want[c * 1024..c * 1024 + 128],
                            got[c * 1024..c * 1024 + 128],
                            "stride {stride} q {q} col {c} fuse={fuse}"
                        );
                        assert!(
                            got[c * 1024 + 128..(c + 1) * 1024]
                                .iter()
                                .all(|&byte| byte == 0x5A)
                        );
                    }
                }
            }
        }
    }

    /// The GFNI matrix builder must agree with `build_sum_table` on every
    /// byte value: affine-applying the sixteen per-stripe matrices to `v`
    /// equals `table[v]` — isolating the `8·(7−i)` byte order and the
    /// `bit j ↔ basis j` mapping from any layout question.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "gfni"
    ))]
    #[test]
    fn fold_mats_from_basis_matches_sum_table() {
        let mut rng = Rng::new(0xFA57_3AB1);
        for _ in 0..8 {
            let eq8: [F128; 8] = std::array::from_fn(|_| rng.f128());
            let mut mats = [0u64; 16];
            kernels::fold_mats_from_basis(&eq8, &mut mats);
            let mut table = vec![F128::ZERO; 256];
            build_sum_table(&eq8, &mut table);
            for v in 0..256usize {
                let mut lo = 0u64;
                let mut hi = 0u64;
                for (byte_k, &m) in mats.iter().enumerate() {
                    let mut out_byte = 0u8;
                    for i in 0..8 {
                        let row = ((m >> (8 * (7 - i))) & 0xff) as u8;
                        let parity = ((row & v as u8).count_ones() & 1) as u8;
                        out_byte |= parity << i;
                    }
                    if byte_k < 8 {
                        lo |= (out_byte as u64) << (8 * byte_k);
                    } else {
                        hi |= (out_byte as u64) << (8 * (byte_k - 8));
                    }
                }
                assert_eq!(
                    F128 { lo, hi },
                    table[v],
                    "affine(mats, {v}) must equal table[{v}]"
                );
            }
        }
    }

    /// Factoring the outer equality tensor preserves both its LSB-first
    /// indexing and the padding-aware block-major fold. The final case uses
    /// the ranked BLAKE3 outer geometry (18 variables split 9+9) without
    /// allocating its full 512 MiB witness.
    #[test]
    fn partial_fold_block_major_factorized_matches_materialized() {
        let cases: &[(usize, usize, usize, usize)] = &[
            // m, k_log, useful_bits, low-factor variables
            (10, 7, 1, 1),
            (13, 7, 127, 2),
            (16, 8, 233, 4),
            (18, 10, 997, 4),
            (25, 7, 121, 9),
        ];

        for &(m, k_log, useful_bits, lo_log) in cases {
            let mut rng =
                Rng::new(0xFAC7_0E00 + (m * 31 + k_log * 7 + useful_bits + lo_log) as u64);
            let n_log = m - k_log;
            let n_outer = 1usize << n_log;
            let chunks_per_block = (1usize << k_log) / 128;
            let z_block_major = rng.f128_vec(n_outer * chunks_per_block);
            let point = rng.f128_vec(n_log);

            let eq_outer = build_eq_table(&point);
            let (point_lo, point_hi) = point.split_at(lo_log);
            let eq_lo = build_eq_table(point_lo);
            let eq_hi = build_eq_table(point_hi);
            let lo_mask = eq_lo.len() - 1;

            // Direct tensor oracle: catches a swapped factor order even if a
            // sparse random witness happens not to observe a particular row.
            for (outer, &dense) in eq_outer.iter().enumerate() {
                assert_eq!(
                    dense,
                    eq_lo[outer & lo_mask] * eq_hi[outer >> lo_log],
                    "factor mismatch at m={m}, outer={outer}",
                );
            }

            let want = partial_fold_packed_z_block_major_padded(
                &z_block_major,
                m,
                k_log,
                useful_bits,
                &eq_outer,
            );
            let got = partial_fold_packed_z_block_major_factorized_padded(
                &z_block_major,
                m,
                k_log,
                useful_bits,
                &eq_lo,
                &eq_hi,
            );
            assert_eq!(
                want, got,
                "m={m} k_log={k_log} useful={useful_bits} lo_log={lo_log}",
            );
        }
    }

    /// Last-ρ kick then wait produces the same packed fold as today's
    /// sequential one-shot (no Fiat–Shamir). Covers the materialized eq
    /// path and the ranked `n_log=18` factorized path without a LOG2=18 prove.
    #[test]
    fn last_rho_kick_then_wait_matches_oneshot_fold() {
        let cases: &[(usize, usize, usize)] = &[
            (16, 8, 241),
            (18, 10, 997),
            (25, 7, 121), // n_log=18 factorized dispatch
        ];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0x1A57_0D00 + (m * 31 + k_log * 7 + useful_bits) as u64);
            let n_log = m - k_log;
            let n_outer = 1usize << n_log;
            let chunks_per_block = (1usize << k_log) / 128;
            let z_block_major = rng.f128_vec(n_outer * chunks_per_block);
            let z_before = z_block_major.clone();
            let inner_rest_len = k_log - 6; // ranked k_skip
            let x_inner_rest = rng.f128_vec(inner_rest_len);
            let x_outer = rng.f128_vec(n_log);
            let mut mlv = x_inner_rest;
            mlv.extend_from_slice(&x_outer);

            let want = if n_log == BLOCK_MAJOR_FACTORED_EQ_N_LOG {
                let (outer_lo, outer_hi) = x_outer.split_at(BLOCK_MAJOR_FACTORED_EQ_LO_LOG);
                partial_fold_packed_z_block_major_factorized_padded(
                    &z_block_major,
                    m,
                    k_log,
                    useful_bits,
                    &build_eq_table(outer_lo),
                    &build_eq_table(outer_hi),
                )
            } else {
                partial_fold_packed_z_block_major_padded(
                    &z_block_major,
                    m,
                    k_log,
                    useful_bits,
                    &build_eq_table(&x_outer),
                )
            };

            let _guard =
                prepare_last_rho_z_fold(&z_block_major, m, k_log, useful_bits, inner_rest_len);
            kick_last_rho_z_fold(&mlv);
            let got = wait_last_rho_z_fold().expect("kick at complete mlv must run");
            assert_eq!(
                want, got,
                "m={m} k_log={k_log} useful={useful_bits} last-ρ ≠ one-shot"
            );
            assert_eq!(
                z_block_major, z_before,
                "m={m} last-ρ kick must not mutate z"
            );
            // Kick after the fold has been waited (stand-in for "after zc
            // returns") is a no-op.
            kick_last_rho_z_fold(&mlv);
            assert!(
                wait_last_rho_z_fold().is_none(),
                "kick after wait must be a no-op"
            );
        }
    }

    /// Kicking with a short `mlv` (zerocheck rounds 2–27, `x_outer`
    /// incomplete) must not start a fold. Sequential one-shot after zc
    /// remains the path.
    #[test]
    fn last_rho_kick_incomplete_mlv_is_noop() {
        let (m, k_log, useful_bits) = (16usize, 8usize, 241usize);
        let mut rng = Rng::new(0xBAD2_2700);
        let n_log = m - k_log;
        let n_outer = 1usize << n_log;
        let chunks_per_block = (1usize << k_log) / 128;
        let z_block_major = rng.f128_vec(n_outer * chunks_per_block);
        let z_before = z_block_major.clone();
        let inner_rest_len = k_log - 6;
        let short_mlv = rng.f128_vec(inner_rest_len + n_log - 1);

        let _guard = prepare_last_rho_z_fold(&z_block_major, m, k_log, useful_bits, inner_rest_len);
        kick_last_rho_z_fold(&short_mlv);
        assert!(
            wait_last_rho_z_fold().is_none(),
            "incomplete mlv must not start the leftover fold"
        );
        assert_eq!(z_block_major, z_before, "no-op kick must not mutate z");
    }

    #[test]
    fn partial_fold_dispatch_handles_small_k() {
        let (m, k_log) = (8usize, 2usize);
        let mut rng = Rng::new(1234);
        let z = rng.bits(1 << m);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let eq = build_eq_table(&rng.f128_vec(m - k_log));

        let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
        let best = partial_fold_packed_z_best(&z_packed, m, k_log, 1 << k_log, &eq);
        assert_eq!(serial, best);
    }

    /// NEON single-matrix kernel matches the scalar reference.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn partial_fold_neon_single_matches_serial() {
        for &(m, k_log) in &[(14usize, 4), (14, 5), (16, 5), (16, 8), (18, 10)] {
            if !n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
                continue;
            }
            let mut rng = Rng::new(7000 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let p = rng.f128_vec(n_log);
            let eq = build_eq_table(&p);

            let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
            let neon = partial_fold_packed_z_neon_single(&z_packed, m, k_log, &eq);
            assert_eq!(serial, neon, "at m={m}, k_log={k_log}");
            let iblock =
                partial_fold_packed_z_neon_iblock_padded(&z_packed, m, k_log, 1usize << k_log, &eq);
            assert_eq!(serial, iblock, "iblock at m={m}, k_log={k_log}");
        }
    }

    /// The default outer(tile)-partitioned fold is **bit-identical** to the legacy
    /// i_inner-partitioned iblock kernel — dense (useful=k) and padded (useful<k,
    /// including a non-byte-aligned shape) across tile-eligible sizes. GF(2¹²⁸) add
    /// is XOR (associative + commutative), so the two partition strategies must
    /// produce the exact same length-k vector.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn partial_fold_oblock_matches_iblock() {
        // (m, k_log, useful_bits); mix of dense and padded, all tile-eligible.
        let cases: &[(usize, usize, usize)] = &[
            (14, 4, 1 << 4),   // dense, small k
            (16, 8, 1 << 8),   // dense
            (18, 10, 1 << 10), // dense
            (20, 10, 597),     // padded, non-byte-aligned
            (22, 14, 15_409),  // padded, non-byte-aligned (k=16384)
        ];
        for &(m, k_log, useful_bits) in cases {
            assert!(
                n_log_ok_for_tile(m, k_log, NEON_TILE_T),
                "case must be tile-eligible"
            );
            let k = 1usize << k_log;
            let n_log = m - k_log;
            let n_blocks = 1usize << n_log;
            let mut rng = Rng::new(7200 + (m * 31 + k_log) as u64);
            let mut z = rng.bits(1 << m);
            // Honest padding: zero rows [useful, k) of every block.
            for blk in 0..n_blocks {
                for j in useful_bits..k {
                    z[blk * k + j] = false;
                }
            }
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let eq = build_eq_table(&rng.f128_vec(n_log));
            let want =
                partial_fold_packed_z_neon_iblock_padded(&z_packed, m, k_log, useful_bits, &eq);
            let got =
                partial_fold_packed_z_neon_oblock_padded(&z_packed, m, k_log, useful_bits, &eq);
            assert_eq!(want, got, "m={m} k_log={k_log} useful={useful_bits}");
        }
    }

    /// The GFNI plane fold must match the tiled gather kernel byte-for-byte,
    /// including a non-64-aligned `useful_bits` boundary over honest zero
    /// padding (the ranked BLAKE3 shape).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "gfni"
    ))]
    #[test]
    fn partial_fold_x86_gfni_matches_tiled() {
        for &(m, k_log, useful_bits) in &[
            (16usize, 8usize, 256usize),
            (18, 10, 1000),
            (20, 14, 15_409),
        ] {
            if !n_log_ok_for_tile(m, k_log, 8) {
                continue;
            }
            let mut rng = Rng::new(9200 + m as u64);
            let block_size = 1usize << k_log;
            let mut z = rng.bits(1 << m);
            for blk in 0..(1usize << (m - k_log)) {
                for j in useful_bits..block_size {
                    z[blk * block_size + j] = false;
                }
            }
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let p = rng.f128_vec(m - k_log);
            let eq = build_eq_table(&p);
            let tiled =
                partial_fold_packed_z_x86_tiled_padded(&z_packed, m, k_log, useful_bits, &eq);
            let gfni = kernels::partial_fold_packed_z_x86_gfni_padded(
                &z_packed,
                m,
                k_log,
                useful_bits,
                &eq,
            );
            assert_eq!(tiled, gfni, "at m={m}, k_log={k_log}, useful={useful_bits}");
        }
    }

    /// `useful_bits = k`, several tile-eligible sizes).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn partial_fold_x86_tiled_matches_serial() {
        for &(m, k_log) in &[(14usize, 4), (14, 5), (16, 5), (16, 8), (18, 10)] {
            if !n_log_ok_for_tile(m, k_log, 8) {
                continue;
            }
            let mut rng = Rng::new(7100 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let p = rng.f128_vec(n_log);
            let eq = build_eq_table(&p);
            let k = 1usize << k_log;
            let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
            let tiled = partial_fold_packed_z_x86_tiled_padded(&z_packed, m, k_log, k, &eq);
            assert_eq!(serial, tiled, "at m={m}, k_log={k_log}");
        }
    }

    /// **Padding skip is byte-identical to the dense partial fold.** On a
    /// witness with honest zeros at rows `[useful_bits, 2^k_log)` of every
    /// block, the padded kernels (fast + NEON single) must produce the
    /// exact same `z_vec` as the dense kernels — and the dense scalar
    /// reference is the ground truth.
    ///
    /// Covers the three hash padding shapes plus a non-byte-aligned
    /// `useful_bits` to exercise the NEON's boundary block (rounded up to
    /// `BLOCK_K = 8`).
    #[test]
    fn partial_fold_padded_matches_dense() {
        // (m, k_log, useful_bits)
        let cases: &[(usize, usize, usize)] = &[
            // BLAKE3 (k_log=14, useful=15409 — boundary not byte-aligned).
            (17, 14, 15_409),
            // SHA-2  (k_log=15, useful=31401 — boundary not byte-aligned).
            (18, 15, 31_401),
            // Keccak (k_log=16, useful=42560 — exact byte boundary).
            (19, 16, 42_560),
        ];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xBADD_BEEF_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let n_log = m - k_log;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << n_log;

            // Random witness with bits [useful_bits, block_size) of every block
            // zeroed — mirrors the hash-module layout.
            let mut z = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    z[blk * block_size + j] = false;
                }
            }
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let outer_point = rng.f128_vec(n_log);
            let eq_outer = build_eq_table(&outer_point);

            let dense_fast = partial_fold_packed_z_fast(&z_packed, m, k_log, &eq_outer);
            let padded_fast =
                partial_fold_packed_z_fast_padded(&z_packed, m, k_log, useful_bits, &eq_outer);
            assert_eq!(
                dense_fast, padded_fast,
                "fast: m={m}, k_log={k_log}, useful={useful_bits}"
            );

            #[cfg(target_arch = "aarch64")]
            if n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
                let dense_neon = partial_fold_packed_z_neon_single(&z_packed, m, k_log, &eq_outer);
                let padded_neon = partial_fold_packed_z_neon_single_padded(
                    &z_packed,
                    m,
                    k_log,
                    useful_bits,
                    &eq_outer,
                );
                assert_eq!(
                    dense_neon, padded_neon,
                    "neon: m={m}, k_log={k_log}, useful={useful_bits}"
                );
                // i_inner-partitioned kernel: dense and padded must both match.
                let dense_iblock = partial_fold_packed_z_neon_iblock_padded(
                    &z_packed,
                    m,
                    k_log,
                    1usize << k_log,
                    &eq_outer,
                );
                let padded_iblock = partial_fold_packed_z_neon_iblock_padded(
                    &z_packed,
                    m,
                    k_log,
                    useful_bits,
                    &eq_outer,
                );
                assert_eq!(
                    dense_neon, dense_iblock,
                    "iblock dense: m={m}, k_log={k_log}, useful={useful_bits}"
                );
                assert_eq!(
                    dense_neon, padded_iblock,
                    "iblock padded: m={m}, k_log={k_log}, useful={useful_bits}"
                );
            }
        }
    }

    /// `partial_fold_packed_z(eq_outer) ↦ ẑ(·, x_outer)` matches direct MLE
    /// evaluation of z at `(i_inner, x_outer)` for boolean i_inner.
    #[test]
    fn partial_fold_is_mle_at_outer_point() {
        let m = 14;
        let k_log = 5;
        let k = 1 << k_log;
        let mut rng = Rng::new(44);
        let z = rng.bits(1 << m);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_outer = rng.f128_vec(m - k_log);
        let eq_outer = build_eq_table(&x_outer);

        let z_partial = partial_fold_packed_z(&z_packed, m, k_log, &eq_outer);

        // For each boolean i_inner ∈ {0,1}^k_log, the partial fold should
        // equal ẑ(i_inner, x_outer).
        for i_inner in 0..k {
            // Construct the m-dim point: first k_log coords from i_inner (boolean lifted),
            // then m-k_log coords from x_outer.
            let mut point = Vec::with_capacity(m);
            for j in 0..k_log {
                point.push(if (i_inner >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                });
            }
            point.extend_from_slice(&x_outer);
            let z_eval = mle_eval_bool(&z, &point);
            assert_eq!(z_partial[i_inner], z_eval, "i_inner={i_inner}");
        }
    }

    // ---- End-to-end prove/verify roundtrip on honest data ----

    /// Build a small honest instance: random sparse A_0/B_0/C_0, random z;
    /// compute a, b, c via apply_block_diag; pick three points; compute true
    /// MLE evals as v, v', v''. Roundtrip prove/verify, check claim matches
    /// what the verifier would re-derive from the (now-known-honest) z.
    #[test]
    fn prove_verify_roundtrip_honest() {
        // Exercise a range of k_skip values:
        //   k_skip = 0 (no skip)     — reduces to multilinear lincheck
        //   k_skip = k_log (max)     — only univariate inner
        //   k_skip < k_log (typical) — protocol-realistic case
        for &(m, k_log, k_skip) in &[
            (10usize, 4, 0),
            (10, 4, 2),
            (10, 4, 4),
            (12, 5, 3),
            (14, 7, 6),
            (14, 7, 0),
        ] {
            let k = 1usize << k_log;
            let mut rng = Rng::new(55 + (m * 100 + k_log * 10 + k_skip) as u64);

            // Random sparse base matrices A_0, B_0 (no C since C = I in our use case).
            let nnz_per_mat = k * 2;
            let a_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);
            let b_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);

            // Random witness z, then a = A·z, b = B·z.
            let z = rng.bits(1 << m);
            let a = apply_block_diag(&a_0, &z, k_log);
            let b = apply_block_diag(&b_0, &z, k_log);
            let z_packed = pack_z_lincheck(&z, m, k_log);

            // **One shared quirky point** (since zerocheck gives a, b claims at
            // the same point).
            let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);

            // True quirky-MLE eval claims at the shared point.
            let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
            let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

            // Prove and verify with matched challengers.
            let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
            let mut ch_p = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"flock-test-v0");
            let claim_v = verify(
                m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &proof, &mut ch_v,
            )
            .unwrap_or_else(|e| {
                panic!("verify rejected honest proof at m={m},k_log={k_log},k_skip={k_skip}: {e:?}")
            });

            assert_eq!(
                claim_p, claim_v,
                "claim mismatch at m={m}, k_log={k_log}, k_skip={k_skip}"
            );

            // The single `w` value must match the true z quirky evaluation
            // at ((r_inner_skip, r_inner_rest), x_ab.x_outer).
            let pt = QuirkyPoint {
                z_skip: claim_v.r_inner_skip,
                x_inner_rest: claim_v.r_inner_rest.clone(),
                x_outer: x_ab.x_outer.clone(),
            };
            assert_eq!(
                claim_v.w,
                mle_eval_bool_quirky(&z, m, k_log, k_skip, &pt),
                "w wrong at m={m}, k_log={k_log}, k_skip={k_skip}"
            );
        }
    }

    /// Verify must reject byte-mutated proofs. Mutation positions are picked
    /// where the corresponding matrix row-vector entry is **nonzero** —
    /// otherwise the inner-product delta vanishes and the mutation is
    /// undetectable (a property of the random sparse matrix, not a verifier
    /// bug). The verifier's consistency check is sound for *any* mutation in
    /// a nonzero-weighted slot.
    #[test]
    fn verify_rejects_mutations() {
        let m = 12;
        let k_log = 4;
        let k_skip = 2;
        let k = 1 << k_log;
        let mut rng = Rng::new(66);
        let a_0 = random_sparse_matrix(k, k * 5, &mut rng);
        let b_0 = random_sparse_matrix(k, k * 5, &mut rng);
        let z = rng.bits(1 << m);
        let a = apply_block_diag(&a_0, &z, k_log);
        let b = apply_block_diag(&b_0, &z, k_log);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);
        let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
        let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

        let _seed: u64 = 0xFEEDFACE;
        let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

        // Pick a mutation position where BOTH row vectors are nonzero so the
        // mutation guarantees both checks would diverge.
        let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
        let row_a = sparse_row_fold(&a_0, &eq_inner);
        let row_b = sparse_row_fold(&b_0, &eq_inner);
        let idx = (0..k)
            .find(|&i| row_a[i] != F128::ZERO || row_b[i] != F128::ZERO)
            .expect("no row-vector slot is nonzero in either A or B — test degenerate");

        // Mutations now target `z_partial` (the post-sumcheck length-2^k_skip
        // vector). Bit-flipping any entry must cause the sumcheck-final check
        // to fail (running_claim ≠ Σ comb_partial · z_partial).
        let n_skip = 1usize << k_skip;
        let skip_idx = idx % n_skip;
        let mutations: Vec<(String, Box<dyn Fn(&LincheckProof) -> LincheckProof>)> = vec![
            (
                format!("z_partial[{skip_idx}].lo bit-flip"),
                Box::new(move |p| {
                    let mut q = p.clone();
                    q.z_partial[skip_idx].lo ^= 1;
                    q
                }),
            ),
            (
                format!("z_partial[{skip_idx}].hi bit-flip"),
                Box::new(move |p| {
                    let mut q = p.clone();
                    q.z_partial[skip_idx].hi ^= 1;
                    q
                }),
            ),
        ];
        for (label, mutate) in mutations {
            let bad = mutate(&proof);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &bad, &mut ch);
            assert!(
                matches!(res, Err(VerifyError::ConsistencyFailed { .. })),
                "verify did not reject {label}: got {res:?}"
            );
        }
    }

    /// Verify must reject shape errors.
    #[test]
    fn verify_rejects_shape_errors() {
        let m = 10;
        let k_log = 3;
        let k_skip = 1;
        let k = 1 << k_log;
        let mut rng = Rng::new(77);
        let a_0 = random_sparse_matrix(k, k, &mut rng);
        let b_0 = random_sparse_matrix(k, k, &mut rng);
        let z = rng.bits(1 << m);
        let a = apply_block_diag(&a_0, &z, k_log);
        let b = apply_block_diag(&b_0, &z, k_log);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);
        let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
        let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

        let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

        // Truncate z_partial.
        let mut bad = proof.clone();
        bad.z_partial.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &bad, &mut ch),
            Err(VerifyError::BadVectorLength { .. })
        ));

        // Wrong x_inner_rest length.
        let mut ch = FsChallenger::new(b"flock-test-v0");
        let bad_x_ab = QuirkyPoint {
            z_skip: x_ab.z_skip,
            x_inner_rest: x_ab.x_inner_rest[..x_ab.x_inner_rest.len() - 1].to_vec(),
            x_outer: x_ab.x_outer.clone(),
        };
        assert!(matches!(
            verify(
                m, k_log, k_skip, &circuit, &bad_x_ab, v_a, v_b, &proof, &mut ch
            ),
            Err(VerifyError::BadInnerRestLength { .. })
        ));

        // k_skip > k_log.
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(
                m,
                k_log,
                k_log + 1,
                &circuit,
                &x_ab,
                v_a,
                v_b,
                &proof,
                &mut ch,
            ),
            Err(VerifyError::KSkipExceedsKLog { .. })
        ));
    }
}
