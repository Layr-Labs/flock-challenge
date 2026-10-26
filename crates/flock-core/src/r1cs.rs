#[derive(Clone, Debug)]
pub struct SparseBinaryMatrix {
    pub num_rows: usize,
    pub num_cols: usize,
    pub rows: Vec<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WitnessLayout {
    #[default]
    RowMajor,
    BatchMajor,
}

#[derive(Debug)]
pub struct BlockR1cs {
    pub m: usize,
    pub k_log: usize,
    pub k_skip: usize,

    pub useful_bits: usize,
    pub a_0: SparseBinaryMatrix,
    pub b_0: SparseBinaryMatrix,
    pub c_0: SparseBinaryMatrix,

    pub layout: WitnessLayout,

    pub const_pin: Option<usize>,

    #[doc(hidden)]
    pub digest_cache: std::sync::OnceLock<[u8; 32]>,

    #[doc(hidden)]
    pub csc_cache: std::sync::OnceLock<crate::lincheck::CscCircuit>,
}

impl Clone for BlockR1cs {
    fn clone(&self) -> Self {
        Self {
            m: self.m,
            k_log: self.k_log,
            k_skip: self.k_skip,
            useful_bits: self.useful_bits,
            a_0: self.a_0.clone(),
            b_0: self.b_0.clone(),
            c_0: self.c_0.clone(),
            layout: self.layout,
            const_pin: self.const_pin,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        }
    }
}

impl BlockR1cs {
    pub fn n_outer(&self) -> usize {
        1usize << self.n_log()
    }

    pub fn n_log(&self) -> usize {
        self.m - self.k_log
    }

    pub fn k(&self) -> usize {
        1usize << self.k_log
    }

    pub fn n(&self) -> usize {
        1usize << self.m
    }

    pub fn c0_is_identity(&self) -> bool {
        self.c_0.num_rows == self.k()
            && self.c_0.num_cols == self.k()
            && self
                .c_0
                .rows
                .iter()
                .enumerate()
                .all(|(i, row)| row.as_slice() == [i])
    }

    pub fn sparse_lincheck_circuit(&self) -> crate::lincheck::SparseMatrixCircuit<'_> {
        crate::lincheck::SparseMatrixCircuit::new(&self.a_0, &self.b_0)
            .with_const_pin(self.const_pin)
    }

    pub fn csc_lincheck_circuit(&self) -> &crate::lincheck::CscCircuit {
        self.csc_cache.get_or_init(|| {
            crate::lincheck::CscCircuit::from_matrices(&self.a_0, &self.b_0)
                .with_const_pin(self.const_pin)
        })
    }

    pub fn apply_a(&self, z: &[bool]) -> Vec<bool> {
        apply_block_diag(&self.a_0, z, self.k_log)
    }

    pub fn apply_b(&self, z: &[bool]) -> Vec<bool> {
        apply_block_diag(&self.b_0, z, self.k_log)
    }

    pub fn apply_c(&self, z: &[bool]) -> Vec<bool> {
        apply_block_diag(&self.c_0, z, self.k_log)
    }

    pub fn satisfies(&self, z: &[bool]) -> bool {
        assert_eq!(z.len(), self.n());
        let a = self.apply_a(z);
        let b = self.apply_b(z);
        let c = self.apply_c(z);
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .all(|((ai, bi), ci)| (*ai & *bi) == *ci)
    }

    pub fn apply_a_packed(&self, z_packed: &[crate::field::F128]) -> Vec<crate::field::F128> {
        apply_block_diag_packed(&self.a_0, z_packed, self.m, self.k_log)
    }

    pub fn apply_b_packed(&self, z_packed: &[crate::field::F128]) -> Vec<crate::field::F128> {
        apply_block_diag_packed(&self.b_0, z_packed, self.m, self.k_log)
    }

    pub fn apply_c_packed(&self, z_packed: &[crate::field::F128]) -> Vec<crate::field::F128> {
        apply_block_diag_packed(&self.c_0, z_packed, self.m, self.k_log)
    }

    pub fn padding_spec(&self) -> crate::zerocheck::PaddingSpec {
        match self.layout {
            WitnessLayout::RowMajor => crate::zerocheck::PaddingSpec {
                k_log: self.k_log,
                useful_bits_per_block: self.useful_bits,
            },
            WitnessLayout::BatchMajor => crate::zerocheck::PaddingSpec {
                k_log: self.m,
                useful_bits_per_block: self.useful_bits.div_ceil(128) << (7 + self.n_log()),
            },
        }
    }

    pub fn x_ab_from_mlv(
        &self,
        z_skip: crate::field::F128,
        mlv: &[crate::field::F128],
    ) -> crate::lincheck::QuirkyPoint {
        let inner_rest_len = self.k_log - self.k_skip;
        assert_eq!(mlv.len(), self.m - self.k_skip);
        match self.layout {
            WitnessLayout::RowMajor => crate::lincheck::QuirkyPoint {
                z_skip,
                x_inner_rest: mlv[..inner_rest_len].to_vec(),
                x_outer: mlv[inner_rest_len..].to_vec(),
            },
            WitnessLayout::BatchMajor => {
                assert!(
                    self.k_log >= 7 && self.k_skip == 6,
                    "BatchMajor needs k_log >= 7, k_skip = 6"
                );
                let n_log = self.n_log();
                let mut x_inner_rest = Vec::with_capacity(inner_rest_len);
                x_inner_rest.push(mlv[0]);
                x_inner_rest.extend_from_slice(&mlv[1 + n_log..]);
                crate::lincheck::QuirkyPoint {
                    z_skip,
                    x_inner_rest,
                    x_outer: mlv[1..1 + n_log].to_vec(),
                }
            }
        }
    }

    pub fn ab_claim_point(
        &self,
        r_inner_skip: crate::field::F128,
        r_inner_rest: &[crate::field::F128],
        x_outer: &[crate::field::F128],
    ) -> crate::lincheck::QuirkyPoint {
        match self.layout {
            WitnessLayout::RowMajor => crate::lincheck::QuirkyPoint {
                z_skip: r_inner_skip,
                x_inner_rest: r_inner_rest.to_vec(),
                x_outer: x_outer.to_vec(),
            },
            WitnessLayout::BatchMajor => {
                let mut suffix = Vec::with_capacity(x_outer.len() + r_inner_rest.len() - 1);
                suffix.extend_from_slice(x_outer);
                suffix.extend_from_slice(&r_inner_rest[1..]);
                crate::lincheck::QuirkyPoint {
                    z_skip: r_inner_skip,
                    x_inner_rest: vec![r_inner_rest[0]],
                    x_outer: suffix,
                }
            }
        }
    }

    pub fn c_claim_point(
        &self,
        z_skip: crate::field::F128,
        r_rest: &[crate::field::F128],
    ) -> crate::lincheck::QuirkyPoint {
        let inner_rest_len = self.k_log - self.k_skip;
        match self.layout {
            WitnessLayout::RowMajor => crate::lincheck::QuirkyPoint {
                z_skip,
                x_inner_rest: r_rest[..inner_rest_len].to_vec(),
                x_outer: r_rest[inner_rest_len..].to_vec(),
            },
            WitnessLayout::BatchMajor => crate::lincheck::QuirkyPoint {
                z_skip,
                x_inner_rest: vec![r_rest[0]],
                x_outer: r_rest[1..].to_vec(),
            },
        }
    }

    pub fn statement_digest(&self) -> [u8; 32] {
        *self.digest_cache.get_or_init(|| {
            let mut h = blake3::Hasher::new();
            h.update(b"flock-r1cs-stmt-v1");
            h.update(&(self.m as u64).to_le_bytes());
            h.update(&(self.k_log as u64).to_le_bytes());
            h.update(&(self.k_skip as u64).to_le_bytes());

            h.update(&[match self.layout {
                WitnessLayout::RowMajor => 0u8,
                WitnessLayout::BatchMajor => 1u8,
            }]);
            absorb_matrix(&mut h, &self.a_0);
            absorb_matrix(&mut h, &self.b_0);
            absorb_matrix(&mut h, &self.c_0);
            *h.finalize().as_bytes()
        })
    }

    pub fn satisfies_packed(&self, z_packed: &[crate::field::F128]) -> bool {
        use crate::field::F128;
        assert_eq!(z_packed.len(), 1usize << (self.m - 7));
        let a = self.apply_a_packed(z_packed);
        let b = self.apply_b_packed(z_packed);
        let c = self.apply_c_packed(z_packed);
        a.iter().zip(b.iter()).zip(c.iter()).all(|((ai, bi), ci)| {
            let ab = F128 {
                lo: ai.lo & bi.lo,
                hi: ai.hi & bi.hi,
            };
            ab == *ci
        })
    }
}

fn absorb_matrix(h: &mut blake3::Hasher, m: &SparseBinaryMatrix) {
    h.update(&(m.num_rows as u64).to_le_bytes());
    h.update(&(m.num_cols as u64).to_le_bytes());
    for row in &m.rows {
        h.update(&(row.len() as u64).to_le_bytes());
        for &col in row {
            h.update(&(col as u64).to_le_bytes());
        }
    }
}

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

pub fn apply_block_diag_packed(
    m_0: &SparseBinaryMatrix,
    z_packed: &[crate::field::F128],
    m: usize,
    k_log: usize,
) -> Vec<crate::field::F128> {
    use crate::field::F128;
    use rayon::prelude::*;

    let k = 1usize << k_log;
    assert_eq!(m_0.num_rows, k);
    assert_eq!(m_0.num_cols, k);
    let n_packed = 1usize << (m - 7);
    assert_eq!(z_packed.len(), n_packed);
    let n_outer = 1usize << (m - k_log);

    let mut out = vec![F128::ZERO; n_packed];

    if k_log >= 7 {
        let (row_ptr, cols) = flatten_csr(m_0);
        let f128_per_block = k / 128;

        let strip = if n_outer / 64 >= rayon::current_num_threads().max(1) {
            64
        } else {
            APPLY_STRIP
        };
        out.par_chunks_mut(strip * f128_per_block)
            .zip(z_packed.par_chunks(strip * f128_per_block))
            .for_each(|(out_strip, z_strip)| {
                let n_blocks = z_strip.len() / f128_per_block;
                if n_blocks == 64 {
                    apply_strip64_csr(&row_ptr, &cols, z_strip, out_strip, f128_per_block);
                } else if n_blocks == APPLY_STRIP {
                    apply_strip_csr(&row_ptr, &cols, z_strip, out_strip, f128_per_block);
                } else {
                    for (ob, zb) in out_strip
                        .chunks_mut(f128_per_block)
                        .zip(z_strip.chunks(f128_per_block))
                    {
                        apply_one_block_csr(&row_ptr, &cols, zb, ob);
                    }
                }
            });
    } else {
        for i_outer in 0..n_outer {
            let block_start_bit = i_outer * k;
            for r in 0..k {
                let mut acc = false;
                for &j in m_0.rows[r].iter() {
                    let global_in_bit = block_start_bit + j;
                    if get_bit_packed(z_packed, global_in_bit) {
                        acc = !acc;
                    }
                }
                if acc {
                    set_bit_packed(&mut out, block_start_bit + r);
                }
            }
        }
    }
    out
}

const APPLY_STRIP: usize = 8;

fn flatten_csr(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    assert!(m.num_cols <= u32::MAX as usize);
    let nnz: usize = m.rows.iter().map(|r| r.len()).sum();
    let mut row_ptr = Vec::with_capacity(m.num_rows + 1);
    let mut cols = Vec::with_capacity(nnz);
    row_ptr.push(0u32);
    for row in &m.rows {
        for &c in row {
            cols.push(c as u32);
        }
        row_ptr.push(cols.len() as u32);
    }
    (row_ptr, cols)
}

#[inline]
fn as_u128s(block: &[crate::field::F128]) -> &[u128] {
    unsafe { std::slice::from_raw_parts(block.as_ptr() as *const u128, block.len()) }
}

fn apply_strip_csr(
    row_ptr: &[u32],
    cols: &[u32],
    z_strip: &[crate::field::F128],
    out_strip: &mut [crate::field::F128],
    f128_per_block: usize,
) {
    use crate::bits::transpose_8_u64s_to_64_bytes;

    debug_assert_eq!(z_strip.len(), APPLY_STRIP * f128_per_block);
    debug_assert_eq!(out_strip.len(), APPLY_STRIP * f128_per_block);
    let k = f128_per_block * 128;
    let u64_per_block = k / 64;

    let z_u64: &[u64] =
        unsafe { std::slice::from_raw_parts(z_strip.as_ptr() as *const u64, z_strip.len() * 2) };

    let mut colbits = vec![0u8; k];
    for w in 0..u64_per_block {
        let lanes: [u64; 8] = std::array::from_fn(|s| z_u64[s * u64_per_block + w]);
        transpose_8_u64s_to_64_bytes(&lanes, &mut colbits[w * 64..w * 64 + 64]);
    }

    for out_idx in 0..f128_per_block {
        let mut acc = [0u128; APPLY_STRIP];
        for offset in 0..128 {
            let r = out_idx * 128 + offset;
            let lo = row_ptr[r] as usize;
            let hi = row_ptr[r + 1] as usize;

            let mut row_bits: u8 = 0;
            for &j in &cols[lo..hi] {
                row_bits ^= colbits[j as usize];
            }
            if row_bits != 0 {
                for (s, a) in acc.iter_mut().enumerate() {
                    *a |= (((row_bits >> s) & 1) as u128) << offset;
                }
            }
        }
        for (s, a) in acc.iter().enumerate() {
            if *a != 0 {
                let slot = &mut out_strip[s * f128_per_block + out_idx];
                slot.lo |= *a as u64;
                slot.hi |= (*a >> 64) as u64;
            }
        }
    }
}

#[inline]
fn transpose_64x64(a: &mut [u64; 64]) {
    let mut j: usize = 32;
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut k: usize = 0;
        let ju = j as u64;
        while k < 64 {
            let ak = a[k];
            let akj = a[k + j];
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            unsafe {
                a[k] = crate::gf2_kernels::gf2_kern_delta_swap_lo(ak, akj, ju, m);
                a[k + j] = crate::gf2_kernels::gf2_kern_delta_swap_hi(ak, akj, ju, m);
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let t = ((ak >> j) ^ akj) & m;
                a[k] = ak ^ (t << j);
                a[k + j] = akj ^ t;
            }
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

fn apply_strip64_csr(
    row_ptr: &[u32],
    cols: &[u32],
    z_strip: &[crate::field::F128],
    out_strip: &mut [crate::field::F128],
    f128_per_block: usize,
) {
    const S: usize = 64;
    debug_assert_eq!(z_strip.len(), S * f128_per_block);
    let k = f128_per_block * 128;
    let u64_per_block = k / 64;

    let z_u64: &[u64] =
        unsafe { std::slice::from_raw_parts(z_strip.as_ptr() as *const u64, z_strip.len() * 2) };
    let out_u64: &mut [u64] = unsafe {
        std::slice::from_raw_parts_mut(out_strip.as_mut_ptr() as *mut u64, out_strip.len() * 2)
    };

    let mut colbits = vec![0u64; k];
    let mut lanes = [0u64; 64];
    for w in 0..u64_per_block {
        for (s, lane) in lanes.iter_mut().enumerate() {
            *lane = z_u64[s * u64_per_block + w];
        }
        transpose_64x64(&mut lanes);
        colbits[w * 64..w * 64 + 64].copy_from_slice(&lanes);
    }

    let mut rowbits = vec![0u64; k];
    for (r, rb) in rowbits.iter_mut().enumerate() {
        let lo = row_ptr[r] as usize;
        let hi = row_ptr[r + 1] as usize;
        let mut x = 0u64;
        for &j in &cols[lo..hi] {
            x ^= colbits[j as usize];
        }
        *rb = x;
    }

    for w in 0..u64_per_block {
        lanes.copy_from_slice(&rowbits[w * 64..w * 64 + 64]);
        transpose_64x64(&mut lanes);
        for (s, lane) in lanes.iter().enumerate() {
            out_u64[s * u64_per_block + w] = *lane;
        }
    }
}

fn apply_one_block_csr(
    row_ptr: &[u32],
    cols: &[u32],
    z_block: &[crate::field::F128],
    out_block: &mut [crate::field::F128],
) {
    let z = as_u128s(z_block);
    let f128_per_block = z_block.len();
    for out_idx in 0..f128_per_block {
        let mut acc: u128 = 0;
        for offset in 0..128 {
            let r = out_idx * 128 + offset;
            let lo = row_ptr[r] as usize;
            let hi = row_ptr[r + 1] as usize;
            let mut row_acc: u128 = 0;
            for &j in &cols[lo..hi] {
                row_acc ^= (z[(j >> 7) as usize] >> (j & 127)) & 1;
            }
            acc |= row_acc << offset;
        }
        if acc != 0 {
            out_block[out_idx].lo |= acc as u64;
            out_block[out_idx].hi |= (acc >> 64) as u64;
        }
    }
}

#[inline]
fn get_bit_packed(z_packed: &[crate::field::F128], global_bit: usize) -> bool {
    let i_packed = global_bit / 128;
    let local = global_bit % 128;
    if local < 64 {
        (z_packed[i_packed].lo >> local) & 1 == 1
    } else {
        (z_packed[i_packed].hi >> (local - 64)) & 1 == 1
    }
}

#[inline]
fn set_bit_packed(z_packed: &mut [crate::field::F128], global_bit: usize) {
    let i_packed = global_bit / 128;
    let local = global_bit % 128;
    if local < 64 {
        z_packed[i_packed].lo |= 1u64 << local;
    } else {
        z_packed[i_packed].hi |= 1u64 << (local - 64);
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(k: usize) -> SparseBinaryMatrix {
        SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: (0..k).map(|i| vec![i]).collect(),
        }
    }

    #[test]
    fn packed_matches_bool_apply() {
        use crate::pcs::pack_witness;

        for &(m, k_log) in &[(10usize, 3), (10, 6), (10, 7), (12, 8), (13, 7), (15, 8)] {
            let k = 1usize << k_log;
            let mut rng = TestRng::new(0xABCD + m as u64 + k_log as u64 * 37);
            let mat = SparseBinaryMatrix {
                num_rows: k,
                num_cols: k,

                rows: (0..k)
                    .map(|_| {
                        let n_nonzero = 2 + (rng.next_u64() % 4) as usize;
                        (0..n_nonzero)
                            .map(|_| (rng.next_u64() as usize) % k)
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            };

            let z: Vec<bool> = (0..(1 << m)).map(|_| rng.next_u64() & 1 == 1).collect();
            let z_packed = pack_witness(&z, m);

            let a_bool = apply_block_diag(&mat, &z, k_log);
            let a_packed = apply_block_diag_packed(&mat, &z_packed, m, k_log);

            for i in 0..(1 << m) {
                let expected = a_bool[i];
                let got = get_bit_packed(&a_packed, i);
                assert_eq!(got, expected, "mismatch at i={i} (m={m}, k_log={k_log})");
            }
        }
    }

    struct TestRng(u64);
    impl TestRng {
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
    }

    #[test]
    fn identity_matrices_accept_any_witness() {
        let k_log = 3;
        let m = 6;
        let r1cs = BlockR1cs {
            m,
            k_log,
            k_skip: 2,
            useful_bits: 1 << k_log,
            a_0: identity(1 << k_log),
            b_0: identity(1 << k_log),
            c_0: identity(1 << k_log),
            layout: WitnessLayout::RowMajor,
            const_pin: None,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        for seed in 0..4 {
            let z: Vec<bool> = (0..(1 << m)).map(|i| ((i ^ seed) & 1) == 1).collect();
            assert!(r1cs.satisfies(&z), "seed={seed}");
        }
    }

    #[test]
    fn zero_matrices_require_zero_witness() {
        let k_log = 3;
        let m = 6;
        let zero = SparseBinaryMatrix {
            num_rows: 1 << k_log,
            num_cols: 1 << k_log,
            rows: vec![Vec::new(); 1 << k_log],
        };
        let r1cs = BlockR1cs {
            m,
            k_log,
            k_skip: 2,
            useful_bits: 1 << k_log,
            a_0: zero.clone(),
            b_0: zero,
            c_0: identity(1 << k_log),
            layout: WitnessLayout::RowMajor,
            const_pin: None,
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        let z_zero = vec![false; 1 << m];
        assert!(r1cs.satisfies(&z_zero));
        let mut z_nonzero = vec![false; 1 << m];
        z_nonzero[5] = true;
        assert!(!r1cs.satisfies(&z_nonzero));
    }
}
