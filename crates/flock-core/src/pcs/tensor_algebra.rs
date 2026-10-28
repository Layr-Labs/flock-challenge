use crate::field::F128;
use core::ops::{Add, AddAssign};

pub const DEGREE: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorAlgebra {
    pub elems: Vec<F128>,
}

impl TensorAlgebra {
    pub fn zero() -> Self {
        Self {
            elems: vec![F128::ZERO; DEGREE],
        }
    }

    pub fn one() -> Self {
        let mut elems = vec![F128::ZERO; DEGREE];
        elems[0] = F128::ONE;
        Self { elems }
    }

    pub fn from_vertical(x: F128) -> Self {
        let mut elems = vec![F128::ZERO; DEGREE];
        elems[0] = x;
        Self { elems }
    }

    pub fn scale_vertical(mut self, scalar: F128) -> Self {
        for e in self.elems.iter_mut() {
            *e *= scalar;
        }
        self
    }

    pub fn scale_horizontal(self, scalar: F128) -> Self {
        self.transpose().scale_vertical(scalar).transpose()
    }

    pub fn transpose(mut self) -> Self {
        square_transpose(&mut self.elems);
        self
    }

    pub fn fold_vertical(self, coeffs: &[F128]) -> F128 {
        assert_eq!(
            coeffs.len(),
            DEGREE,
            "fold_vertical: coeffs.len() must be 128"
        );
        let transposed = self.transpose();
        crate::pcs::ring_switch::inner_product(&transposed.elems, coeffs)
    }
}

impl Add<&TensorAlgebra> for TensorAlgebra {
    type Output = TensorAlgebra;
    fn add(mut self, rhs: &TensorAlgebra) -> TensorAlgebra {
        self += rhs;
        self
    }
}

impl AddAssign<&TensorAlgebra> for TensorAlgebra {
    fn add_assign(&mut self, rhs: &TensorAlgebra) {
        crate::field::f128_slice::xor_assign(&mut self.elems, &rhs.elems);
    }
}

fn square_transpose(elems: &mut [F128]) {
    assert_eq!(
        elems.len(),
        DEGREE,
        "square_transpose: input must be length 128"
    );
    let transposed = crate::pcs::ring_switch::tensor_algebra_transpose(elems);
    elems.copy_from_slice(&transposed);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn nx(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.nx(),
                hi: self.nx(),
            }
        }
        fn ta(&mut self) -> TensorAlgebra {
            let elems = (0..DEGREE).map(|_| self.f128()).collect();
            TensorAlgebra { elems }
        }
    }

    #[test]
    fn transpose_involution() {
        let mut rng = Rng::new(0xC0FFEE);
        for _ in 0..10 {
            let t = rng.ta();
            assert_eq!(t.clone().transpose().transpose(), t);
        }
    }

    #[test]
    fn transpose_bit_semantics() {
        let mut rng = Rng::new(42);
        let original = rng.ta();
        let transposed = original.clone().transpose();

        fn bit(x: F128, b: usize) -> u64 {
            if b < 64 {
                (x.lo >> b) & 1
            } else {
                (x.hi >> (b - 64)) & 1
            }
        }

        for i in 0..DEGREE {
            for j in 0..DEGREE {
                let orig_ij = bit(original.elems[i], j);
                let trans_ji = bit(transposed.elems[j], i);
                assert_eq!(orig_ij, trans_ji, "transpose mismatch at (i={i}, j={j})");
            }
        }
    }

    #[test]
    fn from_vertical_scale_vertical() {
        let mut rng = Rng::new(123);
        for _ in 0..10 {
            let x = rng.f128();
            let y = rng.f128();
            let lhs = TensorAlgebra::from_vertical(x).scale_vertical(y);
            let rhs = TensorAlgebra::from_vertical(x * y);
            assert_eq!(lhs, rhs);
        }
    }

    #[test]
    fn scale_horizontal_via_transpose() {
        let mut rng = Rng::new(456);
        for _ in 0..10 {
            let t = rng.ta();
            let s = rng.f128();
            let via_api = t.clone().scale_horizontal(s);
            let manual = t.transpose().scale_vertical(s).transpose();
            assert_eq!(via_api, manual);
        }
    }

    #[test]
    fn add_is_xor_pairwise() {
        let mut rng = Rng::new(789);
        let a = rng.ta();
        let b = rng.ta();
        let sum = a.clone() + &b;
        for i in 0..DEGREE {
            let expected = F128 {
                lo: a.elems[i].lo ^ b.elems[i].lo,
                hi: a.elems[i].hi ^ b.elems[i].hi,
            };
            assert_eq!(sum.elems[i], expected);
        }
    }

    #[test]
    fn add_zero_is_identity() {
        let mut rng = Rng::new(1011);
        let t = rng.ta();
        let z = TensorAlgebra::zero();
        assert_eq!(t.clone() + &z, t);
    }

    #[test]
    fn one_from_vertical_one() {
        assert_eq!(
            TensorAlgebra::one(),
            TensorAlgebra::from_vertical(F128::ONE)
        );
    }

    #[test]
    fn scale_vertical_distributes_over_add() {
        let mut rng = Rng::new(1213);
        let a = rng.ta();
        let b = rng.ta();
        let s = rng.f128();
        let lhs = (a.clone() + &b).scale_vertical(s);
        let rhs = a.scale_vertical(s) + &b.scale_vertical(s);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn fold_vertical_with_zero_coeffs_is_zero() {
        let mut rng = Rng::new(1415);
        let t = rng.ta();
        let zeros = vec![F128::ZERO; DEGREE];
        assert_eq!(t.fold_vertical(&zeros), F128::ZERO);
    }
}
