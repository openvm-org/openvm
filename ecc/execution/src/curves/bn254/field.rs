use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, Fq6},
    ff::Field,
};

use crate::common::{EvaluatedLine, FieldExtension, Fp12Constructor, Fp2Constructor, LineDType};

pub const BN254_XI: Fq2 = Fq2 {
    c0: Fq::from_raw([9, 0, 0, 0]),
    c1: Fq::ONE,
};

// from gnark implementation: https://github.com/Consensys/gnark/blob/42dcb0c3673b2394bf1fd82f5128f7a121d7d48e/std/algebra/emulated/sw_bn254/pairing.go#L356
// loopCounter = 6x₀+2 = 29793968203157093288
// in 2-NAF (nonadjacent form)
pub const GNARK_BN254_PBE_NAF: [i32; 66] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

impl Fp2Constructor<Fq> for Fq2 {
    fn new(c0: Fq, c1: Fq) -> Self {
        Fq2 { c0, c1 }
    }
}

impl Fp12Constructor<Fq2> for Fq12 {
    fn new(c00: Fq2, c01: Fq2, c02: Fq2, c10: Fq2, c11: Fq2, c12: Fq2) -> Self {
        Fq12 {
            c0: Fq6 {
                c0: c00,
                c1: c01,
                c2: c02,
            },
            c1: Fq6 {
                c0: c10,
                c1: c11,
                c2: c12,
            },
        }
    }
}

/// FieldExtension for Fq2 with Fq as base field
impl FieldExtension for Fq2 {
    type BaseField = Fq;

    fn from_coeffs(coeffs: &[Self::BaseField]) -> Self {
        assert!(coeffs.len() <= 2, "coeffs must have at most 2 elements");
        let mut coeffs = coeffs.to_vec();
        coeffs.resize(2, Self::BaseField::ZERO);

        Fq2 {
            c0: coeffs[0],
            c1: coeffs[1],
        }
    }

    fn embed(base_elem: &Self::BaseField) -> Self {
        Fq2 {
            c0: *base_elem,
            c1: Fq::ZERO,
        }
    }

    fn conjugate(&self) -> Self {
        let mut s = *self;
        Fq2::conjugate(&mut s);
        s
    }

    fn frobenius_map(&self, power: Option<usize>) -> Self {
        let mut s = *self;
        Fq2::frobenius_map(&mut s, power.unwrap());
        s
    }

    fn mul_base(&self, rhs: &Self::BaseField) -> Self {
        Fq2 {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}

/// FieldExtension for Fq12 with Fq6 as base field since halo2curves does not implement `Field` for Fq6.
impl FieldExtension for Fq12 {
    type BaseField = Fq2;

    fn from_coeffs(coeffs: &[Self::BaseField]) -> Self {
        assert!(coeffs.len() <= 6, "coeffs must have at most 6 elements");
        let mut coeffs = coeffs.to_vec();
        coeffs.resize(6, Self::BaseField::ZERO);

        Fq12 {
            c0: Fq6 {
                c0: coeffs[0],
                c1: coeffs[2],
                c2: coeffs[4],
            },
            c1: Fq6 {
                c0: coeffs[1],
                c1: coeffs[3],
                c2: coeffs[5],
            },
        }
    }

    fn embed(base_elem: &Self::BaseField) -> Self {
        let fq6_pt = Fq6 {
            c0: *base_elem,
            c1: Fq2::zero(),
            c2: Fq2::zero(),
        };
        Fq12 {
            c0: fq6_pt,
            c1: Fq6::zero(),
        }
    }

    fn conjugate(&self) -> Self {
        let mut s = *self;
        Fq12::conjugate(&mut s);
        s
    }

    fn frobenius_map(&self, power: Option<usize>) -> Self {
        let mut s = *self;
        Fq12::frobenius_map(&mut s, power.unwrap());
        s
    }

    fn mul_base(&self, rhs: &Self::BaseField) -> Self {
        let fq6_pt = Fq6 {
            c0: *rhs,
            c1: Fq2::zero(),
            c2: Fq2::zero(),
        };
        Fq12 {
            c0: self.c0 * fq6_pt,
            c1: self.c1 * fq6_pt,
        }
    }
}

impl LineDType<Fq, Fq2, Fq12> for Fq12 {
    fn from_evaluated_line_d_type(line: EvaluatedLine<Fq, Fq2>) -> Fq12 {
        Fq12::from_coeffs(&[Fq2::ONE, line.b, Fq2::ZERO, line.c, Fq2::ZERO, Fq2::ZERO])
    }
}
