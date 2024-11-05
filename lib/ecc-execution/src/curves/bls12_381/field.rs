use axvm_ecc::{
    field::{ExpBigInt, FieldExtension, Fp12Constructor, Fp2Constructor},
    pairing::{EvaluatedLine, LineMType},
};
use halo2curves_axiom::{
    bls12_381::{Fq, Fq12, Fq2, Fq6},
    ff::Field,
};

pub struct FieldExtFq2(pub(crate) Fq2);

impl Fp2Constructor<Fq> for FieldExtFq2 {
    fn new(c0: Fq, c1: Fq) -> Self {
        FieldExtFq2(Fq2 { c0, c1 })
    }
}

pub struct FieldExtFq12(pub(crate) Fq12);

impl Fp12Constructor<FieldExtFq2> for FieldExtFq12 {
    fn new(
        c00: FieldExtFq2,
        c01: FieldExtFq2,
        c02: FieldExtFq2,
        c10: FieldExtFq2,
        c11: FieldExtFq2,
        c12: FieldExtFq2,
    ) -> Self {
        FieldExtFq12(Fq12 {
            c0: Fq6 {
                c0: c00.0,
                c1: c01.0,
                c2: c02.0,
            },
            c1: Fq6 {
                c0: c10.0,
                c1: c11.0,
                c2: c12.0,
            },
        })
    }
}

/// FieldExtension for Fq2 with Fq as base field
impl FieldExtension for FieldExtFq2 {
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
        FieldExtFq2(Fq2 {
            c0: *base_elem,
            c1: Fq::ZERO,
        })
    }

    fn conjugate(&self) -> Self {
        FieldExtFq2(Fq2::conjugate(&self.0))
    }

    fn frobenius_map(&self, _power: Option<usize>) -> Self {
        FieldExtFq2(Fq2::frobenius_map(&self.0))
    }

    fn mul_base(&self, rhs: &Self::BaseField) -> Self {
        FieldExtFq2(Fq2 {
            c0: self.0.c0 * rhs,
            c1: self.0.c1 * rhs,
        })
    }
}

///
/// Note that halo2curves does not implement `Field` for Fq6, so we need to implement the intermediate points manually.
///
/// FieldExtension for Fq12 with Fq2 as base field since halo2curves does not implement `Field` for Fq6.
impl FieldExtension for FieldExtFq12 {
    type BaseField = Fq2;

    fn from_coeffs(coeffs: &[Self::BaseField]) -> Self {
        assert!(coeffs.len() <= 6, "coeffs must have at most 6 elements");
        let mut coeffs = coeffs.to_vec();
        coeffs.resize(6, Self::BaseField::ZERO);

        FieldExtFq12(Fq12 {
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
        })
    }

    fn embed(base_elem: &Self::BaseField) -> Self {
        let fq6_pt = Fq6 {
            c0: *base_elem,
            c1: Fq2::zero(),
            c2: Fq2::zero(),
        };
        FieldExtFq12(Fq12 {
            c0: fq6_pt,
            c1: Fq6::zero(),
        })
    }

    fn conjugate(&self) -> Self {
        FieldExtFq12(Fq12::conjugate(&self.0))
    }

    fn frobenius_map(&self, _power: Option<usize>) -> Self {
        FieldExtFq12(Fq12::frobenius_map(&self.0))
    }

    fn mul_base(&self, rhs: &Self::BaseField) -> Self {
        let fq6_pt = Fq6 {
            c0: *rhs,
            c1: Fq2::zero(),
            c2: Fq2::zero(),
        };
        FieldExtFq12(Fq12 {
            c0: self.0.c0 * fq6_pt,
            c1: self.0.c1 * fq6_pt,
        })
    }
}

impl LineMType<Fq, FieldExtFq2, FieldExtFq12> for FieldExtFq12 {
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fq, FieldExtFq2>) -> Self {
        FieldExtFq12(Fq12::from_coeffs(
            &[line.c, Fq2::ZERO, line.b, Fq2::ONE, Fq2::ZERO, Fq2::ZERO].map(|x| FieldExtFq2(x).0),
        ))
    }
}

impl ExpBigInt<FieldExtFq12> for FieldExtFq12 {}
