pub use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, Fq6},
    ff::Field,
};

use crate::{
    field::{ExpBigInt, FieldExtension},
    pairing::{EvaluatedLine, LineDType},
};

/// FieldExtension for Fq12 with Fq6 as base field since halo2curves does not implement `Field` for Fq6.
impl FieldExtension for Fq12 {
    type BaseField = Fq2;
    type Coeffs = [Self::BaseField; 6];

    fn from_coeffs(coeffs: Self::Coeffs) -> Self {
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

    fn to_coeffs(self) -> Self::Coeffs {
        [
            self.c0.c0, self.c1.c0, self.c0.c1, self.c1.c1, self.c0.c2, self.c1.c2,
        ]
    }

    fn embed(base_elem: Self::BaseField) -> Self {
        let fq6_pt = Fq6 {
            c0: base_elem,
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

    fn mul_base(&self, rhs: Self::BaseField) -> Self {
        let fq6_pt = Fq6 {
            c0: rhs,
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
        Fq12::from_coeffs([Fq2::ONE, line.b, Fq2::ZERO, line.c, Fq2::ZERO, Fq2::ZERO])
    }
}

impl ExpBigInt<Fq12> for Fq12 {}
