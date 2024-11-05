use halo2curves_axiom::{
    bls12_381::{Fq, Fq12, Fq2, Fq6},
    ff::Field,
};

#[cfg(test)]
use crate::common::FeltPrint;
use crate::common::{
    EvaluatedLine, ExpBigInt, FieldExtension, Fp12Constructor, Fp2Constructor, LineMType,
};

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
        Fq2::conjugate(self)
    }

    fn frobenius_map(&self, _power: Option<usize>) -> Self {
        Fq2::frobenius_map(self)
    }

    fn mul_base(&self, rhs: &Self::BaseField) -> Self {
        Fq2 {
            c0: self.c0 * rhs,
            c1: self.c1 * rhs,
        }
    }
}

///
/// Note that halo2curves does not implement `Field` for Fq6, so we need to implement the intermediate points manually.
///
/// FieldExtension for Fq12 with Fq2 as base field since halo2curves does not implement `Field` for Fq6.
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
        Fq12::conjugate(self)
    }

    fn frobenius_map(&self, _power: Option<usize>) -> Self {
        Fq12::frobenius_map(self)
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

impl LineMType<Fq, Fq2, Fq12> for Fq12 {
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fq, Fq2>) -> Fq12 {
        Fq12::from_coeffs(&[line.c, Fq2::ZERO, line.b, Fq2::ONE, Fq2::ZERO, Fq2::ZERO])
    }
}

impl ExpBigInt<Fq12> for Fq12 {}

#[cfg(test)]
impl FeltPrint<Fq> for Fq {
    fn felt_print(&self, label: &str) {
        println!("{} {:?}", label, self);
    }
}

#[cfg(test)]
impl FeltPrint<Fq12> for Fq12 {
    fn felt_print(&self, label: &str) {
        println!("felt_print - {}", label);
        print!("c0.c0.c0:");
        self.c0.c0.c0.felt_print("");
        print!("c0.c0.c1:");
        self.c0.c0.c1.felt_print("");
        print!("c0.c1.c0:");
        self.c0.c1.c0.felt_print("");
        print!("c0.c1.c1:");
        self.c0.c1.c1.felt_print("");
        print!("c0.c2.c0:");
        self.c0.c2.c0.felt_print("");
        print!("c0.c2.c1:");
        self.c0.c2.c1.felt_print("");
        print!("c1.c0.c0:");
        self.c1.c0.c0.felt_print("");
        print!("c1.c0.c1:");
        self.c1.c0.c1.felt_print("");
        print!("c1.c1.c0:");
        self.c1.c1.c0.felt_print("");
        print!("c1.c1.c1:");
        self.c1.c1.c1.felt_print("");
        print!("c1.c2.c0:");
        self.c1.c2.c0.felt_print("");
        print!("c1.c2.c1:");
        self.c1.c2.c1.felt_print("");
    }
}
