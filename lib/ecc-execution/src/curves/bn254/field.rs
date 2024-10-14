use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, Fq6},
    ff::Field,
};
use num::BigInt;

#[cfg(test)]
use crate::common::FeltPrint;
use crate::common::{
    EvaluatedLine, ExpBigInt, FieldExtension, Fp12Constructor, Fp2Constructor, LineDType,
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

impl ExpBigInt<Fq12> for Fq12 {
    fn exp_bigint(&self, k: BigInt) -> Fq12 {
        // let (sign, digits) = k.to_u64_digits();
        // let mut res = self.pow_vartime(digits);
        // if sign == Sign::Minus {
        //     res = res.invert().unwrap();
        // }
        // res
        if k == BigInt::from(0) {
            return Fq12::one();
        }

        let mut e = k.clone();
        let mut x = *self;

        if k < BigInt::from(0) {
            x = x.invert().unwrap();
            e = -k;
        }

        let mut res = Fq12::one();

        let x_sq = x.square();
        let ops = [x, x_sq, x_sq * x];

        let bytes = e.to_bytes_be();
        for &b in bytes.1.iter() {
            let mut mask = 0xc0;
            for j in 0..4 {
                res = res.square().square();
                let c = (b & mask) >> (6 - 2 * j);
                if c != 0 {
                    res *= &ops[(c - 1) as usize];
                }
                mask >>= 2;
            }
        }

        res
    }
}

#[cfg(test)]
impl FeltPrint<Fq> for Fq {
    fn felt_print(&self, label: &str) {
        println!("{} {:?}", label, self.0);
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
