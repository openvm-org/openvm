mod curve;
mod final_exp;
mod line;
mod miller_loop;

pub use curve::*;
pub use line::*;

#[cfg(test)]
pub mod tests;

// Make public for use by tests in guest-libs/pairing/
pub mod test_utils;

use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2};
use openvm_algebra_guest::field::FieldExtension;

use crate::pairing::{Evaluatable, EvaluatedLine, FromLineMType, UnevaluatedLine};

impl FromLineMType<Fq2> for Fq12 {
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fq2>) -> Fq12 {
        Fq12::from_coeffs([
            line.c,
            Fq2::zero(),
            line.b,
            Fq2::one(),
            Fq2::zero(),
            Fq2::zero(),
        ])
    }
}

impl Evaluatable<Fq, Fq2> for UnevaluatedLine<Fq2> {
    fn evaluate(&self, xy_frac: &(Fq, Fq)) -> EvaluatedLine<Fq2> {
        let (x_over_y, y_inv) = xy_frac;
        EvaluatedLine {
            b: self.b.mul_base(x_over_y),
            c: self.c.mul_base(y_inv),
        }
    }
}
