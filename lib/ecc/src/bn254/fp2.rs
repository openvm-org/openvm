use core::ops::Neg;

use axvm_algebra::{
    field::{Complex, FieldExtension},
    Field, IntMod,
};

use super::{fp_invert_assign, Fp};

pub type Fp2 = Complex<Fp>;

impl FieldExtension<Fp> for Fp2 {
    const D: usize = 2;
    type Coeffs = [Fp; 2];

    fn from_coeffs([c0, c1]: Self::Coeffs) -> Self {
        Self { c0, c1 }
    }

    fn to_coeffs(self) -> Self::Coeffs {
        [self.c0, self.c1]
    }

    fn embed(c0: Fp) -> Self {
        Self {
            c0,
            c1: <Fp as Field>::ZERO,
        }
    }

    fn frobenius_map(&self, power: usize) -> Self {
        if power % 2 == 0 {
            self.clone()
        } else {
            Self {
                c0: self.c0.clone(),
                c1: (&self.c1).neg(),
            }
        }
    }

    fn mul_base(&self, rhs: &Fp) -> Self {
        Self {
            c0: &self.c0 * rhs,
            c1: &self.c1 * rhs,
        }
    }
}

// pub(crate) fn fp2_mul_by_nonresidue_assign(x: &mut Fp2) {
//     // (xu+y)(u+9) = (9x+y)u+(9y-x)
//     let t0 = x.c0.clone();
//     let t1 = x.c1.clone();

//     // 8*x*i + 8*y
//     x.double_assign();
//     x.double_assign();
//     x.double_assign();

//     // 9x + y
//     x.c0 += &t0;
//     // (9&y - x)
//     x.c1 -= &t1;

//     // (9*x)u
//     x.c1 += &t1;
//     // (9*x + y)
//     x.c1 += &t0;
// }

pub(crate) fn fp2_invert_assign(x: &mut Fp2) {
    let mut t1 = x.c1.clone();
    <Fp as Field>::square_assign(&mut t1);
    let mut t0 = x.c0.clone();
    <Fp as Field>::square_assign(&mut t0);
    t0 += &t1;
    fp_invert_assign(&mut t0);
    let mut tmp = Fp2 {
        c0: x.c0.clone(),
        c1: x.c1.clone(),
    };
    tmp.c0 *= &t0;
    tmp.c1 *= &t0;
    tmp.c1.neg_assign();

    *x = tmp;
}
