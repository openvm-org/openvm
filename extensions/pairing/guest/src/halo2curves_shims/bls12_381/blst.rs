use core::mem::MaybeUninit;

use blst::{
    blst_bendian_from_fp, blst_fp, blst_fp12, blst_fp12_inverse, blst_fp12_is_equal, blst_fp12_mul,
    blst_fp12_one, blst_fp12_sqr, blst_fp2, blst_fp6, blst_fp_from_bendian,
};
use halo2curves_axiom::bls12_381::{Fq, Fq12, Fq2, Fq6};

use super::final_exp::{
    FINAL_EXP_FACTOR_NAF, FINAL_EXP_TIMES_27_MOD_POLY_NAF, LAMBDA_INV_FINAL_EXP_NAF,
    POLY_FACTOR_NAF, TEN_NAF, TWENTY_SEVEN_NAF,
};

const FQ_BYTES: usize = 48;
const FP12_ZERO: blst_fp12 = blst_fp12 {
    fp6: [blst_fp6 {
        fp2: [blst_fp2 {
            fp: [blst_fp { l: [0; 6] }; 2],
        }; 3],
    }; 2],
};

#[derive(Clone, Copy)]
struct BlstFp12(blst_fp12);

impl BlstFp12 {
    fn one() -> Self {
        // BLST owns this immutable process-wide constant.
        Self(unsafe { *blst_fp12_one() })
    }

    fn square(self) -> Self {
        let mut output = MaybeUninit::uninit();
        unsafe {
            blst_fp12_sqr(output.as_mut_ptr(), &self.0);
            Self(output.assume_init())
        }
    }

    fn mul(self, rhs: Self) -> Self {
        let mut output = MaybeUninit::uninit();
        unsafe {
            blst_fp12_mul(output.as_mut_ptr(), &self.0, &rhs.0);
            Self(output.assume_init())
        }
    }

    fn invert_or_one(self) -> Self {
        if self.is_zero() {
            return Self::one();
        }

        let mut output = MaybeUninit::uninit();
        unsafe {
            blst_fp12_inverse(output.as_mut_ptr(), &self.0);
            Self(output.assume_init())
        }
    }

    fn is_one(self) -> bool {
        unsafe { blst_fp12_is_equal(&self.0, blst_fp12_one()) }
    }

    fn is_zero(self) -> bool {
        unsafe { blst_fp12_is_equal(&self.0, &FP12_ZERO) }
    }

    fn exp_naf(self, is_positive: bool, digits: &[i8]) -> Self {
        if digits.is_empty() {
            return Self::one();
        }

        let base = if is_positive {
            self
        } else {
            self.invert_or_one()
        };
        let base_inv = digits.contains(&-1).then(|| base.invert_or_one());

        let mut result = Self::one();
        for &digit in digits.iter().rev() {
            result = result.square();
            if digit == 1 {
                result = result.mul(base);
            } else if digit == -1 {
                result = result.mul(
                    base_inv.expect("negative digit requires an inverse of the exponent base"),
                );
            }
        }
        result
    }
}

impl From<&Fq12> for BlstFp12 {
    fn from(value: &Fq12) -> Self {
        Self(blst_fp12 {
            fp6: [fq6_to_blst(&value.c0), fq6_to_blst(&value.c1)],
        })
    }
}

impl From<BlstFp12> for Fq12 {
    fn from(value: BlstFp12) -> Self {
        Self {
            c0: fq6_from_blst(&value.0.fp6[0]),
            c1: fq6_from_blst(&value.0.fp6[1]),
        }
    }
}

fn fq6_to_blst(value: &Fq6) -> blst_fp6 {
    blst_fp6 {
        fp2: [
            fq2_to_blst(&value.c0),
            fq2_to_blst(&value.c1),
            fq2_to_blst(&value.c2),
        ],
    }
}

fn fq6_from_blst(value: &blst_fp6) -> Fq6 {
    Fq6 {
        c0: fq2_from_blst(&value.fp2[0]),
        c1: fq2_from_blst(&value.fp2[1]),
        c2: fq2_from_blst(&value.fp2[2]),
    }
}

fn fq2_to_blst(value: &Fq2) -> blst_fp2 {
    blst_fp2 {
        fp: [fq_to_blst(&value.c0), fq_to_blst(&value.c1)],
    }
}

fn fq2_from_blst(value: &blst_fp2) -> Fq2 {
    Fq2 {
        c0: fq_from_blst(&value.fp[0]),
        c1: fq_from_blst(&value.fp[1]),
    }
}

fn fq_to_blst(value: &Fq) -> blst_fp {
    // Canonical encoding keeps the conversion independent of each field's
    // in-memory representation.
    let bytes = value.to_bytes_be();
    let mut output = MaybeUninit::uninit();
    unsafe {
        blst_fp_from_bendian(output.as_mut_ptr(), bytes.as_ptr());
        output.assume_init()
    }
}

fn fq_from_blst(value: &blst_fp) -> Fq {
    let mut bytes = [0; FQ_BYTES];
    unsafe { blst_bendian_from_fp(bytes.as_mut_ptr(), value) };
    Option::from(Fq::from_bytes_be(&bytes)).expect("BLST produced a non-canonical field element")
}

pub(super) fn final_exp_hint(f: &Fq12) -> (Fq12, Fq12) {
    let f = BlstFp12::from(f);
    let f_final_exp = f.exp_naf(true, &FINAL_EXP_FACTOR_NAF);
    let root = f_final_exp.exp_naf(true, &TWENTY_SEVEN_NAF);

    let root_pth_inv = if root.is_one() {
        BlstFp12::one()
    } else {
        root.exp_naf(false, &FINAL_EXP_TIMES_27_MOD_POLY_NAF)
    };

    let root = f_final_exp.exp_naf(true, &POLY_FACTOR_NAF);
    let root_27th_inv = if root.exp_naf(true, &TWENTY_SEVEN_NAF).is_one() {
        root.exp_naf(false, &TEN_NAF)
    } else {
        BlstFp12::one()
    };

    let s = root_pth_inv.mul(root_27th_inv);
    let c = f.mul(s).exp_naf(true, &LAMBDA_INV_FINAL_EXP_NAF);

    (c.into(), s.into())
}
