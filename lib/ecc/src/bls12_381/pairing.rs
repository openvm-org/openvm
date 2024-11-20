use alloc::vec::Vec;

use axvm_algebra::{
    field::{ComplexConjugate, FieldExtension},
    Field,
};
use itertools::izip;
#[cfg(target_os = "zkvm")]
use {
    crate::pairing::shifted_funct7,
    axvm_platform::constants::{Custom1Funct3, PairingBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

use super::{Bls12_381, Fp, Fp12, Fp2};
#[cfg(not(target_os = "zkvm"))]
use crate::pairing::PairingIntrinsics;
use crate::{
    pairing::{
        Evaluatable, EvaluatedLine, FromLineMType, LineMulMType, MillerStep, MultiMillerLoop,
        UnevaluatedLine,
    },
    AffinePoint,
};

// TODO[jpw]: make macro
impl Evaluatable<Fp, Fp2> for UnevaluatedLine<Fp2> {
    fn evaluate(&self, xy_frac: &(Fp, Fp)) -> EvaluatedLine<Fp2> {
        #[cfg(not(target_os = "zkvm"))]
        {
            let (x_over_y, y_inv) = xy_frac;
            EvaluatedLine {
                b: self.b.mul_base(x_over_y),
                c: self.c.mul_base(y_inv),
            }
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<EvaluatedLine<Fp2>> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::Pairing as usize,
                shifted_funct7::<Bls12_381>(PairingBaseFunct7::EvaluateLine),
                uninit.as_mut_ptr(),
                self as *const UnevaluatedLine<Fp2>,
                xy_frac as *const (Fp, Fp)
            );
            unsafe { uninit.assume_init() }
        }
    }
}

impl FromLineMType<Fp2> for Fp12 {
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fp2>) -> Fp12 {
        Fp12::from_coeffs([line.c, Fp2::ZERO, line.b, Fp2::ONE, Fp2::ZERO, Fp2::ZERO])
    }
}

// TODO[jpw]: make this into a macro depending on P::PAIRING_IDX when we have more curves
impl LineMulMType<Fp2, Fp12> for Bls12_381 {
    /// Multiplies two lines in 023-form to get an element in 02345-form
    fn mul_023_by_023(l0: &EvaluatedLine<Fp2>, l1: &EvaluatedLine<Fp2>) -> [Fp2; 5] {
        #[cfg(not(target_os = "zkvm"))]
        {
            let b0 = &l0.b;
            let c0 = &l0.c;
            let b1 = &l1.b;
            let c1 = &l1.c;

            // where w⁶ = xi
            // l0 * l1 = c0c1 + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 +b1)w⁵ + w⁶
            //         = (c0c1 + xi) + (c0b1 + c1b0)w² + (c0 + c1)w³ + (b0b1)w⁴ + (b0 + b1)w⁵
            let x0 = c0 * c1 + Bls12_381::XI;
            let x2 = c0 * b1 + c1 * b0;
            let x3 = c0 + c1;
            let x4 = b0 * b1;
            let x5 = b0 + b1;

            [x0, x2, x3, x4, x5]
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<[Fp2; 5]> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::Pairing as usize,
                shifted_funct7::<Bls12_381>(PairingBaseFunct7::Mul023By023),
                uninit.as_mut_ptr(),
                l0 as *const EvaluatedLine<Fp2>,
                l1 as *const EvaluatedLine<Fp2>
            );
            unsafe { uninit.assume_init() }
        }
    }

    /// Multiplies a line in 02345-form with a Fp12 element to get an Fp12 element
    fn mul_by_023(f: &Fp12, l: &EvaluatedLine<Fp2>) -> Fp12 {
        #[cfg(not(target_os = "zkvm"))]
        {
            Fp12::from_evaluated_line_m_type(l.clone()) * f
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Fp12> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::Pairing as usize,
                shifted_funct7::<Bls12_381>(PairingBaseFunct7::MulBy023),
                uninit.as_mut_ptr(),
                f as *const Fp12,
                l as *const EvaluatedLine<Fp2>
            );
            unsafe { uninit.assume_init() }
        }
    }

    /// Multiplies a line in 02345-form with a Fp12 element to get an Fp12 element
    fn mul_by_02345(f: &Fp12, x: &[Fp2; 5]) -> Fp12 {
        #[cfg(not(target_os = "zkvm"))]
        {
            // we update the order of the coefficients to match the Fp12 coefficient ordering:
            // Fp12 {
            //   c0: Fp6 {
            //     c0: x0,
            //     c1: x2,
            //     c2: x4,
            //   },
            //   c1: Fp6 {
            //     c0: x1,
            //     c1: x3,
            //     c2: x5,
            //   },
            // }
            let o0 = &x[0]; // coeff x0
            let o1 = &x[1]; // coeff x2
            let o2 = &x[3]; // coeff x4
            let o4 = &x[2]; // coeff x3
            let o5 = &x[4]; // coeff x5

            let xi = &Bls12_381::XI;

            let self_coeffs = f.clone().to_coeffs();
            let s0 = &self_coeffs[0];
            let s1 = &self_coeffs[2];
            let s2 = &self_coeffs[4];
            let s3 = &self_coeffs[1];
            let s4 = &self_coeffs[3];
            let s5 = &self_coeffs[5];

            // NOTE[yj]: Hand-calculated multiplication for Fp12 * 02345 ∈ Fp2; this is likely not the most efficient implementation
            // c00 = cs0co0 + xi(cs1co2 + cs2co1 + cs3co5 + cs4co4)
            // c01 = cs0co1 + cs1co0 + xi(cs2co2 + cs4co5 + cs5co4)
            // c02 = cs0co2 + cs1co1 + cs2co0 + cs3co4 + xi(cs5co5)
            // c10 = cs3co0 + xi(cs1co5 + cs2co4 + cs4co2 + cs5co1)
            // c11 = cs0co4 + cs3co1 + cs4co0 + xi(cs2co5 + cs5co2)
            // c12 = cs0co5 + cs1co4 + cs3co2 + cs4co1 + cs5co0
            //   where cs*: self.c*
            let c00 = s0 * o0 + xi * &(s1 * o2 + s2 * o1 + s3 * o5 + s4 * o4);
            let c01 = s0 * o1 + s1 * o0 + xi * &(s2 * o2 + s4 * o5 + s5 * o4);
            let c02 = s0 * o2 + s1 * o1 + s2 * o0 + s3 * o4 + xi * &(s5 * o5);
            let c10 = s3 * o0 + xi * &(s1 * o5 + s2 * o4 + s4 * o2 + s5 * o1);
            let c11 = s0 * o4 + s3 * o1 + s4 * o0 + xi * &(s2 * o5 + s5 * o2);
            let c12 = s0 * o5 + s1 * o4 + s3 * o2 + s4 * o1 + s5 * o0;

            Fp12::from_coeffs([c00, c10, c01, c11, c02, c12])
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Fp12> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::Pairing as usize,
                shifted_funct7::<Bls12_381>(PairingBaseFunct7::MulBy02345),
                uninit.as_mut_ptr(),
                f as *const Fp12,
                x as *const [Fp2; 5]
            );
            unsafe { uninit.assume_init() }
        }
    }
}

#[allow(non_snake_case)]
impl MultiMillerLoop for Bls12_381 {
    type Fp = Fp;
    type Fp12 = Fp12;

    const SEED_ABS: u64 = 0xd201000000010000;
    const PSEUDO_BINARY_ENCODING: &[i8] = &[
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 1, 1,
    ];

    fn evaluate_lines_vec(f: Self::Fp12, lines: Vec<EvaluatedLine<Self::Fp2>>) -> Self::Fp12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = Self::mul_by_023(&f, &lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = Self::mul_023_by_023(&line0, &line1);
                f = Self::mul_by_02345(&f, &prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    /// The expected output of this function when running the Miller loop with embedded exponent is c^3 * l_{3Q}
    fn pre_loop(
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        let mut f = f.clone();

        if c.is_some() {
            // for the miller loop with embedded exponent, f will be set to c at the beginning of the function, and we
            // will multiply by c again due to the last two values of the pseudo-binary encoding (BN12_381_PBE) being 1.
            // Therefore, the final value of f at the end of this block is c^3.
            f.square_assign();
            f *= c.unwrap();
        }

        let mut Q_acc = Q_acc;

        // Special case the first iteration of the Miller loop with pseudo_binary_encoding = 1:
        // this means that the first step is a double and add, but we need to separate the two steps since the optimized
        // `miller_double_and_add_step` will fail because Q_acc is equal to Q_signed on the first iteration
        let (Q_out_double, lines_2S) = Q_acc
            .into_iter()
            .map(|Q| Self::miller_double_step(&Q))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_double;

        let mut initial_lines = Vec::<EvaluatedLine<Self::Fp2>>::new();

        let lines_iter = izip!(lines_2S.iter(), xy_fracs.iter());
        for (line_2S, xy_frac) in lines_iter {
            let line = line_2S.evaluate(xy_frac);
            initial_lines.push(line);
        }

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(Q.iter())
            .map(|(Q_acc, Q)| Self::miller_add_step(&Q_acc, &Q))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), xy_fracs.iter());
        for (lines_S_plus_Q, xy_frac) in lines_iter {
            let line = lines_S_plus_Q.evaluate(xy_frac);
            initial_lines.push(line);
        }

        f = Self::evaluate_lines_vec(f, initial_lines);

        (f, Q_acc)
    }

    /// After running the main body of the Miller loop, we conjugate f due to the curve seed x being negative.
    fn post_loop(
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        _Q: &[AffinePoint<Self::Fp2>],
        _c: Option<Self::Fp12>,
        _xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        // Conjugate for negative component of the seed
        let mut f = f.clone();
        f.conjugate_assign();
        (f, Q_acc)
    }
}
