use alloc::vec::Vec;

use axvm_algebra::{field::FieldExtension, Field};
use itertools::izip;
#[cfg(target_os = "zkvm")]
use {
    crate::pairing::shifted_funct7,
    axvm_platform::constants::{Custom1Funct3, PairingBaseFunct7, CUSTOM_1},
    axvm_platform::custom_insn_r,
    core::mem::MaybeUninit,
};

use super::{Bn254, Fp, Fp12, Fp2};
#[cfg(not(target_os = "zkvm"))]
use crate::pairing::PairingIntrinsics;
use crate::{
    pairing::{
        Evaluatable, EvaluatedLine, FromLineDType, LineMulDType, MillerStep, MultiMillerLoop,
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
                shifted_funct7::<Bn254>(PairingBaseFunct7::EvaluateLine),
                uninit.as_mut_ptr(),
                self as *const UnevaluatedLine<Fp2>,
                xy_frac as *const (Fp, Fp)
            );
            unsafe { uninit.assume_init() }
        }
    }
}

impl FromLineDType<Fp2> for Fp12 {
    fn from_evaluated_line_d_type(line: EvaluatedLine<Fp2>) -> Fp12 {
        FieldExtension::<Fp2>::from_coeffs([
            Fp2::ONE,
            line.b,
            Fp2::ZERO,
            line.c,
            Fp2::ZERO,
            Fp2::ZERO,
        ])
    }
}

// TODO[jpw]: make this into a macro depending on P::PAIRING_IDX when we have more curves
impl LineMulDType<Fp2, Fp12> for Bn254 {
    /// Multiplies two lines in 013-form to get an element in 01234-form
    fn mul_013_by_013(l0: &EvaluatedLine<Fp2>, l1: &EvaluatedLine<Fp2>) -> [Fp2; 5] {
        #[cfg(not(target_os = "zkvm"))]
        {
            let b0 = &l0.b;
            let c0 = &l0.c;
            let b1 = &l1.b;
            let c1 = &l1.c;

            // where w⁶ = xi
            // l0 * l1 = 1 + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴ + (c0c1)w⁶
            //         = (1 + c0c1 * xi) + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴
            let x0 = Fp2::ONE + c0 * c1 * &Bn254::XI;
            let x1 = b0 + b1;
            let x2 = b0 * b1;
            let x3 = c0 + c1;
            let x4 = b0 * c1 + b1 * c0;

            [x0, x1, x2, x3, x4]
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<[Fp2; 5]> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::Pairing as usize,
                shifted_funct7::<Bn254>(PairingBaseFunct7::Mul013By013),
                uninit.as_mut_ptr(),
                l0 as *const EvaluatedLine<Fp2>,
                l1 as *const EvaluatedLine<Fp2>
            );
            unsafe { uninit.assume_init() }
        }
    }

    /// Multiplies a line in 013-form with a Fp12 element to get an Fp12 element
    fn mul_by_013(f: &Fp12, l: &EvaluatedLine<Fp2>) -> Fp12 {
        #[cfg(not(target_os = "zkvm"))]
        {
            Fp12::from_evaluated_line_d_type(l.clone()) * f
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Fp12> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::Pairing as usize,
                shifted_funct7::<Bn254>(PairingBaseFunct7::MulBy013),
                uninit.as_mut_ptr(),
                f as *const Fp12,
                l as *const EvaluatedLine<Fp2>
            );
            unsafe { uninit.assume_init() }
        }
    }

    /// Multiplies a line in 01234-form with a Fp12 element to get an Fp12 element
    fn mul_by_01234(f: &Fp12, x: &[Fp2; 5]) -> Fp12 {
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
            let o0 = &x[0];
            let o1 = &x[2];
            let o2 = &x[4];
            let o3 = &x[1];
            let o4 = &x[3];

            let xi = &Bn254::XI;

            let self_coeffs = f.clone().to_coeffs();
            let s0 = &self_coeffs[0];
            let s1 = &self_coeffs[2];
            let s2 = &self_coeffs[4];
            let s3 = &self_coeffs[1];
            let s4 = &self_coeffs[3];
            let s5 = &self_coeffs[5];

            // NOTE[yj]: Hand-calculated multiplication for Fp12 * 01234 ∈ Fp2; this is likely not the most efficient implementation
            // c00 = cs0co0 + xi(cs1co2 + cs2co1 + cs4co4 + cs5co3)
            // c01 = cs0co1 + cs1co0 + cs3co3 + xi(cs2co2 + cs5co4)
            // c02 = cs0co2 + cs1co1 + cs2co0 + cs3co4 + cs4co3
            // c10 = cs0co3 + cs3co0 + xi(cs2co4 + cs4co2 + cs5co1)
            // c11 = cs0co4 + cs1co3 + cs3co1 + cs4co0 + xi(cs5co2)
            // c12 = cs1co4 + cs2co3 + cs3co2 + cs4co1 + cs5co0
            let c00 = s0 * o0 + xi * &(s1 * o2 + s2 * o1 + s4 * o4 + s5 * o3);
            let c01 = s0 * o1 + s1 * o0 + s3 * o3 + xi * &(s2 * o2 + s5 * o4);
            let c02 = s0 * o2 + s1 * o1 + s2 * o0 + s3 * o4 + s4 * o3;
            let c10 = s0 * o3 + s3 * o0 + xi * &(s2 * o4 + s4 * o2 + s5 * o1);
            let c11 = s0 * o4 + s1 * o3 + s3 * o1 + s4 * o0 + xi * &(s5 * o2);
            let c12 = s1 * o4 + s2 * o3 + s3 * o2 + s4 * o1 + s5 * o0;

            Fp12::from_coeffs([c00, c10, c01, c11, c02, c12])
        }
        #[cfg(target_os = "zkvm")]
        {
            let mut uninit: MaybeUninit<Fp12> = MaybeUninit::uninit();
            custom_insn_r!(
                CUSTOM_1,
                Custom1Funct3::Pairing as usize,
                shifted_funct7::<Bn254>(PairingBaseFunct7::MulBy01234),
                uninit.as_mut_ptr(),
                f as *const Fp12,
                x as *const [Fp2; 5]
            );
            unsafe { uninit.assume_init() }
        }
    }
}

#[allow(non_snake_case)]
impl MultiMillerLoop for Bn254 {
    type Fp = Fp;
    type Fp12 = Fp12;

    const SEED_ABS: u64 = 0x44e992b44a6909f1;
    const PSEUDO_BINARY_ENCODING: &[i8] = &[
        0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0,
        0, -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0,
        -1, 0, 0, 0, 1, 0, -1, 0, 1,
    ];

    fn evaluate_lines_vec(f: Self::Fp12, lines: Vec<EvaluatedLine<Self::Fp2>>) -> Self::Fp12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = Self::mul_by_013(&f, &lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = Self::mul_013_by_013(&line0, &line1);
                f = Self::mul_by_01234(&f, &prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    fn pre_loop(
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        _Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        let mut f = f.clone();

        if c.is_some() {
            f.square_assign();
        }

        let mut Q_acc = Q_acc;
        let mut initial_lines = Vec::<EvaluatedLine<Self::Fp2>>::new();

        let (Q_out_double, lines_2S) = Q_acc
            .into_iter()
            .map(|Q| Self::miller_double_step(&Q))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_double;

        let lines_iter = izip!(lines_2S.iter(), xy_fracs.iter());
        for (line_2S, xy_frac) in lines_iter {
            let line = line_2S.evaluate(xy_frac);
            initial_lines.push(line);
        }

        f = Self::evaluate_lines_vec(f, initial_lines);

        (f, Q_acc)
    }

    fn post_loop(
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        Q: &[AffinePoint<Self::Fp2>],
        _c: Option<Self::Fp12>,
        xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        let mut Q_acc = Q_acc;
        let mut lines = Vec::<EvaluatedLine<Self::Fp2>>::new();

        let x_to_q_minus_1_over_3 = &Self::FROBENIUS_COEFF_FQ6_C1[1];
        let x_to_q_sq_minus_1_over_3 = &Self::FROBENIUS_COEFF_FQ6_C1[2];
        let q1_vec = Q
            .iter()
            .map(|Q| {
                let x = Q.x.frobenius_map(1);
                let x = x * x_to_q_minus_1_over_3;
                let y = Q.y.frobenius_map(1);
                let y = y * &Self::XI_TO_Q_MINUS_1_OVER_2;
                AffinePoint { x, y }
            })
            .collect::<Vec<_>>();

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(q1_vec.iter())
            .map(|(Q_acc, q1)| Self::miller_add_step(&Q_acc, &q1))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), xy_fracs.iter());
        for (lines_S_plus_Q, xy_frac) in lines_iter {
            let line = lines_S_plus_Q.evaluate(xy_frac);
            lines.push(line);
        }

        let q2_vec = Q
            .iter()
            .map(|Q| {
                // There is a frobenius mapping π²(Q) that we skip here since it is equivalent to the identity mapping
                let x = &Q.x * &x_to_q_sq_minus_1_over_3;
                AffinePoint { x, y: Q.y.clone() }
            })
            .collect::<Vec<_>>();

        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(q2_vec.iter())
            .map(|(Q_acc, q2)| Self::miller_add_step(&Q_acc, &q2))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), xy_fracs.iter());
        for (lines_S_plus_Q, xy_frac) in lines_iter {
            let line = lines_S_plus_Q.evaluate(xy_frac);
            lines.push(line);
        }

        let mut f = f.clone();
        f = Self::evaluate_lines_vec(f, lines);

        (f, Q_acc)
    }
}
