extern crate alloc;

use alloc::vec::Vec;

use itertools::izip;
use openvm_algebra_guest::{field::FieldExtension, DivUnsafe, Field};
use openvm_ecc_guest::AffinePoint;
use openvm_pairing_guest::{
    bn254::{BN254_PSEUDO_BINARY_ENCODING, BN254_SEED},
    pairing::{
        exp_check_fallback, Evaluatable, EvaluatedLine, FromLineDType, LineMulDType, MillerStep,
        MultiMillerLoop, PairingCheck, PairingCheckError, PairingIntrinsics, UnevaluatedLine,
    },
};
#[cfg(all(feature = "halo2curves", not(target_os = "zkvm")))]
use openvm_pairing_guest::{
    halo2curves_shims::bn254::Bn254 as Halo2CurvesBn254, pairing::FinalExp,
};
#[cfg(target_os = "zkvm")]
use {
    core::mem::MaybeUninit,
    openvm_pairing_guest::{PairingBaseFunct7, OPCODE, PAIRING_FUNCT3},
    openvm_platform::custom_insn_r,
    openvm_rv32im_guest::hint_buffer_u32,
};

use super::{Bn254, Fp, Fp12, Fp2};
#[cfg(all(feature = "halo2curves", not(target_os = "zkvm")))]
use crate::bn254::utils::{
    convert_bn254_fp2_to_halo2_fq2, convert_bn254_fp_to_halo2_fq, convert_bn254_halo2_fq12_to_fp12,
};

impl Evaluatable<Fp, Fp2> for UnevaluatedLine<Fp2> {
    fn evaluate(&self, xy_frac: &(Fp, Fp)) -> EvaluatedLine<Fp2> {
        let (x_over_y, y_inv) = xy_frac;
        // Represents the line L(x,y) = 1 + b (x/y) w^1 + c (1/y) w^3
        EvaluatedLine {
            b: self.b.mul_base(x_over_y),
            c: self.c.mul_base(y_inv),
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

    /// Multiplies a line in 013-form with a Fp12 element to get an Fp12 element
    fn mul_by_013(f: &Fp12, l: &EvaluatedLine<Fp2>) -> Fp12 {
        Fp12::from_evaluated_line_d_type(l.clone()) * f
    }

    /// Multiplies a line in 01234-form with a Fp12 element to get an Fp12 element
    fn mul_by_01234(f: &Fp12, x: &[Fp2; 5]) -> Fp12 {
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

        let self_coeffs = &f.c;
        let s0 = &self_coeffs[0];
        let s1 = &self_coeffs[2];
        let s2 = &self_coeffs[4];
        let s3 = &self_coeffs[1];
        let s4 = &self_coeffs[3];
        let s5 = &self_coeffs[5];

        // NOTE[yj]: Hand-calculated multiplication for Fp12 * 01234 ∈ Fp2; this is likely not the
        // most efficient implementation c00 = cs0co0 + xi(cs1co2 + cs2co1 + cs4co4 +
        // cs5co3) c01 = cs0co1 + cs1co0 + cs3co3 + xi(cs2co2 + cs5co4)
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
}

#[allow(non_snake_case)]
impl MultiMillerLoop for Bn254 {
    type Fp = Fp;
    type Fp12 = Fp12;

    const SEED_ABS: u64 = BN254_SEED;
    const PSEUDO_BINARY_ENCODING: &[i8] = &BN254_PSEUDO_BINARY_ENCODING;

    fn evaluate_lines_vec(f: Self::Fp12, lines: Vec<EvaluatedLine<Self::Fp2>>) -> Self::Fp12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = Self::mul_by_013(&f, &lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = Self::mul_013_by_013(line0, line1);
                f = Self::mul_by_01234(&f, &prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    /// The expected output of this function when running the Miller loop with embedded exponent is
    /// c^2 * l_{2Q}
    fn pre_loop(
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        _Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        let mut f = if let Some(mut c) = c {
            // for the miller loop with embedded exponent, f will be set to c at the beginning of
            // the function, and we will square c due to the last two values of the
            // pseudo-binary encoding (BN254_PSEUDO_BINARY_ENCODING) being 0 and 1.
            // Therefore, the final value of f at the end of this block is c^2.
            c.square_assign();
            c
        } else {
            Self::Fp12::ONE
        };

        let mut Q_acc = Q_acc;
        let mut initial_lines = Vec::<EvaluatedLine<Self::Fp2>>::new();

        // We don't need to special case the first iteration for Bn254, but since we are using the
        // same Miller loop implementation for both Bn254 and Bls12_381, we need to do the
        // first iteration separately here.
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

    /// Compute f_{Miller,Q}(P) from f_{6x+2,Q}(P)
    fn post_loop(
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>, // at this point, Q_acc = (6x+2)Q
        Q: &[AffinePoint<Self::Fp2>],
        _c: Option<Self::Fp12>,
        xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        let mut Q_acc = Q_acc;
        let mut lines = Vec::<EvaluatedLine<Self::Fp2>>::new();

        let x_to_q_minus_1_over_3 = &Self::FROBENIUS_COEFF_FQ6_C1[1];
        let x_to_q_sq_minus_1_over_3 = &Self::FROBENIUS_COEFF_FQ6_C1[2];

        // For each q, compute q1 such that `frob_p(twist(q)) = twist(q1)`
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

        // compute l_{(6x+2)\Psi(Q), \phi_p(\Psi(Q))} where \phi_p is the Frobenius map
        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(q1_vec.iter())
            .map(|(Q_acc, q1)| Self::miller_add_step(Q_acc, q1))
            .unzip::<_, _, Vec<_>, Vec<_>>();
        Q_acc = Q_out_add;

        let lines_iter = izip!(lines_S_plus_Q.iter(), xy_fracs.iter());
        for (lines_S_plus_Q, xy_frac) in lines_iter {
            let line = lines_S_plus_Q.evaluate(xy_frac);
            lines.push(line);
        }

        // For each q, compute q2 such that `-frob_p^2(twist(q)) = twist(q2)`
        let q2_vec = Q
            .iter()
            .map(|Q| {
                // There is a frobenius mapping π²(Q) that we skip here since it is equivalent to
                // the identity mapping
                let x = &Q.x * x_to_q_sq_minus_1_over_3;
                AffinePoint { x, y: Q.y.clone() }
            })
            .collect::<Vec<_>>();

        // compute l_{(6x+2)\Psi(Q) + \phi_p(\Psi(Q)), -(\phi_p)^2(\Psi(Q))} where \phi_p is the
        // Frobenius map
        let (Q_out_add, lines_S_plus_Q) = Q_acc
            .iter()
            .zip(q2_vec.iter())
            .map(|(Q_acc, q2)| Self::miller_add_step(Q_acc, q2))
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

#[allow(non_snake_case)]
impl PairingCheck for Bn254 {
    type Fp = Fp;
    type Fp2 = Fp2;
    type Fp12 = Fp12;

    #[allow(unused_variables)]
    fn pairing_check_hint(
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
    ) -> (Self::Fp12, Self::Fp12) {
        #[cfg(not(target_os = "zkvm"))]
        {
            #[cfg(not(feature = "halo2curves"))]
            panic!("`halo2curves` feature must be enabled to use pairing check hint on host");

            #[cfg(feature = "halo2curves")]
            {
                let p_halo2 = P
                    .iter()
                    .map(|p| {
                        AffinePoint::new(
                            convert_bn254_fp_to_halo2_fq(p.x.clone()),
                            convert_bn254_fp_to_halo2_fq(p.y.clone()),
                        )
                    })
                    .collect::<Vec<_>>();
                let q_halo2 = Q
                    .iter()
                    .map(|q| {
                        AffinePoint::new(
                            convert_bn254_fp2_to_halo2_fq2(q.x.clone()),
                            convert_bn254_fp2_to_halo2_fq2(q.y.clone()),
                        )
                    })
                    .collect::<Vec<_>>();
                let fq12 = Halo2CurvesBn254::multi_miller_loop(&p_halo2, &q_halo2);
                let (c_fq12, s_fq12) = Halo2CurvesBn254::final_exp_hint(&fq12);
                let c = convert_bn254_halo2_fq12_to_fp12(c_fq12);
                let s = convert_bn254_halo2_fq12_to_fp12(s_fq12);
                (c, s)
            }
        }
        #[cfg(target_os = "zkvm")]
        {
            let hint = MaybeUninit::<(Fp12, Fp12)>::uninit();
            // We do not rely on the slice P's memory layout since rust does not guarantee it across
            // compiler versions.
            let p_fat_ptr = (P.as_ptr() as u32, P.len() as u32);
            let q_fat_ptr = (Q.as_ptr() as u32, Q.len() as u32);
            unsafe {
                custom_insn_r!(
                    opcode = OPCODE,
                    funct3 = PAIRING_FUNCT3,
                    funct7 = ((Bn254::PAIRING_IDX as u8) * PairingBaseFunct7::PAIRING_MAX_KINDS + PairingBaseFunct7::HintFinalExp as u8),
                    rd = Const "x0",
                    rs1 = In &p_fat_ptr,
                    rs2 = In &q_fat_ptr
                );
                let ptr = hint.as_ptr() as *const u8;
                hint_buffer_u32!(ptr, (32 * 12 * 2) / 4);
                hint.assume_init()
            }
        }
    }

    fn pairing_check(
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
    ) -> Result<(), PairingCheckError> {
        Self::try_honest_pairing_check(P, Q).unwrap_or_else(|| {
            let f = Self::multi_miller_loop(P, Q);
            exp_check_fallback(&f, &Self::FINAL_EXPONENT)
        })
    }
}

#[allow(non_snake_case)]
impl Bn254 {
    fn try_honest_pairing_check(
        P: &[AffinePoint<<Self as PairingCheck>::Fp>],
        Q: &[AffinePoint<<Self as PairingCheck>::Fp2>],
    ) -> Option<Result<(), PairingCheckError>> {
        let (c, u) = Self::pairing_check_hint(P, Q);
        if c == Fp12::ZERO {
            return None;
        }
        let c_inv = Fp12::ONE.div_unsafe(&c);

        // We follow Theorem 3 of https://eprint.iacr.org/2024/640.pdf to check that the pairing equals 1
        // By the theorem, it suffices to provide c and u such that f * u == c^λ.
        // Since λ = 6x + 2 + q^3 - q^2 + q, we will check the equivalent condition:
        // f * c^-{6x + 2} * u * c^-{q^3 - q^2 + q} == 1
        // This is because we can compute f * c^-{6x+2} by embedding the c^-{6x+2} computation in
        // the miller loop.

        // c_mul = c^-{q^3 - q^2 + q}
        let c_q3_inv = FieldExtension::frobenius_map(&c_inv, 3);
        let c_q2 = FieldExtension::frobenius_map(&c, 2);
        let c_q_inv = FieldExtension::frobenius_map(&c_inv, 1);
        let c_mul = c_q3_inv * c_q2 * c_q_inv;

        // Pass c inverse into the miller loop so that we compute fc == f * c^-{6x + 2}
        let fc = Self::multi_miller_loop_embedded_exp(P, Q, Some(c_inv));

        if fc * c_mul * u == Fp12::ONE {
            Some(Ok(()))
        } else {
            None
        }
    }
}
