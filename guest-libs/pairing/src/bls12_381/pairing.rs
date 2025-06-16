extern crate alloc;

use alloc::vec::Vec;

use itertools::izip;
use openvm_algebra_guest::{
    field::{ComplexConjugate, FieldExtension},
    DivUnsafe, Field,
};
use openvm_ecc_guest::AffinePoint;
use openvm_pairing_guest::{
    bls12_381::{BLS12_381_PSEUDO_BINARY_ENCODING, BLS12_381_SEED_ABS},
    pairing::{
        exp_check_fallback, Evaluatable, EvaluatedLine, FromLineMType, LineMulMType, MillerStep,
        MultiMillerLoop, PairingCheck, PairingCheckError, PairingIntrinsics, UnevaluatedLine,
    },
};
#[cfg(all(feature = "halo2curves", not(target_os = "zkvm")))]
use openvm_pairing_guest::{
    halo2curves_shims::bls12_381::Bls12_381 as Halo2CurvesBls12_381, pairing::FinalExp,
};
#[cfg(target_os = "zkvm")]
use {
    core::mem::MaybeUninit,
    openvm_pairing_guest::{PairingBaseFunct7, OPCODE, PAIRING_FUNCT3},
    openvm_platform::custom_insn_r,
    openvm_rv32im_guest,
    openvm_rv32im_guest::hint_buffer_u32,
};

use super::{Bls12_381, Fp, Fp12, Fp2};
#[cfg(all(feature = "halo2curves", not(target_os = "zkvm")))]
use crate::bls12_381::utils::{
    convert_bls12381_fp2_to_halo2_fq2, convert_bls12381_fp_to_halo2_fq,
    convert_bls12381_halo2_fq12_to_fp12,
};

impl Evaluatable<Fp, Fp2> for UnevaluatedLine<Fp2> {
    fn evaluate(&self, xy_frac: &(Fp, Fp)) -> EvaluatedLine<Fp2> {
        let (x_over_y, y_inv) = xy_frac;
        // Represents the line L(x,y) = 1 + b (x/y) w^-1 + c (1/y) w^-3
        EvaluatedLine {
            b: self.b.mul_base(x_over_y),
            c: self.c.mul_base(y_inv),
        }
    }
}

impl FromLineMType<Fp2> for Fp12 {
    // Since multiplying by w^3 doesn't change the miller loop result, we transform the line
    // into L_new(x,y) = w^3 L(x,y) = w^3 + b (x/y) w^2 + c (1/y)
    fn from_evaluated_line_m_type(line: EvaluatedLine<Fp2>) -> Fp12 {
        Fp12::from_coeffs([line.c, Fp2::ZERO, line.b, Fp2::ONE, Fp2::ZERO, Fp2::ZERO])
    }
}

// TODO[jpw]: make this into a macro depending on P::PAIRING_IDX when we have more curves
impl LineMulMType<Fp2, Fp12> for Bls12_381 {
    /// Multiplies two lines in 023-form to get an element in 02345-form
    fn mul_023_by_023(l0: &EvaluatedLine<Fp2>, l1: &EvaluatedLine<Fp2>) -> [Fp2; 5] {
        // l0 = c0 + b0 w^2 + w^3
        let b0 = &l0.b;
        let c0 = &l0.c;
        // l1 = c1 + b1 w^2 + w^3
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

    /// Multiplies a line in 02345-form with a Fp12 element to get an Fp12 element
    fn mul_by_023(f: &Fp12, l: &EvaluatedLine<Fp2>) -> Fp12 {
        // this is only used if the number of lines is odd, which doesn't happen for our
        // applications right now, so we can use this suboptimal implementation
        Fp12::from_evaluated_line_m_type(l.clone()) * f
    }

    /// Multiplies a line in 02345-form with a Fp12 element to get an Fp12 element
    fn mul_by_02345(f: &Fp12, x: &[Fp2; 5]) -> Fp12 {
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

        let self_coeffs = &f.c;
        let s0 = &self_coeffs[0];
        let s1 = &self_coeffs[2];
        let s2 = &self_coeffs[4];
        let s3 = &self_coeffs[1];
        let s4 = &self_coeffs[3];
        let s5 = &self_coeffs[5];

        // NOTE[yj]: Hand-calculated multiplication for Fp12 * 02345 ∈ Fp2; this is likely not the
        // most efficient implementation c00 = cs0co0 + xi(cs1co2 + cs2co1 + cs3co5 +
        // cs4co4) c01 = cs0co1 + cs1co0 + xi(cs2co2 + cs4co5 + cs5co4)
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
}

#[allow(non_snake_case)]
impl MultiMillerLoop for Bls12_381 {
    type Fp = Fp;
    type Fp12 = Fp12;

    const SEED_ABS: u64 = BLS12_381_SEED_ABS;
    const PSEUDO_BINARY_ENCODING: &[i8] = &BLS12_381_PSEUDO_BINARY_ENCODING;

    fn evaluate_lines_vec(f: Self::Fp12, lines: Vec<EvaluatedLine<Self::Fp2>>) -> Self::Fp12 {
        let mut f = f;
        let mut lines = lines;
        if lines.len() % 2 == 1 {
            f = Self::mul_by_023(&f, &lines.pop().unwrap());
        }
        for chunk in lines.chunks(2) {
            if let [line0, line1] = chunk {
                let prod = Self::mul_023_by_023(line0, line1);
                f = Self::mul_by_02345(&f, &prod);
            } else {
                panic!("lines.len() % 2 should be 0 at this point");
            }
        }
        f
    }

    /// The expected output of this function when running the Miller loop with embedded exponent is
    /// c^3 * l_{3Q}
    fn pre_loop(
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        Q: &[AffinePoint<Self::Fp2>],
        c: Option<Self::Fp12>,
        xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        let mut f = if let Some(mut c) = c {
            // for the miller loop with embedded exponent, f will be set to c at the beginning of
            // the function, and we will multiply by c again due to the last two values
            // of the pseudo-binary encoding (BLS12_381_PSEUDO_BINARY_ENCODING) being 1.
            // Therefore, the final value of f at the end of this block is c^3.
            let mut c3 = c.clone();
            c.square_assign();
            c3 *= &c;
            c3
        } else {
            Self::Fp12::ONE
        };

        let mut Q_acc = Q_acc;

        // Special case the first iteration of the Miller loop with pseudo_binary_encoding = 1:
        // this means that the first step is a double and add, but we need to separate the two steps
        // since the optimized `miller_double_and_add_step` will fail because Q_acc is equal
        // to Q_signed on the first iteration
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
            .map(|(Q_acc, Q)| Self::miller_add_step(Q_acc, Q))
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

    /// After running the main body of the Miller loop, we conjugate f due to the curve seed x being
    /// negative.
    fn post_loop(
        f: &Self::Fp12,
        Q_acc: Vec<AffinePoint<Self::Fp2>>,
        _Q: &[AffinePoint<Self::Fp2>],
        _c: Option<Self::Fp12>,
        _xy_fracs: &[(Self::Fp, Self::Fp)],
    ) -> (Self::Fp12, Vec<AffinePoint<Self::Fp2>>) {
        // Conjugate for negative component of the seed
        // By Lemma 1 from https://www.iacr.org/archive/eurocrypt2011/66320047/66320047.pdf f_{x,Q} = conjugate( f_{|x|,Q} )
        let mut f = f.clone();
        f.conjugate_assign();
        (f, Q_acc)
    }
}

#[allow(non_snake_case)]
impl PairingCheck for Bls12_381 {
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
                            convert_bls12381_fp_to_halo2_fq(p.x.clone()),
                            convert_bls12381_fp_to_halo2_fq(p.y.clone()),
                        )
                    })
                    .collect::<Vec<_>>();
                let q_halo2 = Q
                    .iter()
                    .map(|q| {
                        AffinePoint::new(
                            convert_bls12381_fp2_to_halo2_fq2(q.x.clone()),
                            convert_bls12381_fp2_to_halo2_fq2(q.y.clone()),
                        )
                    })
                    .collect::<Vec<_>>();
                let fq12 = Halo2CurvesBls12_381::multi_miller_loop(&p_halo2, &q_halo2);
                let (c_fq12, s_fq12) = Halo2CurvesBls12_381::final_exp_hint(&fq12);
                let c = convert_bls12381_halo2_fq12_to_fp12(c_fq12);
                let s = convert_bls12381_halo2_fq12_to_fp12(s_fq12);
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
                    funct7 = ((Bls12_381::PAIRING_IDX as u8) * PairingBaseFunct7::PAIRING_MAX_KINDS + PairingBaseFunct7::HintFinalExp as u8),
                    rd = Const "x0",
                    rs1 = In &p_fat_ptr,
                    rs2 = In &q_fat_ptr
                );
                let ptr = hint.as_ptr() as *const u8;
                hint_buffer_u32!(ptr, (48 * 12 * 2) / 4);
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
impl Bls12_381 {
    // The paper only describes the implementation for Bn254, so we use the gnark implementation for
    // Bls12_381. Adapted from the gnark implementation:
    // https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/fields_bls12381/e12_pairing.go#L394C1-L395C1
    fn try_honest_pairing_check(
        P: &[AffinePoint<<Self as PairingCheck>::Fp>],
        Q: &[AffinePoint<<Self as PairingCheck>::Fp2>],
    ) -> Option<Result<(), PairingCheckError>> {
        let (c, s) = Self::pairing_check_hint(P, Q);

        // The gnark implementation checks that f * s = c^{q - x} where x is the curve seed.
        // We check an equivalent condition: f * c^x * s = c^q.
        // This is because we can compute f * c^x by embedding the c^x computation in the miller
        // loop.

        // We compute c^q before c is consumed by conjugate() below
        let c_q = FieldExtension::frobenius_map(&c, 1);

        // Since the Bls12_381 curve has a negative seed, the miller loop for Bls12_381 is computed
        // as f_{Miller,x,Q}(P) = conjugate( f_{Miller,-x,Q}(P) * c^{-x} ).
        // We will pass in the conjugate inverse of c into the miller loop so that we compute
        // fc = conjugate( f_{Miller,-x,Q}(P) * c'^{-x} )  (where c' is the conjugate inverse of c)
        //    = f_{Miller,x,Q}(P) * c^x
        let c_conj = c.conjugate();
        if c_conj == Fp12::ZERO {
            return None;
        }
        let c_conj_inv = Fp12::ONE.div_unsafe(&c_conj);
        let fc = Self::multi_miller_loop_embedded_exp(P, Q, Some(c_conj_inv));

        if fc * s == c_q {
            Some(Ok(()))
        } else {
            None
        }
    }
}
