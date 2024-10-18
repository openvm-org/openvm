use std::{cell::RefCell, rc::Rc};

use super::Fp2;
use crate::field_expression::{ExprBuilder, FieldVariable};

/// Field extension of Fp12 defined with coefficients in Fp2. Fp6-equivalent coefficients are c0: (c0, c2, c4), c1: (c1, c3, c5).
pub struct Fp12 {
    pub c0: Fp2,
    pub c1: Fp2,
    pub c2: Fp2,
    pub c3: Fp2,
    pub c4: Fp2,
    pub c5: Fp2,
}

impl Fp12 {
    pub fn new(builder: Rc<RefCell<ExprBuilder>>) -> Self {
        let c0 = Fp2::new(builder.clone());
        let c1 = Fp2::new(builder.clone());
        let c2 = Fp2::new(builder.clone());
        let c3 = Fp2::new(builder.clone());
        let c4 = Fp2::new(builder.clone());
        let c5 = Fp2::new(builder.clone());
        Fp12 {
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
        }
    }

    pub fn save(&mut self) {
        self.c0.save();
        self.c1.save();
        self.c2.save();
        self.c3.save();
        self.c4.save();
        self.c5.save();
    }

    pub fn add(&mut self, other: &mut Fp12) -> Fp12 {
        Fp12 {
            c0: self.c0.add(&mut other.c0),
            c1: self.c1.add(&mut other.c1),
            c2: self.c2.add(&mut other.c2),
            c3: self.c3.add(&mut other.c3),
            c4: self.c4.add(&mut other.c4),
            c5: self.c5.add(&mut other.c5),
        }
    }

    pub fn sub(&mut self, other: &mut Fp12) -> Fp12 {
        Fp12 {
            c0: self.c0.sub(&mut other.c0),
            c1: self.c1.sub(&mut other.c1),
            c2: self.c2.sub(&mut other.c2),
            c3: self.c3.sub(&mut other.c3),
            c4: self.c4.sub(&mut other.c4),
            c5: self.c5.sub(&mut other.c5),
        }
    }

    pub fn mul(&mut self, other: &mut Fp12, xi: &mut Fp2) -> Fp12 {
        // c0 = cs0co0 + xi(cs1co5 + cs2co4 + cs3co3 + cs4co2 + cs5co1)
        // c1 = cs0co1 + cs1co0 + xi(cs2co5 + cs3co4 + cs4co3 + cs5co2)
        // c2 = cs0co2 + cs1co1 + cs2co0 + xi(cs3co5 + cs4co4 + cs5co3)
        // c3 = cs0co3 + cs1co2 + cs2co1 + cs3co0 + xi(cs4co5 + cs5co4)
        // c4 = cs0co4 + cs1co3 + cs2co2 + cs3co1 + cs4co0 + xi(cs5co5)
        // c5 = cs0co5 + cs1co4 + cs2co3 + cs3co2 + cs4co1 + cs5co0
        //   where cs*: self.c*, co*: other.c*
        let mut c0_xi = xi.mul(
            &mut (self
                .c1
                .mul(&mut other.c5)
                .add(&mut self.c1.mul(&mut other.c5))
                .add(&mut self.c2.mul(&mut other.c4))
                .add(&mut self.c3.mul(&mut other.c3))
                .add(&mut self.c4.mul(&mut other.c1))
                .add(&mut self.c5.mul(&mut other.c0))),
        );
        let c0 = self.c0.mul(&mut other.c0).add(&mut c0_xi);

        let mut c1_xi = xi.mul(
            &mut (self
                .c2
                .mul(&mut other.c5)
                .add(&mut self.c2.mul(&mut other.c5))
                .add(&mut self.c3.mul(&mut other.c4))
                .add(&mut self.c4.mul(&mut other.c3))
                .add(&mut self.c5.mul(&mut other.c2))),
        );
        let c1 = self
            .c0
            .mul(&mut other.c1)
            .add(&mut self.c1.mul(&mut other.c0))
            .add(&mut c1_xi);

        let mut c2_xi = xi.mul(
            &mut (self
                .c3
                .mul(&mut other.c5)
                .add(&mut self.c3.mul(&mut other.c5))
                .add(&mut self.c4.mul(&mut other.c4))
                .add(&mut self.c5.mul(&mut other.c3))),
        );
        let c2 = self
            .c0
            .mul(&mut other.c2)
            .add(&mut self.c1.mul(&mut other.c1))
            .add(&mut self.c2.mul(&mut other.c0))
            .add(&mut c2_xi);

        let mut c3_xi = xi.mul(
            &mut (self
                .c4
                .mul(&mut other.c5)
                .add(&mut self.c5.mul(&mut other.c4))),
        );
        let c3 = self
            .c0
            .mul(&mut other.c3)
            .add(&mut self.c1.mul(&mut other.c2))
            .add(&mut self.c2.mul(&mut other.c1))
            .add(&mut self.c3.mul(&mut other.c0))
            .add(&mut c3_xi);

        let mut c4_xi = xi.mul(&mut (self.c5.mul(&mut other.c5)));
        let c4 = self
            .c0
            .mul(&mut other.c4)
            .add(&mut self.c1.mul(&mut other.c3))
            .add(&mut self.c2.mul(&mut other.c2))
            .add(&mut self.c3.mul(&mut other.c1))
            .add(&mut self.c4.mul(&mut other.c0))
            .add(&mut c4_xi);

        let c5 = self
            .c0
            .mul(&mut other.c5)
            .add(&mut self.c1.mul(&mut other.c4))
            .add(&mut self.c2.mul(&mut other.c3))
            .add(&mut self.c3.mul(&mut other.c2))
            .add(&mut self.c4.mul(&mut other.c1))
            .add(&mut self.c5.mul(&mut other.c0));

        Fp12 {
            c0: self.c0.clone(),
            c1: self.c1.clone(),
            c2: self.c2.clone(),
            c3: self.c3.clone(),
            c4: self.c4.clone(),
            c5: self.c5.clone(),
        }
    }

    pub fn div(&mut self, _other: &mut Fp12, _xi: &mut Fp2) -> Fp12 {
        todo!()
    }

    pub fn scalar_mul(&mut self, fp: &mut FieldVariable) -> Fp12 {
        Fp12 {
            c0: self.c0.scalar_mul(fp),
            c1: self.c1.scalar_mul(fp),
            c2: self.c2.scalar_mul(fp),
            c3: self.c3.scalar_mul(fp),
            c4: self.c4.scalar_mul(fp),
            c5: self.c5.scalar_mul(fp),
        }
    }
}

#[cfg(test)]
mod tests {
    use afs_primitives::sub_chip::LocalTraceInstructions;
    use ax_sdk::{
        any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
        utils::create_seeded_rng,
    };
    use halo2curves_axiom::{
        bn256::{Fq, Fq12, Fq2},
        ff::Field,
    };
    use num_bigint_dig::BigUint;
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;

    use super::{
        super::super::{field_expression::*, test_utils::*},
        *,
    };

    fn bn254_xi() -> Fq2 {
        Fq2::new(Fq::from_raw([9, 0, 0, 0]), Fq::one())
    }

    fn generate_random_fp12() -> Fq12 {
        let mut rng = create_seeded_rng();
        Fq12::random(&mut rng)
    }

    fn run_fp12_test_mul(
        x: Fq12,
        y: Fq12,
        fp12_fn: impl Fn(&mut Fp12, &mut Fp12, &mut Fp2) -> Fp12,
        fq12_fn: impl Fn(&Fq12, &Fq12) -> Fq12,
        save_result: bool,
    ) {
        let prime = bn254_prime();
        let (subair, range_checker, builder) = setup(&prime);

        let mut x_fp12 = Fp12::new(builder.clone());
        let mut y_fp12 = Fp12::new(builder.clone());
        let mut xi_fp2 = Fp2::new(builder.clone());
        let mut r = fp12_fn(&mut x_fp12, &mut y_fp12, &mut xi_fp2);
        if save_result {
            r.save();
        }

        let builder = builder.borrow().clone();
        let air = FieldExpr {
            builder,
            check_carry_mod_to_zero: subair,
            range_bus: range_checker.bus(),
        };
        let width = BaseAir::<BabyBear>::width(&air);

        let x_fq12 = x;
        let y_fq12 = y;
        let xi = bn254_xi();
        let r_fq12 = fq12_fn(&x_fq12, &y_fq12);
        let mut inputs = fq12_to_biguint_vec(&x_fq12);
        inputs.extend(fq12_to_biguint_vec(&y_fq12));
        inputs.extend(fq2_to_biguint_vec(&xi));

        let row = air.generate_trace_row((inputs, range_checker.clone(), vec![]));
        let FieldExprCols { vars, .. } = air.load_vars(&row);
        let trace = RowMajorMatrix::new(row, width);
        let range_trace = range_checker.generate_trace();

        assert_eq!(vars.len(), 12);
        let r_c0 = evaluate_biguint(&vars[0], LIMB_BITS);
        let r_c1 = evaluate_biguint(&vars[1], LIMB_BITS);
        let r_c2 = evaluate_biguint(&vars[2], LIMB_BITS);
        let r_c3 = evaluate_biguint(&vars[3], LIMB_BITS);
        let r_c4 = evaluate_biguint(&vars[4], LIMB_BITS);
        let r_c5 = evaluate_biguint(&vars[5], LIMB_BITS);
        let r_c6 = evaluate_biguint(&vars[6], LIMB_BITS);
        let r_c7 = evaluate_biguint(&vars[7], LIMB_BITS);
        let r_c8 = evaluate_biguint(&vars[8], LIMB_BITS);
        let r_c9 = evaluate_biguint(&vars[9], LIMB_BITS);
        let r_c10 = evaluate_biguint(&vars[10], LIMB_BITS);
        let r_c11 = evaluate_biguint(&vars[11], LIMB_BITS);
        let exp_r_c0_c0_c0 = bn254_fq_to_biguint(&r_fq12.c0.c0.c0);
        let exp_r_c0_c0_c1 = bn254_fq_to_biguint(&r_fq12.c0.c0.c1);
        let exp_r_c0_c1_c0 = bn254_fq_to_biguint(&r_fq12.c0.c1.c0);
        let exp_r_c0_c1_c1 = bn254_fq_to_biguint(&r_fq12.c0.c1.c1);
        let exp_r_c0_c2_c0 = bn254_fq_to_biguint(&r_fq12.c0.c2.c0);
        let exp_r_c0_c2_c1 = bn254_fq_to_biguint(&r_fq12.c0.c2.c1);
        let exp_r_c1_c0_c0 = bn254_fq_to_biguint(&r_fq12.c1.c0.c0);
        let exp_r_c1_c0_c1 = bn254_fq_to_biguint(&r_fq12.c1.c0.c1);
        let exp_r_c1_c1_c0 = bn254_fq_to_biguint(&r_fq12.c1.c1.c0);
        let exp_r_c1_c1_c1 = bn254_fq_to_biguint(&r_fq12.c1.c1.c1);
        let exp_r_c1_c2_c0 = bn254_fq_to_biguint(&r_fq12.c1.c2.c0);
        let exp_r_c1_c2_c1 = bn254_fq_to_biguint(&r_fq12.c1.c2.c1);
        // assert_eq!(r_c0, exp_r_c0_c0_c0);
        assert_eq!(r_c1, exp_r_c0_c0_c1);
        assert_eq!(r_c2, exp_r_c0_c1_c0);
        assert_eq!(r_c3, exp_r_c0_c1_c1);
        assert_eq!(r_c4, exp_r_c0_c2_c0);
        assert_eq!(r_c5, exp_r_c0_c2_c1);
        assert_eq!(r_c6, exp_r_c1_c0_c0);
        assert_eq!(r_c7, exp_r_c1_c0_c1);
        assert_eq!(r_c8, exp_r_c1_c1_c0);
        assert_eq!(r_c9, exp_r_c1_c1_c1);
        assert_eq!(r_c10, exp_r_c1_c2_c0);
        assert_eq!(r_c11, exp_r_c1_c2_c1);

        BabyBearBlake3Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![air, range_checker.air],
            vec![trace, range_trace],
        )
        .expect("Verification failed");
    }

    // #[test]
    // fn test_fp12_add() {
    //     let x = generate_random_fp12();
    //     let y = generate_random_fp12();
    //     run_fp12_test_add(x, y, Fp12::add, |x, y| x + y, true);
    // }

    // #[test]
    // fn test_fp12_sub() {
    //     let x = generate_random_fp12();
    //     let y = generate_random_fp12();
    //     run_fp12_test_add(x, y, Fp12::sub, |x, y| x - y, true);
    // }

    #[test]
    fn test_fp12_mul() {
        // let x = generate_random_fp12();
        // let y = generate_random_fp12();
        let one = Fq12::one();
        let two = one + one;
        let three = one + two;
        let x = two;
        let y = three;
        run_fp12_test_mul(x, y, Fp12::mul, |x, y| x * y, true);
    }

    // #[test]
    // fn test_fp12_div() {
    // let x = generate_random_fp12();
    // let y = generate_random_fp12();
    // let xi = bn254_xi();
    //     test_fp12(x, y, Fp2::div, |x, y| x * y.invert().unwrap(), false);
    // }
}
