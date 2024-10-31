use std::{cell::RefCell, rc::Rc};

use super::Fp2;
use crate::field_expression::{ExprBuilder, FieldVariable};

/// Field extension Fp12 defined with coefficients in Fp2. Fp6-equivalent coefficients are c0: (c0, c2, c4), c1: (c1, c3, c5).
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

    pub fn save(&mut self) -> [usize; 12] {
        let c0_indices = self.c0.save();
        let c1_indices = self.c1.save();
        let c2_indices = self.c2.save();
        let c3_indices = self.c3.save();
        let c4_indices = self.c4.save();
        let c5_indices = self.c5.save();

        [
            c0_indices[0],
            c0_indices[1],
            c1_indices[0],
            c1_indices[1],
            c2_indices[0],
            c2_indices[1],
            c3_indices[0],
            c3_indices[1],
            c4_indices[0],
            c4_indices[1],
            c5_indices[0],
            c5_indices[1],
        ]
    }

    pub fn save_output(&mut self) {
        self.c0.save_output();
        self.c1.save_output();
        self.c2.save_output();
        self.c3.save_output();
        self.c4.save_output();
        self.c5.save_output();
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

    pub fn mul(&mut self, other: &mut Fp12, xi: [isize; 2]) -> Fp12 {
        // c0 = cs0co0 + xi(cs1co2 + cs2co1 + cs3co5 + cs4co4 + cs5co5)
        // c1 = cs0co1 + cs1co0 + cs3co0 + xi(cs2co2 + cs4co5 + cs5co4)
        // c2 = cs0co2 + cs1co1 + cs2co0 + cs3co4 +cs4co3 + xi(cs5co5)
        // c3 = cs0co3 + cs3co0 + xi(cs1co5 + cs2co4 + cs4co2 + cs5co1)
        // c4 = cs0co4 + cs1co3 + cs3co1 + cs4co0 + xi(cs2co5 + cs5co2)
        // c5 = cs0co5 + cs1co4 + cs2co3 + cs3co2 + cs4co1 + cs5co0
        //   where cs*: self.c*, co*: other.c*

        let mut c0 = self.mul_c0(other, xi);
        let mut c1 = self.mul_c1(other, xi);
        let mut c2 = self.mul_c2(other, xi);
        let mut c3 = self.mul_c3(other, xi);
        let mut c4 = self.mul_c4(other, xi);
        let mut c5 = self.mul_c5(other);

        c0.save();
        c1.save();
        c2.save();
        c3.save();
        c4.save();
        c5.save();

        Fp12 {
            c0,
            c1,
            c2,
            c3,
            c4,
            c5,
        }
    }

    pub fn div(&mut self, _other: &mut Fp12, _xi: [isize; 2]) -> Fp12 {
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

    fn mul_c0(&mut self, other: &mut Fp12, xi: [isize; 2]) -> Fp2 {
        // c0 = cs0co0 + xi(cs1co2 + cs2co1 + cs3co5 + cs4co4 + cs5co3)
        let mut main_sum = self.c0.mul(&mut other.c0);
        let mut xi_sum = self
            .c1
            .mul(&mut other.c2)
            .add(&mut self.c2.mul(&mut other.c1))
            .add(&mut self.c3.mul(&mut other.c5))
            .add(&mut self.c4.mul(&mut other.c4))
            .add(&mut self.c5.mul(&mut other.c3));
        main_sum.add(&mut xi_sum.int_mul(xi))
    }

    fn mul_c1(&mut self, other: &mut Fp12, xi: [isize; 2]) -> Fp2 {
        // c1 = cs0co1 + cs1co0 + cs3co3 + xi(cs2co2 + cs4co5 + cs5co4)
        let mut main_sum = self
            .c0
            .mul(&mut other.c1)
            .add(&mut self.c1.mul(&mut other.c0))
            .add(&mut self.c3.mul(&mut other.c3));
        let mut xi_sum = self
            .c2
            .mul(&mut other.c2)
            .add(&mut self.c4.mul(&mut other.c5))
            .add(&mut self.c5.mul(&mut other.c4));
        main_sum.add(&mut xi_sum.int_mul(xi))
    }

    fn mul_c2(&mut self, other: &mut Fp12, xi: [isize; 2]) -> Fp2 {
        // c2 = cs0co2 + cs1co1 + cs2co0 + cs3co4 +cs4co3 + xi(cs5co5)
        let mut main_sum = self
            .c0
            .mul(&mut other.c2)
            .add(&mut self.c1.mul(&mut other.c1))
            .add(&mut self.c2.mul(&mut other.c0))
            .add(&mut self.c3.mul(&mut other.c4))
            .add(&mut self.c4.mul(&mut other.c3));
        let mut xi_sum = self.c5.mul(&mut other.c5);
        main_sum.add(&mut xi_sum.int_mul(xi))
    }

    fn mul_c3(&mut self, other: &mut Fp12, xi: [isize; 2]) -> Fp2 {
        // c3 = cs0co3 + cs3co0 + xi(cs1co5 + cs2co4 + cs4co2 + cs5co1)
        let mut main_sum = self
            .c0
            .mul(&mut other.c3)
            .add(&mut self.c3.mul(&mut other.c0));
        let mut xi_sum = self
            .c1
            .mul(&mut other.c5)
            .add(&mut self.c2.mul(&mut other.c4))
            .add(&mut self.c4.mul(&mut other.c2))
            .add(&mut self.c5.mul(&mut other.c1));
        main_sum.add(&mut xi_sum.int_mul(xi))
    }

    fn mul_c4(&mut self, other: &mut Fp12, xi: [isize; 2]) -> Fp2 {
        // c4 = cs0co4 + cs1co3 + cs3co1 + cs4co0 + xi(cs2co5 + cs5co2)
        let mut main_sum = self
            .c0
            .mul(&mut other.c4)
            .add(&mut self.c1.mul(&mut other.c3))
            .add(&mut self.c3.mul(&mut other.c1))
            .add(&mut self.c4.mul(&mut other.c0));
        let mut xi_sum = self
            .c2
            .mul(&mut other.c5)
            .add(&mut self.c5.mul(&mut other.c2));
        main_sum.add(&mut xi_sum.int_mul(xi))
    }

    fn mul_c5(&mut self, other: &mut Fp12) -> Fp2 {
        // c5 = cs0co5 + cs1co4 + cs2co3 + cs3co2 + cs4co1 + cs5co0
        self.c0
            .mul(&mut other.c5)
            .add(&mut self.c1.mul(&mut other.c4))
            .add(&mut self.c2.mul(&mut other.c3))
            .add(&mut self.c3.mul(&mut other.c2))
            .add(&mut self.c4.mul(&mut other.c1))
            .add(&mut self.c5.mul(&mut other.c0))
    }
}

#[cfg(test)]
mod tests {
    use ax_circuit_primitives::TraceSubRowGenerator;
    use ax_stark_sdk::{
        any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
        utils::create_seeded_rng,
    };
    use axvm_ecc_constants::BN254;
    use halo2curves_axiom::{bn256::Fq12, ff::Field};
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;

    use super::{
        super::super::{field_expression::*, test_utils::*},
        *,
    };

    fn generate_random_fq12() -> Fq12 {
        let mut rng = create_seeded_rng();
        Fq12::random(&mut rng)
    }

    fn run_fp12_test(
        x: Fq12,
        y: Fq12,
        xi: Option<[isize; 2]>,
        fp12_fn_addsub: Option<impl Fn(&mut Fp12, &mut Fp12) -> Fp12>,
        fp12_fn_mul: Option<impl Fn(&mut Fp12, &mut Fp12, [isize; 2]) -> Fp12>,
        fq12_fn: impl Fn(&Fq12, &Fq12) -> Fq12,
    ) {
        if fp12_fn_addsub.is_none() && fp12_fn_mul.is_none() {
            panic!("Either fp12_fn_addsub or fp12_fn_mul must be provided");
        }
        if fp12_fn_addsub.is_some() && fp12_fn_mul.is_some() {
            panic!("Only one of fp12_fn_addsub or fp12_fn_mul must be provided");
        }

        let prime = BN254.MODULUS.clone();
        let (range_checker, builder) = setup(&prime);

        let mut x_fp12 = Fp12::new(builder.clone());
        let mut y_fp12 = Fp12::new(builder.clone());
        let mut r = if let Some(fp12_fn_addsub) = fp12_fn_addsub {
            fp12_fn_addsub(&mut x_fp12, &mut y_fp12)
        } else {
            let fp12_fn_mul = fp12_fn_mul.unwrap();
            fp12_fn_mul(&mut x_fp12, &mut y_fp12, xi.unwrap())
        };
        let indices = r.save();

        let builder = builder.borrow().clone();
        let air = FieldExpr::new(builder, range_checker.bus());
        let width = BaseAir::<BabyBear>::width(&air);

        let x_fq12 = x;
        let y_fq12 = y;
        let r_fq12 = fq12_fn(&x_fq12, &y_fq12);
        let mut inputs = bn254_fq12_to_biguint_vec(&x_fq12);
        inputs.extend(bn254_fq12_to_biguint_vec(&y_fq12));

        let mut row = vec![BabyBear::zero(); width];
        air.generate_subrow((&range_checker, inputs, vec![]), &mut row);
        let FieldExprCols { vars, .. } = air.load_vars(&row);
        let trace = RowMajorMatrix::new(row, width);
        let range_trace = range_checker.generate_trace();

        let r_c0 = evaluate_biguint(&vars[indices[0]], LIMB_BITS);
        let r_c1 = evaluate_biguint(&vars[indices[1]], LIMB_BITS);
        let r_c2 = evaluate_biguint(&vars[indices[2]], LIMB_BITS);
        let r_c3 = evaluate_biguint(&vars[indices[3]], LIMB_BITS);
        let r_c4 = evaluate_biguint(&vars[indices[4]], LIMB_BITS);
        let r_c5 = evaluate_biguint(&vars[indices[5]], LIMB_BITS);
        let r_c6 = evaluate_biguint(&vars[indices[6]], LIMB_BITS);
        let r_c7 = evaluate_biguint(&vars[indices[7]], LIMB_BITS);
        let r_c8 = evaluate_biguint(&vars[indices[8]], LIMB_BITS);
        let r_c9 = evaluate_biguint(&vars[indices[9]], LIMB_BITS);
        let r_c10 = evaluate_biguint(&vars[indices[10]], LIMB_BITS);
        let r_c11 = evaluate_biguint(&vars[indices[11]], LIMB_BITS);
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

        assert_eq!(r_c0, exp_r_c0_c0_c0);
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

    #[test]
    fn test_fp12_add() {
        let x = generate_random_fq12();
        let y = generate_random_fq12();
        run_fp12_test(
            x,
            y,
            None,
            Some(Fp12::add),
            None::<fn(&mut Fp12, &mut Fp12, [isize; 2]) -> Fp12>,
            |x, y| x + y,
        );
    }

    #[test]
    fn test_fp12_sub() {
        let x = generate_random_fq12();
        let y = generate_random_fq12();
        run_fp12_test(
            x,
            y,
            None,
            Some(Fp12::sub),
            None::<fn(&mut Fp12, &mut Fp12, [isize; 2]) -> Fp12>,
            |x, y| x - y,
        );
    }

    #[test]
    fn test_fp12_mul() {
        let x = generate_random_fq12();
        let y = generate_random_fq12();
        let xi = [9, 1];
        run_fp12_test(
            x,
            y,
            Some(xi),
            None::<fn(&mut Fp12, &mut Fp12) -> Fp12>,
            Some(Fp12::mul),
            |x, y| x * y,
        );
    }
}
