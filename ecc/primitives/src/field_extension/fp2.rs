use std::{cell::RefCell, rc::Rc};

use crate::field_expression::{ExprBuilder, FieldVariable, FieldVariableConfig};

pub struct Fp2<C: FieldVariableConfig> {
    pub c0: FieldVariable<C>,
    pub c1: FieldVariable<C>,
}

impl<C: FieldVariableConfig> Fp2<C> {
    pub fn new(builder: Rc<RefCell<ExprBuilder>>) -> Self {
        let c0 = ExprBuilder::new_input::<C>(builder.clone());
        let c1 = ExprBuilder::new_input::<C>(builder.clone());
        Fp2 { c0, c1 }
    }

    pub fn save(&mut self) {
        self.c0.save();
        self.c1.save();
    }

    pub fn add(&mut self, other: &mut Fp2<C>) -> Fp2<C> {
        Fp2 {
            c0: self.c0.add(&mut other.c0),
            c1: self.c1.add(&mut other.c1),
        }
    }

    pub fn sub(&mut self, other: &mut Fp2<C>) -> Fp2<C> {
        Fp2 {
            c0: self.c0.sub(&mut other.c0),
            c1: self.c1.sub(&mut other.c1),
        }
    }

    pub fn mul(&mut self, other: &mut Fp2<C>) -> Fp2<C> {
        let c0 = &mut self.c0 * &mut other.c0 - &mut self.c1 * &mut other.c1;
        let c1 = &mut self.c0 * &mut other.c1 + &mut self.c1 * &mut other.c0;
        Fp2 { c0, c1 }
    }

    pub fn div(&mut self, other: &mut Fp2<C>) -> Fp2<C> {
        let mut denom = other.c0.clone() * other.c0.clone() + other.c1.clone() * other.c1.clone();
        denom.save();
        let c0 = (&mut self.c0 * &mut other.c0 + &mut self.c1 * &mut other.c1) / &denom;
        let c1 = (&mut self.c1 * &mut other.c0 - &mut self.c0 * &mut other.c1) / &denom;
        Fp2 { c0, c1 }
    }

    pub fn scalar_mul(&mut self, fp: &mut FieldVariable<C>) -> Fp2<C> {
        Fp2 {
            c0: self.c0.mul(fp),
            c1: self.c1.mul(fp),
        }
    }
}

#[cfg(test)]
mod tests {
    use afs_primitives::sub_chip::LocalTraceInstructions;
    use ax_sdk::{
        any_rap_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
        utils::create_seeded_rng,
    };
    use elliptic_curve::Group;
    use halo2curves_axiom::bn256::{Fq, Fq2, G1};
    use num_bigint_dig::BigUint;
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;

    use super::{
        super::super::{field_expression::*, test_utils::*},
        *,
    };

    fn fq_to_biguint(fq: &Fq) -> BigUint {
        let bytes = fq.to_bytes();
        BigUint::from_bytes_le(&bytes)
    }

    fn generate_random_fp2() -> Fq2 {
        let mut rng = create_seeded_rng();
        let point_x = G1::random(&mut rng);
        Fq2 {
            c0: point_x.x,
            c1: point_x.y,
        }
    }

    fn two_fp2_input(x: &Fq2, y: &Fq2) -> Vec<BigUint> {
        vec![
            fq_to_biguint(&x.c0),
            fq_to_biguint(&x.c1),
            fq_to_biguint(&y.c0),
            fq_to_biguint(&y.c1),
        ]
    }

    fn test_fp2(
        fp2_fn: impl Fn(&mut Fp2<TestConfig>, &mut Fp2<TestConfig>) -> Fp2<TestConfig>,
        fq2_fn: impl Fn(&Fq2, &Fq2) -> Fq2,
    ) {
        let prime = bn254_prime();
        let (subair, range_checker, builder) = setup(&prime);

        let mut x_fp2 = Fp2::<TestConfig>::new(builder.clone());
        let mut y_fp2 = Fp2::<TestConfig>::new(builder.clone());
        let mut r = fp2_fn(&mut x_fp2, &mut y_fp2);
        r.save();

        let builder = builder.borrow().clone();
        let chip = FieldExprChip {
            builder,
            check_carry_mod_to_zero: subair,
            range_checker: range_checker.clone(),
        };

        let x_fp2 = generate_random_fp2();
        let y_fp2 = generate_random_fp2();
        let r_fp2 = fq2_fn(&x_fp2, &y_fp2);
        let inputs = two_fp2_input(&x_fp2, &y_fp2);

        let row = chip.generate_trace_row((inputs, range_checker.clone()));
        let (_, _, vars, _, _) = chip.load_vars(&row);
        let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&chip));
        let range_trace = range_checker.generate_trace();
        assert_eq!(vars.len(), 2);
        let r_c0 = evaluate_biguint(&vars[0], LIMB_BITS);
        let r_c1 = evaluate_biguint(&vars[1], LIMB_BITS);
        let expected_c0 = fq_to_biguint(&r_fp2.c0);
        let expected_c1 = fq_to_biguint(&r_fp2.c1);
        assert_eq!(r_c0, expected_c0);
        assert_eq!(r_c1, expected_c1);

        BabyBearBlake3Engine::run_simple_test_no_pis(
            &any_rap_vec![&chip, &range_checker.air],
            vec![trace, range_trace],
        )
        .expect("Verification failed");
    }

    #[test]
    fn test_fp2_add() {
        test_fp2(Fp2::add, |x, y| x + y);
    }

    #[test]
    fn test_fp2_sub() {
        test_fp2(Fp2::sub, |x, y| x - y);
    }

    #[test]
    fn test_fp2_mul() {
        test_fp2(Fp2::mul, |x, y| x * y);
    }

    #[test]
    fn test_fp2_div() {
        let prime = bn254_prime();
        let (subair, range_checker, builder) = setup(&prime);

        let mut x_fp2 = Fp2::<TestConfig>::new(builder.clone());
        let mut y_fp2 = Fp2::<TestConfig>::new(builder.clone());
        let _r = x_fp2.div(&mut y_fp2);
        // no need to save as div auto save.

        let builder = builder.borrow().clone();
        let chip = FieldExprChip {
            builder,
            check_carry_mod_to_zero: subair,
            range_checker: range_checker.clone(),
        };

        let x_fp2 = generate_random_fp2();
        let y_fp2 = generate_random_fp2();
        let r_fp2 = y_fp2.invert().unwrap() * x_fp2;
        let inputs = two_fp2_input(&x_fp2, &y_fp2);

        let row = chip.generate_trace_row((inputs, range_checker.clone()));
        let (_, _, vars, _, _) = chip.load_vars(&row);
        let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&chip));
        let range_trace = range_checker.generate_trace();
        assert_eq!(vars.len(), 3); // one more var for denom.
        let r_c0 = evaluate_biguint(&vars[1], LIMB_BITS);
        let r_c1 = evaluate_biguint(&vars[2], LIMB_BITS);
        let expected_c0 = fq_to_biguint(&r_fp2.c0);
        let expected_c1 = fq_to_biguint(&r_fp2.c1);
        assert_eq!(r_c0, expected_c0);
        assert_eq!(r_c1, expected_c1);

        BabyBearBlake3Engine::run_simple_test_no_pis(
            &any_rap_vec![&chip, &range_checker.air],
            vec![trace, range_trace],
        )
        .expect("Verification failed");
    }
}
