use std::{cell::RefCell, rc::Rc};

use crate::field_expression::{ExprBuilder, FieldVariable};

/// Quadratic field extension of `Fp` defined by `Fp2 = Fp[u]/(1 + u^2)`. Assumes that `-1` is not a quadratic residue in `Fp`, which is equivalent to `p` being congruent to `3 (mod 4)`.
#[derive(Clone)]
pub struct Fp2 {
    pub c0: FieldVariable,
    pub c1: FieldVariable,
}

impl Fp2 {
    pub fn new(builder: Rc<RefCell<ExprBuilder>>) -> Self {
        let c0 = ExprBuilder::new_input(builder.clone());
        let c1 = ExprBuilder::new_input(builder.clone());
        Fp2 { c0, c1 }
    }

    pub fn save(&mut self) -> [usize; 2] {
        let c0_idx = self.c0.save();
        let c1_idx = self.c1.save();
        [c0_idx, c1_idx]
    }

    pub fn save_output(&mut self) {
        self.c0.save_output();
        self.c1.save_output();
    }

    pub fn add(&mut self, other: &mut Fp2) -> Fp2 {
        Fp2 {
            c0: &mut self.c0 + &mut other.c0,
            c1: &mut self.c1 + &mut other.c1,
        }
    }

    pub fn sub(&mut self, other: &mut Fp2) -> Fp2 {
        Fp2 {
            c0: &mut self.c0 - &mut other.c0,
            c1: &mut self.c1 - &mut other.c1,
        }
    }

    pub fn mul(&mut self, other: &mut Fp2) -> Fp2 {
        let c0 = &mut self.c0 * &mut other.c0 - &mut self.c1 * &mut other.c1;
        let c1 = &mut self.c0 * &mut other.c1 + &mut self.c1 * &mut other.c0;
        Fp2 { c0, c1 }
    }

    pub fn div(&mut self, other: &mut Fp2) -> Fp2 {
        println!("div fp2!!!!\n\n");
        let mut builder = self.c0.builder.borrow_mut();
        let z0 = builder.new_var();
        let z1 = builder.new_var();
        let prime = builder.prime.clone();
        let limb_bits = builder.limb_bits;
        let num_limbs = builder.num_limbs;
        drop(builder);

        // Compute should not be affected by whether auto save is triggered.
        // So we must do compute first.
        // Compute z0
        let compute_denom = &other.c0.expr * &other.c0.expr + &other.c1.expr * &other.c1.expr;
        let compute_z0_nom = &self.c0.expr * &other.c0.expr + &self.c1.expr * &other.c1.expr;
        let compute_z0 = &compute_z0_nom / &compute_denom;
        // Compute z1
        let compute_z1_nom = &self.c1.expr * &other.c0.expr - &self.c0.expr * &other.c1.expr;
        let compute_z1 = &compute_z1_nom / &compute_denom;

        // Constraint 1: x0 = y0*z0 - y1*z1
        let constraint1 = &self.c0.expr - &other.c0.expr * &z0 + &other.c1.expr * &z1;
        let carry_bits = constraint1.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs);
        if carry_bits > self.c0.range_checker_bits {
            // TODO: should save the "bigger" one first (the one with higher limb_max_abs)
            self.save();
        }
        let constraint1 = &self.c0.expr - &other.c0.expr * &z0 + &other.c1.expr * &z1;
        let carry_bits = constraint1.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs);
        if carry_bits > self.c0.range_checker_bits {
            other.save();
        }
        let constraint1 = &self.c0.expr - &other.c0.expr * &z0 + &other.c1.expr * &z1;

        // Constraint 2: x1 = y1*z0 + y0*z1
        let constraint2 = &self.c1.expr - &other.c1.expr * &z0 - &other.c0.expr * &z1;
        let carry_bits = constraint2.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs);
        if carry_bits > self.c0.range_checker_bits {
            // TODO: should save the "bigger" one first (the one with higher limb_max_abs)
            self.save();
        }
        let constraint2 = &self.c1.expr - &other.c1.expr * &z0 - &other.c0.expr * &z1;
        let carry_bits = constraint2.constraint_carry_bits_with_pq(&prime, limb_bits, num_limbs);
        if carry_bits > self.c0.range_checker_bits {
            other.save();
        }
        let constraint2 = &self.c1.expr - &other.c1.expr * &z0 - &other.c0.expr * &z1;

        let mut builder = self.c0.builder.borrow_mut();
        builder.add_constraint(constraint1);
        builder.add_constraint(constraint2);
        builder.add_compute(compute_z0);
        builder.add_compute(compute_z1);
        drop(builder);

        let z0_var = FieldVariable::from_var(self.c0.builder.clone(), z0);
        let z1_var = FieldVariable::from_var(self.c0.builder.clone(), z1);
        Fp2 {
            c0: z0_var,
            c1: z1_var,
        }
    }

    pub fn scalar_mul(&mut self, fp: &mut FieldVariable) -> Fp2 {
        Fp2 {
            c0: &mut self.c0 * fp,
            c1: &mut self.c1 * fp,
        }
    }

    // c is like a Fp2, but with both c0 and c1 being very small numbers.
    pub fn int_mul(&mut self, c: [isize; 2]) -> Fp2 {
        let c0 = self.c0.int_mul(c[0]) - self.c1.int_mul(c[1]);
        let c1 = self.c0.int_mul(c[1]) + self.c1.int_mul(c[0]);
        Fp2 { c0, c1 }
    }
}

#[cfg(test)]
mod tests {
    use ax_circuit_primitives::TraceSubRowGenerator;
    use ax_sdk::{
        any_rap_arc_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
        utils::create_seeded_rng,
    };
    use halo2curves_axiom::{bn256::Fq2, ff::Field};
    use num_bigint_dig::BigUint;
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;

    use super::{
        super::super::{field_expression::*, test_utils::*},
        *,
    };

    fn generate_random_fq2() -> Fq2 {
        let mut rng = create_seeded_rng();
        Fq2::random(&mut rng)
    }

    fn two_fp2_input(x: &Fq2, y: &Fq2) -> Vec<BigUint> {
        vec![
            bn254_fq_to_biguint(&x.c0),
            bn254_fq_to_biguint(&x.c1),
            bn254_fq_to_biguint(&y.c0),
            bn254_fq_to_biguint(&y.c1),
        ]
    }

    fn test_fp2(
        fp2_fn: impl Fn(&mut Fp2, &mut Fp2) -> Fp2,
        fq2_fn: impl Fn(&Fq2, &Fq2) -> Fq2,
        save_result: bool,
    ) {
        let prime = bn254_prime();
        let (subair, range_checker, builder) = setup(&prime);

        let mut x_fp2 = Fp2::new(builder.clone());
        let mut y_fp2 = Fp2::new(builder.clone());
        let mut r = fp2_fn(&mut x_fp2, &mut y_fp2);
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

        let x_fp2 = generate_random_fq2();
        let y_fp2 = generate_random_fq2();
        let r_fp2 = fq2_fn(&x_fp2, &y_fp2);
        let inputs = two_fp2_input(&x_fp2, &y_fp2);

        let mut row = vec![BabyBear::zero(); width];
        air.generate_subrow((&range_checker, inputs, vec![]), &mut row);
        let FieldExprCols { vars, .. } = air.load_vars(&row);
        let trace = RowMajorMatrix::new(row, width);
        let range_trace = range_checker.generate_trace();
        assert_eq!(vars.len(), 2);
        let r_c0 = evaluate_biguint(&vars[0], LIMB_BITS);
        let r_c1 = evaluate_biguint(&vars[1], LIMB_BITS);
        let expected_c0 = bn254_fq_to_biguint(&r_fp2.c0);
        let expected_c1 = bn254_fq_to_biguint(&r_fp2.c1);
        assert_eq!(r_c0, expected_c0);
        assert_eq!(r_c1, expected_c1);

        BabyBearBlake3Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![air, range_checker.air],
            vec![trace, range_trace],
        )
        .expect("Verification failed");
    }

    #[test]
    fn test_fp2_add() {
        test_fp2(Fp2::add, |x, y| x + y, true);
    }

    #[test]
    fn test_fp2_sub() {
        test_fp2(Fp2::sub, |x, y| x - y, true);
    }

    #[test]
    fn test_fp2_mul() {
        test_fp2(Fp2::mul, |x, y| x * y, true);
    }

    #[test]
    fn test_fp2_div() {
        test_fp2(Fp2::div, |x, y| x * y.invert().unwrap(), false);
    }

    #[test]
    fn test_fp2_div2() {
        let prime = bn254_prime();
        let (subair, range_checker, builder) = setup(&prime);

        let mut x_fp2 = Fp2::new(builder.clone());
        let mut y_fp2 = Fp2::new(builder.clone());
        let mut z_fp2 = Fp2::new(builder.clone());
        let mut xy = x_fp2.mul(&mut y_fp2);
        let _r = xy.div(&mut z_fp2);
        // no need to save as div auto save.

        let builder = builder.borrow().clone();
        let air = FieldExpr {
            builder: builder.clone(),
            check_carry_mod_to_zero: subair,
            range_bus: range_checker.bus(),
        };
        let width = BaseAir::<BabyBear>::width(&air);

        let x_fp2 = generate_random_fq2();
        let y_fp2 = generate_random_fq2();
        let z_fp2 = generate_random_fq2();
        let r_fp2 = z_fp2.invert().unwrap() * x_fp2 * y_fp2;
        let inputs = vec![
            bn254_fq_to_biguint(&x_fp2.c0),
            bn254_fq_to_biguint(&x_fp2.c1),
            bn254_fq_to_biguint(&y_fp2.c0),
            bn254_fq_to_biguint(&y_fp2.c1),
            bn254_fq_to_biguint(&z_fp2.c0),
            bn254_fq_to_biguint(&z_fp2.c1),
        ];
        let mut row = vec![BabyBear::zero(); width];
        air.generate_subrow((&range_checker, inputs, vec![]), &mut row);
        let FieldExprCols { vars, .. } = air.load_vars(&row);
        let trace = RowMajorMatrix::new(row, width);
        let range_trace = range_checker.generate_trace();
        assert_eq!(vars.len(), 2);
        let r_c0 = evaluate_biguint(&vars[0], LIMB_BITS);
        let r_c1 = evaluate_biguint(&vars[1], LIMB_BITS);
        let expected_c0 = bn254_fq_to_biguint(&r_fp2.c0);
        let expected_c1 = bn254_fq_to_biguint(&r_fp2.c1);
        assert_eq!(r_c0, expected_c0);
        assert_eq!(r_c1, expected_c1);

        BabyBearBlake3Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![air, range_checker.air],
            vec![trace, range_trace],
        )
        .expect("Verification failed");
    }
}
