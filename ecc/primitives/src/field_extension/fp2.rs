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
    use afs_primitives::{bigint::utils::*, sub_chip::LocalTraceInstructions};
    use ax_sdk::{
        any_rap_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
    };
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;

    use super::{
        super::super::{field_expression::*, test_utils::*},
        *,
    };

    #[test]
    fn test_fp2_add() {
        let prime = secp256k1_coord_prime();
        let (subair, range_checker, builder) = setup(&prime);

        let mut x_fp2 = Fp2::<TestConfig>::new(builder.clone());
        let mut y_fp2 = Fp2::<TestConfig>::new(builder.clone());
        let mut r = x_fp2.add(&mut y_fp2);
        r.save();

        let builder = builder.borrow().clone();
        let chip = FieldExprChip {
            builder,
            check_carry_mod_to_zero: subair,
            range_checker: range_checker.clone(),
        };

        let x_c0 = generate_random_biguint(&prime);
        let x_c1 = generate_random_biguint(&prime);
        let y_c0 = generate_random_biguint(&prime);
        let y_c1 = generate_random_biguint(&prime);
        let inputs = vec![x_c0, x_c1, y_c0, y_c1];
        let row = chip.generate_trace_row((inputs, range_checker.clone()));
        let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&chip));
        let range_trace = range_checker.generate_trace();

        BabyBearBlake3Engine::run_simple_test_no_pis(
            &any_rap_vec![&chip, &range_checker.air],
            vec![trace, range_trace],
        )
        .expect("Verification failed");
    }
}
