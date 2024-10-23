use std::{cell::RefCell, rc::Rc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::field_expression::{ExprBuilder, FieldExpr};
use num_bigint_dig::BigUint;

use super::super::FIELD_ELEMENT_BITS;

pub fn ec_double_expr(
    modulus: BigUint, // The coordinate field.
    num_limbs: usize,
    limb_bits: usize,
    range_bus: VariableRangeCheckerBus,
) -> FieldExpr {
    assert!(modulus.bits() <= num_limbs * limb_bits);
    let subair = CheckCarryModToZeroSubAir::new(
        modulus.clone(),
        limb_bits,
        range_bus.index,
        range_bus.range_max_bits,
        FIELD_ELEMENT_BITS,
    );
    let builder = ExprBuilder::new(modulus, limb_bits, num_limbs, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let mut lambda = x1.square().int_mul(3) / (y1.int_mul(2));
    let mut x3 = lambda.square() - x1.int_mul(2);
    x3.save_output();
    let mut y3 = lambda * (x1 - x3.clone()) - y1;
    y3.save_output();

    let builder = builder.borrow().clone();
    FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus,
    }
}
