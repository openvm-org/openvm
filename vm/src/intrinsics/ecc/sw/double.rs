use std::{cell::RefCell, rc::Rc};

use ax_circuit_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr};

use super::super::FIELD_ELEMENT_BITS;

pub fn ec_double_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
) -> FieldExpr {
    assert!(config.modulus.bits() <= config.num_limbs * config.limb_bits);
    let subair = CheckCarryModToZeroSubAir::new(
        config.modulus.clone(),
        config.limb_bits,
        range_bus.index,
        range_bus.range_max_bits,
        FIELD_ELEMENT_BITS,
    );
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
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
