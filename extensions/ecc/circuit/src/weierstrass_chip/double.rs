use std::{cell::RefCell, rc::Rc};

use num_bigint_dig::BigUint;
use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_mod_circuit_builder::{ExprBuilder, ExprBuilderConfig, FieldExpr};

pub fn ec_double_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x1 = ExprBuilder::new_input(builder.clone());
    let mut y1 = ExprBuilder::new_input(builder.clone());
    let a = ExprBuilder::new_const(builder.clone(), a_biguint);
    let mut lambda = (x1.square().int_mul(3) + a) / (y1.int_mul(2));
    let mut x3 = lambda.square() - x1.int_mul(2);
    x3.save_output();
    let mut y3 = lambda * (x1 - x3.clone()) - y1;
    y3.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus, true)
}
