use std::{cell::RefCell, rc::Rc};

use num_bigint::BigUint;
use num_traits::One;
use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_mod_circuit_builder::{ExprBuilder, ExprBuilderConfig, FieldExpr};

pub fn ec_add_expr(
    config: ExprBuilderConfig, // The coordinate field.
    range_bus: VariableRangeCheckerBus,
    a_biguint: BigUint,
    d_biguint: BigUint,
) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let a = ExprBuilder::new_const(builder.clone(), a_biguint.clone());
    let d = ExprBuilder::new_const(builder.clone(), d_biguint.clone());
    let one = ExprBuilder::new_const(builder.clone(), BigUint::one());

    let x1y2 = x1.clone() * y2.clone();
    let x2y1 = x2.clone() * y1.clone();
    let y1y2 = y1 * y2;
    let x1x2 = x1 * x2;
    let dx1x2y1y2 = d * x1x2.clone() * y1y2.clone();

    let mut x3 = (x1y2 + x2y1) / (one.clone() + dx1x2y1y2.clone());
    let mut y3 = (y1y2 - a * x1x2) / (one - dx1x2y1y2);

    x3.save_output();
    y3.save_output();

    let builder = builder.borrow().clone();

    FieldExpr::new_with_setup_values(builder, range_bus, true, vec![a_biguint, d_biguint])
}
