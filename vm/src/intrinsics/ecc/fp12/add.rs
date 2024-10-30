use std::{cell::RefCell, rc::Rc};

use ax_circuit_primitives::{
    var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr},
    field_extension::Fp12,
};

pub fn fp12_add_expr(config: ExprBuilderConfig, range_bus: VariableRangeCheckerBus) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config.clone(), range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x = Fp12::new(builder.clone());
    let mut y = Fp12::new(builder.clone());
    let mut res = x.add(&mut y);
    res.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(config, builder, range_bus)
}
