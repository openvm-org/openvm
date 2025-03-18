use std::{cell::RefCell, rc::Rc};

use openvm_circuit_primitives::var_range::VariableRangeCheckerBus;
use openvm_mod_circuit_builder::{ExprBuilder, ExprBuilderConfig, FieldExpr};

use crate::Fp12;

pub fn fp12_add_expr(config: ExprBuilderConfig, range_bus: VariableRangeCheckerBus) -> FieldExpr {
    config.check_valid();
    let builder = ExprBuilder::new(config, range_bus.range_max_bits);
    let builder = Rc::new(RefCell::new(builder));

    let mut x = Fp12::new(builder.clone());
    let mut y = Fp12::new(builder.clone());
    let mut res = x.add(&mut y);
    res.save_output();

    let builder = builder.borrow().clone();
    FieldExpr::new(builder, range_bus, false)
}
