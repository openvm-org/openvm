use std::{cell::RefCell, rc::Rc};

use ax_circuit_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, ExprBuilderConfig, FieldExpr},
    field_extension::Fp12,
};

use crate::intrinsics::ecc::FIELD_ELEMENT_BITS;

pub fn fp12_sub_expr(config: ExprBuilderConfig, range_bus: VariableRangeCheckerBus) -> FieldExpr {
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

    let mut x = Fp12::new(builder.clone());
    let mut y = Fp12::new(builder.clone());
    let mut res = x.sub(&mut y);
    res.save_output();

    let builder = builder.borrow().clone();
    FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus,
    }
}
