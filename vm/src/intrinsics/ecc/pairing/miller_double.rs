use std::{cell::RefCell, rc::Rc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, FieldExpr},
    field_extension::Fp2,
};
use num_bigint_dig::BigUint;

use super::super::FIELD_ELEMENT_BITS;

pub fn miller_double_expr(
    modulus: BigUint,
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

    let mut s_x = Fp2::new(builder.clone());
    let mut s_y = Fp2::new(builder.clone());

    let mut three_x_square = s_x.square().int_mul([3, 3]);
    let mut lambda = three_x_square.div(&mut s_y.int_mul([2, 2]));
    let mut x_2s = lambda.square().sub(&mut s_x.int_mul([2, 2]));
    let mut y_2s = lambda.mul(&mut (s_x.sub(&mut x_2s))).sub(&mut s_y);
    x_2s.save_output();
    y_2s.save_output();

    let mut b = lambda.int_mul([-1, -1]);
    let mut c = lambda.mul(&mut s_x).sub(&mut s_y);
    b.save_output();
    c.save_output();

    let builder = builder.borrow().clone();
    FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus,
    }
}
