use std::{cell::RefCell, rc::Rc};

use ax_circuit_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir, var_range::VariableRangeCheckerBus,
};
use ax_ecc_primitives::{
    field_expression::{ExprBuilder, FieldExpr},
    field_extension::Fp2,
};
use num_bigint_dig::BigUint;

use crate::intrinsics::ecc::FIELD_ELEMENT_BITS;

pub fn mul_013_by_013_expr(
    modulus: BigUint,
    num_limbs: usize,
    limb_bits: usize,
    range_bus: VariableRangeCheckerBus,
    xi: [isize; 2],
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

    let mut b0 = Fp2::new(builder.clone()); // x1
    let mut c0 = Fp2::new(builder.clone()); // x3
    let mut b1 = Fp2::new(builder.clone()); // y1
    let mut c1 = Fp2::new(builder.clone()); // y3

    // where w⁶ = xi
    // l0 * l1 = 1 + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴ + (c0c1)w⁶
    //         = (1 + c0c1 * xi) + (b0 + b1)w + (b0b1)w² + (c0 + c1)w³ + (b0c1 + b1c0)w⁴
    let mut l0 = c0.mul(&mut c1); //.int_mul(xi); //.int_add([1, 0]);
    let mut l1 = b0.add(&mut b1);
    let mut l2 = b0.mul(&mut b1);
    let mut l3 = c0.add(&mut c1);
    let mut l4 = b0.mul(&mut c1).add(&mut b1.mul(&mut c0));

    // let [mut l0, mut l1, mut l2, mut l3, mut l4] =
    //     mul_013_by_013(&mut b0, &mut c0, &mut b1, &mut c1, xi);
    l0.save_output();
    l1.save_output();
    l2.save_output();
    l3.save_output();
    l4.save_output();

    let builder = builder.borrow().clone();
    FieldExpr {
        builder,
        check_carry_mod_to_zero: subair,
        range_bus,
    }
}
