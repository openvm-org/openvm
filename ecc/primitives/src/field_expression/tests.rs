use std::{cell::RefCell, rc::Rc, sync::Arc};

use afs_primitives::{
    bigint::{check_carry_mod_to_zero::CheckCarryModToZeroSubAir, utils::*},
    ecc::SampleEcPoints,
    sub_chip::LocalTraceInstructions,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use ax_sdk::{
    any_rap_vec, config::baby_bear_blake3::BabyBearBlake3Engine, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use num_bigint_dig::BigUint;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_matrix::dense::RowMajorMatrix;
use rand::RngCore;

use super::{ExprBuilder, FieldExprChip, LIMB_BITS};

pub fn generate_random_biguint(prime: &BigUint) -> BigUint {
    let mut rng = create_seeded_rng();
    let len = 32;
    let x = (0..len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x);
    x % prime
}

fn get_sub_air(prime: &BigUint) -> (CheckCarryModToZeroSubAir, Arc<VariableRangeCheckerChip>) {
    let field_element_bits = 30;
    let range_bus = 1;
    let range_decomp = 16;
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        range_bus,
        range_decomp,
    )));
    let subair = CheckCarryModToZeroSubAir::new(
        prime.clone(),
        LIMB_BITS,
        range_bus,
        range_decomp,
        field_element_bits,
    );
    (subair, range_checker)
}

#[test]
fn test_add() {
    let prime = secp256k1_coord_prime();
    let (subair, range_checker) = get_sub_air(&prime);

    let builder = ExprBuilder::new(prime.clone());
    let builder = Rc::new(RefCell::new(builder));
    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let mut x3 = x1 + x2;
    x3.save();
    let builder = builder.borrow().clone();

    let chip = FieldExprChip {
        builder,
        num_limbs: 32, // 256 bits / 8 bits per limb.
        check_carry_mod_to_zero: subair,
    };

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let inputs = vec![x, y];

    let row = chip.generate_trace_row((inputs, range_checker.clone()));
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&chip));
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis(
        &any_rap_vec![&chip, &range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_div() {
    let prime = secp256k1_coord_prime();
    let (subair, range_checker) = get_sub_air(&prime);

    let builder = ExprBuilder::new(prime.clone());
    let builder = Rc::new(RefCell::new(builder));
    let x1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let _x3 = x1 / x2; // auto save on division.
    let builder = builder.borrow().clone();

    let chip = FieldExprChip {
        builder,
        num_limbs: 32, // 256 bits / 8 bits per limb.
        check_carry_mod_to_zero: subair,
    };

    let x = generate_random_biguint(&prime);
    let y = generate_random_biguint(&prime);
    let inputs = vec![x, y];

    let row = chip.generate_trace_row((inputs, range_checker.clone()));
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&chip));
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis(
        &any_rap_vec![&chip, &range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_ec_add() {
    let prime = secp256k1_coord_prime();
    let (subair, range_checker) = get_sub_air(&prime);

    let builder = ExprBuilder::new(prime.clone());
    let builder = Rc::new(RefCell::new(builder));
    let x1 = ExprBuilder::new_input(builder.clone());
    let y1 = ExprBuilder::new_input(builder.clone());
    let x2 = ExprBuilder::new_input(builder.clone());
    let y2 = ExprBuilder::new_input(builder.clone());
    let dx = x2.clone() - x1.clone();
    let dy = y2.clone() - y1.clone();
    let lambda = dy / dx; // auto save on division.
    let mut x3 = lambda.clone() * lambda.clone() - x1.clone() - x2;
    x3.save();
    let mut y3 = lambda * (x1 - x3) - y1;
    y3.save();
    let builder = builder.borrow().clone();

    let chip = FieldExprChip {
        builder,
        num_limbs: 32, // 256 bits / 8 bits per limb.
        check_carry_mod_to_zero: subair,
    };

    let (x1, y1) = SampleEcPoints[0].clone();
    let (x2, y2) = SampleEcPoints[1].clone();
    let inputs = vec![x1, y1, x2, y2];

    let row = chip.generate_trace_row((inputs, range_checker.clone()));
    let trace = RowMajorMatrix::new(row, BaseAir::<BabyBear>::width(&chip));
    let range_trace = range_checker.generate_trace();

    BabyBearBlake3Engine::run_simple_test_no_pis(
        &any_rap_vec![&chip, &range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}
