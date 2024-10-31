use ax_ecc_primitives::{
    field_expression::{ExprBuilderConfig, FieldExpr},
    test_utils::{bls12381_fq12_random, bn254_fq12_random, bn254_fq12_to_biguint_vec},
};
use axvm_ecc_constants::{BLS12381, BN254};
use axvm_instructions::{Bls12381Fp12Opcode, Bn254Fp12Opcode, Fp12Opcode, UsizeOpcode};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{fp12_add_expr, fp12_mul_expr, fp12_sub_expr};
use crate::{
    arch::{testing::VmChipTestBuilder, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip,
    utils::{biguint_to_limbs, rv32_write_heap_default},
};

const BN254_NUM_LIMBS: usize = 32;
const BN254_LIMB_BITS: usize = 8;

const BLS12381_NUM_LIMBS: usize = 64;
const BLS12381_LIMB_BITS: usize = 8;

type F = BabyBear;

#[allow(clippy::too_many_arguments)]
fn test_fp12_fn<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    mut tester: VmChipTestBuilder<F>,
    expr: FieldExpr,
    offset: usize,
    local_opcode_idx: usize,
    name: &str,
    x: Vec<BigUint>,
    y: Vec<BigUint>,
    var_len: usize,
) {
    let core = FieldExpressionCoreChip::new(
        expr,
        offset,
        vec![local_opcode_idx],
        vec![],
        tester.memory_controller().borrow().range_checker.clone(),
        name,
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 12, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let x_limbs = x
        .iter()
        .map(|x| {
            biguint_to_limbs::<NUM_LIMBS>(x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<[BabyBear; NUM_LIMBS]>>();
    let y_limbs = y
        .iter()
        .map(|y| {
            biguint_to_limbs::<NUM_LIMBS>(y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32)
        })
        .collect::<Vec<[BabyBear; NUM_LIMBS]>>();
    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let res = chip.core.air.expr.execute([x, y].concat(), vec![]);
    assert_eq!(res.len(), var_len);

    let instruction = rv32_write_heap_default(
        &mut tester,
        x_limbs,
        y_limbs,
        chip.core.air.offset + local_opcode_idx,
    );
    tester.execute(&mut chip, instruction);

    let run_tester = tester.build().load(chip).finalize();
    run_tester.simple_test().expect("Verification failed");
}

#[test]
fn test_fp12_add_bn254() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BN254.MODULUS.clone(),
        num_limbs: BN254_NUM_LIMBS,
        limb_bits: BN254_LIMB_BITS,
    };
    let expr = fp12_add_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bn254_fq12_to_biguint_vec(bn254_fq12_random(1));
    let y = bn254_fq12_to_biguint_vec(bn254_fq12_random(2));

    test_fp12_fn::<BN254_NUM_LIMBS, BN254_LIMB_BITS>(
        tester,
        expr,
        Bn254Fp12Opcode::default_offset(),
        Fp12Opcode::ADD as usize,
        "Bn254Fp12Add",
        x,
        y,
        12,
    );
}

#[test]
fn test_fp12_sub_bn254() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BN254.MODULUS.clone(),
        num_limbs: BN254_NUM_LIMBS,
        limb_bits: BN254_LIMB_BITS,
    };
    let expr = fp12_sub_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bn254_fq12_to_biguint_vec(bn254_fq12_random(59));
    let y = bn254_fq12_to_biguint_vec(bn254_fq12_random(3));

    test_fp12_fn::<BN254_NUM_LIMBS, BN254_LIMB_BITS>(
        tester,
        expr,
        Bn254Fp12Opcode::default_offset(),
        Fp12Opcode::SUB as usize,
        "Bn254Fp12Sub",
        x,
        y,
        12,
    );
}

#[test]
fn test_fp12_mul_bn254() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BN254.MODULUS.clone(),
        num_limbs: BN254_NUM_LIMBS,
        limb_bits: BN254_LIMB_BITS,
    };
    let xi = BN254.XI;
    let expr = fp12_mul_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
        xi,
    );

    let x = bn254_fq12_to_biguint_vec(bn254_fq12_random(5));
    let y = bn254_fq12_to_biguint_vec(bn254_fq12_random(25));

    test_fp12_fn::<BN254_NUM_LIMBS, BN254_LIMB_BITS>(
        tester,
        expr,
        Bn254Fp12Opcode::default_offset(),
        Fp12Opcode::MUL as usize,
        "Bn254Fp12Mul",
        x,
        y,
        33,
    );
}

#[test]
fn test_fp12_add_bls12381() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BLS12381.MODULUS.clone(),
        num_limbs: BLS12381_NUM_LIMBS,
        limb_bits: BLS12381_LIMB_BITS,
    };
    let expr = fp12_add_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bls12381_fq12_random(3);
    let y = bls12381_fq12_random(99);

    test_fp12_fn::<BLS12381_NUM_LIMBS, BLS12381_LIMB_BITS>(
        tester,
        expr,
        Bls12381Fp12Opcode::default_offset(),
        Fp12Opcode::ADD as usize,
        "Bls12381Fp12Add",
        x,
        y,
        12,
    );
}

#[test]
fn test_fp12_sub_bls12381() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BLS12381.MODULUS.clone(),
        num_limbs: BLS12381_NUM_LIMBS,
        limb_bits: BLS12381_LIMB_BITS,
    };
    let expr = fp12_sub_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bls12381_fq12_random(8);
    let y = bls12381_fq12_random(9);

    test_fp12_fn::<BLS12381_NUM_LIMBS, BLS12381_LIMB_BITS>(
        tester,
        expr,
        Bls12381Fp12Opcode::default_offset(),
        Fp12Opcode::SUB as usize,
        "Bls12381Fp12Sub",
        x,
        y,
        12,
    );
}

// NOTE[yj]: This test requires RUST_MIN_STACK=8388608 to run without overflowing the stack, so it is ignored by the test runner for now
#[test]
#[ignore]
fn test_fp12_mul_bls12381() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BLS12381.MODULUS.clone(),
        num_limbs: BLS12381_NUM_LIMBS,
        limb_bits: BLS12381_LIMB_BITS,
    };
    let xi = BLS12381.XI;
    let expr = fp12_mul_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
        xi,
    );

    let x = bls12381_fq12_random(5);
    let y = bls12381_fq12_random(25);

    test_fp12_fn::<BLS12381_NUM_LIMBS, BLS12381_LIMB_BITS>(
        tester,
        expr,
        Bls12381Fp12Opcode::default_offset(),
        Fp12Opcode::MUL as usize,
        "Bls12381Fp12Mul",
        x,
        y,
        82,
    );
}
