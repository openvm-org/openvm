use ax_ecc_primitives::test_utils::{bls12381_fq12_random, bn254_fq12_random};
use axvm_ecc_curves::{BLS12381, BN254};
use axvm_instructions::{Bls12381Fp12Opcode, Bn254Fp12Opcode, Fp12Opcode, UsizeOpcode};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::fp12_multiply_expr;
use crate::{
    arch::{testing::VmChipTestBuilder, VmChipWrapper},
    intrinsics::field_expression::FieldExpressionCoreChip,
    rv32im::adapters::Rv32VecHeapAdapterChip,
    utils::{biguint_to_limbs, rv32_write_heap_default},
};

type F = BabyBear;

#[test]
fn test_fp12_multiply_bn254() {
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = BN254.MODULUS.clone();
    let xi = BN254.XI;
    let expr = fp12_multiply_expr(
        modulus,
        NUM_LIMBS,
        LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
        xi,
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        Bn254Fp12Opcode::default_offset(),
        vec![Fp12Opcode::MUL as usize],
        tester.memory_controller().borrow().range_checker.clone(),
        "Bn254Fp12Mul",
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 12, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let x = bn254_fq12_random(1);
    let y = bn254_fq12_random(2);

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

    let _res = chip.core.air.expr.execute([x, y].concat(), vec![]);

    let instruction = rv32_write_heap_default(
        &mut tester,
        x_limbs,
        y_limbs,
        chip.core.air.offset + Fp12Opcode::MUL as usize,
    );
    tester.execute(&mut chip, instruction);

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

// NOTE[yj]: This test requires RUST_MIN_STACK=8388608 to run without overflowing the stack, so it is ignored by the test runner for now
#[test]
#[ignore]
fn test_fp12_multiply_bls12381() {
    const NUM_LIMBS: usize = 64;
    const LIMB_BITS: usize = 8;

    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let modulus = BLS12381.MODULUS.clone();
    let xi = BLS12381.XI;
    let expr = fp12_multiply_expr(
        modulus,
        NUM_LIMBS,
        LIMB_BITS,
        tester.memory_controller().borrow().range_checker.bus(),
        xi,
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        Bls12381Fp12Opcode::default_offset(),
        vec![Fp12Opcode::MUL as usize],
        tester.memory_controller().borrow().range_checker.clone(),
        "Bls12381Fp12Mul",
    );
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 12, 12, NUM_LIMBS, NUM_LIMBS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );

    let x = bls12381_fq12_random(1);
    let y = bls12381_fq12_random(2);

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

    let _res = chip.core.air.expr.execute([x, y].concat(), vec![]);

    let instruction = rv32_write_heap_default(
        &mut tester,
        x_limbs,
        y_limbs,
        chip.core.air.offset + Fp12Opcode::MUL as usize,
    );
    tester.execute(&mut chip, instruction);

    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}
