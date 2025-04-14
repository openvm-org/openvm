use num_bigint::BigUint;
use openvm_circuit::arch::{
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    VmChipWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_mod_circuit_builder::{
    test_utils::{
        biguint_to_limbs, bls12381_fq12_random, bn254_fq12_random, bn254_fq12_to_biguint_vec,
    },
    ExprBuilderConfig, FieldExpr, FieldExpressionCoreChip,
};
use openvm_pairing_guest::{
    bls12_381::{
        BLS12_381_BLOCK_SIZE, BLS12_381_LIMB_BITS, BLS12_381_MODULUS, BLS12_381_NUM_LIMBS,
        BLS12_381_XI_ISIZE,
    },
    bn254::{BN254_BLOCK_SIZE, BN254_LIMB_BITS, BN254_MODULUS, BN254_NUM_LIMBS, BN254_XI_ISIZE},
};
use openvm_pairing_transpiler::{Bls12381Fp12Opcode, Bn254Fp12Opcode, Fp12Opcode};
use openvm_rv32_adapters::{rv32_write_heap_default, Rv32VecHeapAdapterChip};
use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::{fp12_add_expr, fp12_mul_expr, fp12_sub_expr};

type F = BabyBear;

#[allow(clippy::too_many_arguments)]
fn test_fp12_fn<
    const INPUT_SIZE: usize,
    const NUM_LIMBS: usize,
    const LIMB_BITS: usize,
    const BLOCK_SIZE: usize,
>(
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
        false,
    );
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let adapter =
        Rv32VecHeapAdapterChip::<F, 2, INPUT_SIZE, INPUT_SIZE, BLOCK_SIZE, BLOCK_SIZE>::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_bridge(),
            tester.address_bits(),
            bitwise_chip.clone(),
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
    let mut chip = VmChipWrapper::new(adapter, core, tester.offline_memory_mutex_arc());

    let res = chip.core.air.expr.execute([x, y].concat(), vec![]);
    assert_eq!(res.len(), var_len);

    let instruction = rv32_write_heap_default(
        &mut tester,
        x_limbs,
        y_limbs,
        chip.core.air.offset + local_opcode_idx,
    );
    tester.execute(&mut chip, &instruction);

    let run_tester = tester.build().load(chip).load(bitwise_chip).finalize();
    run_tester.simple_test().expect("Verification failed");
}

#[test]
fn test_fp12_add_bn254() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BN254_MODULUS.clone(),
        num_limbs: BN254_NUM_LIMBS,
        limb_bits: BN254_LIMB_BITS,
    };
    let expr = fp12_add_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bn254_fq12_to_biguint_vec(bn254_fq12_random(1));
    let y = bn254_fq12_to_biguint_vec(bn254_fq12_random(2));

    test_fp12_fn::<12, BN254_NUM_LIMBS, BN254_LIMB_BITS, BN254_BLOCK_SIZE>(
        tester,
        expr,
        Bn254Fp12Opcode::CLASS_OFFSET,
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
        modulus: BN254_MODULUS.clone(),
        num_limbs: BN254_NUM_LIMBS,
        limb_bits: BN254_LIMB_BITS,
    };
    let expr = fp12_sub_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bn254_fq12_to_biguint_vec(bn254_fq12_random(59));
    let y = bn254_fq12_to_biguint_vec(bn254_fq12_random(3));

    test_fp12_fn::<12, BN254_NUM_LIMBS, BN254_LIMB_BITS, BN254_BLOCK_SIZE>(
        tester,
        expr,
        Bn254Fp12Opcode::CLASS_OFFSET,
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
        modulus: BN254_MODULUS.clone(),
        num_limbs: BN254_NUM_LIMBS,
        limb_bits: BN254_LIMB_BITS,
    };
    let xi = BN254_XI_ISIZE;
    let expr = fp12_mul_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
        xi,
    );

    let x = bn254_fq12_to_biguint_vec(bn254_fq12_random(5));
    let y = bn254_fq12_to_biguint_vec(bn254_fq12_random(25));

    test_fp12_fn::<12, BN254_NUM_LIMBS, BN254_LIMB_BITS, BN254_BLOCK_SIZE>(
        tester,
        expr,
        Bn254Fp12Opcode::CLASS_OFFSET,
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
        modulus: BLS12_381_MODULUS.clone(),
        num_limbs: BLS12_381_NUM_LIMBS,
        limb_bits: BLS12_381_LIMB_BITS,
    };
    let expr = fp12_add_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bls12381_fq12_random(3);
    let y = bls12381_fq12_random(99);

    test_fp12_fn::<36, BLS12_381_NUM_LIMBS, BLS12_381_LIMB_BITS, BLS12_381_BLOCK_SIZE>(
        tester,
        expr,
        Bls12381Fp12Opcode::CLASS_OFFSET,
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
        modulus: BLS12_381_MODULUS.clone(),
        num_limbs: BLS12_381_NUM_LIMBS,
        limb_bits: BLS12_381_LIMB_BITS,
    };
    let expr = fp12_sub_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );

    let x = bls12381_fq12_random(8);
    let y = bls12381_fq12_random(9);

    test_fp12_fn::<36, BLS12_381_NUM_LIMBS, BLS12_381_LIMB_BITS, BLS12_381_BLOCK_SIZE>(
        tester,
        expr,
        Bls12381Fp12Opcode::CLASS_OFFSET,
        Fp12Opcode::SUB as usize,
        "Bls12381Fp12Sub",
        x,
        y,
        12,
    );
}

// NOTE[yj]: This test requires RUST_MIN_STACK=8388608 to run without overflowing the stack, so it
// is ignored by the test runner for now
#[test]
#[ignore]
fn test_fp12_mul_bls12381() {
    let tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: BLS12_381_MODULUS.clone(),
        num_limbs: BLS12_381_NUM_LIMBS,
        limb_bits: BLS12_381_LIMB_BITS,
    };
    let xi = BLS12_381_XI_ISIZE;
    let expr = fp12_mul_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
        xi,
    );

    let x = bls12381_fq12_random(5);
    let y = bls12381_fq12_random(25);

    test_fp12_fn::<36, BLS12_381_NUM_LIMBS, BLS12_381_LIMB_BITS, BLS12_381_BLOCK_SIZE>(
        tester,
        expr,
        Bls12381Fp12Opcode::CLASS_OFFSET,
        Fp12Opcode::MUL as usize,
        "Bls12381Fp12Mul",
        x,
        y,
        46,
    );
}
