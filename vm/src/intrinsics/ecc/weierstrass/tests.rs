use std::sync::Arc;

use ax_circuit_primitives::{
    bigint::utils::{secp256k1_coord_prime, secp256r1_coord_prime},
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
};
use ax_ecc_primitives::field_expression::ExprBuilderConfig;
use axvm_ecc_constants::SampleEcPoints;
use axvm_instructions::{riscv::RV32_CELL_BITS, UsizeOpcode};
use num_bigint_dig::BigUint;
use num_traits::{Num, Zero};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use crate::{
    arch::{
        instructions::Rv32WeierstrassOpcode, testing::VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
    },
    intrinsics::ecc::weierstrass::{EcAddNeChip, EcDoubleChip},
    rv32im::adapters::Rv32VecHeapAdapterChip,
    utils::{biguint_to_limbs, rv32_write_heap_default},
};

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;
const BLOCK_SIZE: usize = 32;
type F = BabyBear;

#[test]
fn test_add_ne() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: secp256k1_coord_prime(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let adapter = Rv32VecHeapAdapterChip::<F, 2, 2, 2, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        bitwise_chip.clone(),
    );
    let mut chip = EcAddNeChip::new(
        adapter,
        tester.memory_controller(),
        config,
        Rv32WeierstrassOpcode::default_offset(),
    );

    let (p1_x, p1_y) = SampleEcPoints[0].clone();
    let (p2_x, p2_y) = SampleEcPoints[1].clone();

    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p2_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p2_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p2_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p2_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    let r = chip
        .0
        .core
        .expr()
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[2].0);
    assert_eq!(r[2], SampleEcPoints[2].1);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![p2_x_limbs, p2_y_limbs],
        chip.0.core.air.offset + Rv32WeierstrassOpcode::EC_ADD_NE as usize,
    );

    tester.execute(&mut chip, instruction);

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_double() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: secp256k1_coord_prime(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let adapter = Rv32VecHeapAdapterChip::<F, 1, 2, 2, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        bitwise_chip.clone(),
    );

    let (p1_x, p1_y) = SampleEcPoints[1].clone();
    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    let mut chip = EcDoubleChip::new(
        adapter,
        tester.memory_controller(),
        config,
        Rv32WeierstrassOpcode::default_offset(),
        BigUint::zero(),
    );

    let r = chip.0.core.expr().execute(vec![p1_x, p1_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[3].0);
    assert_eq!(r[2], SampleEcPoints[3].1);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![],
        chip.0.core.air.offset + Rv32WeierstrassOpcode::EC_DOUBLE as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn test_p256_double() {
    let mut tester: VmChipTestBuilder<F> = VmChipTestBuilder::default();
    let config = ExprBuilderConfig {
        modulus: secp256r1_coord_prime(),
        num_limbs: NUM_LIMBS,
        limb_bits: LIMB_BITS,
    };
    let a = BigUint::from_str_radix(
        "ffffffff00000001000000000000000000000000fffffffffffffffffffffffc",
        16,
    )
    .unwrap();
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let adapter = Rv32VecHeapAdapterChip::<F, 1, 2, 2, BLOCK_SIZE, BLOCK_SIZE>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        bitwise_chip.clone(),
    );

    // Testing data from: http://point-at-infinity.org/ecc/nisttv
    let p1_x = BigUint::from_str_radix(
        "6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
        16,
    )
    .unwrap();
    let p1_y = BigUint::from_str_radix(
        "4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
        16,
    )
    .unwrap();
    let p1_x_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_x.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);
    let p1_y_limbs =
        biguint_to_limbs::<NUM_LIMBS>(p1_y.clone(), LIMB_BITS).map(BabyBear::from_canonical_u32);

    let mut chip = EcDoubleChip::new(
        adapter,
        tester.memory_controller(),
        config,
        Rv32WeierstrassOpcode::default_offset(),
        a,
    );

    let r = chip.0.core.expr().execute(vec![p1_x, p1_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    let expected_double_x = BigUint::from_str_radix(
        "7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978",
        16,
    )
    .unwrap();
    let expected_double_y = BigUint::from_str_radix(
        "07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1",
        16,
    )
    .unwrap();
    assert_eq!(r[1], expected_double_x);
    assert_eq!(r[2], expected_double_y);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![],
        chip.0.core.air.offset + Rv32WeierstrassOpcode::EC_DOUBLE as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}
