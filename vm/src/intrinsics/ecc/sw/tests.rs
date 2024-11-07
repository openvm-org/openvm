use std::sync::Arc;

use ax_circuit_primitives::{
    bigint::utils::secp256k1_coord_prime,
    bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip},
};
use ax_ecc_primitives::field_expression::ExprBuilderConfig;
use axvm_ecc_constants::SampleEcPoints;
use axvm_instructions::{riscv::RV32_CELL_BITS, UsizeOpcode};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

use super::{ec_add_ne_expr, ec_double_expr};
use crate::{
    arch::{
        instructions::EccOpcode, testing::VmChipTestBuilder, VmChipWrapper, BITWISE_OP_LOOKUP_BUS,
    },
    intrinsics::field_expression::FieldExpressionCoreChip,
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
    let expr = ec_add_ne_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        EccOpcode::default_offset(),
        vec![EccOpcode::EC_ADD_NE as usize],
        vec![],
        tester.memory_controller().borrow().range_checker.clone(),
        "EcAddNe",
    );
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

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let r = chip
        .core
        .air
        .expr
        .execute(vec![p1_x, p1_y, p2_x, p2_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[2].0);
    assert_eq!(r[2], SampleEcPoints[2].1);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![p2_x_limbs, p2_y_limbs],
        chip.core.air.offset + EccOpcode::EC_ADD_NE as usize,
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
    let expr = ec_double_expr(
        config,
        tester.memory_controller().borrow().range_checker.bus(),
    );
    let core = FieldExpressionCoreChip::new(
        expr,
        EccOpcode::default_offset(),
        vec![EccOpcode::EC_DOUBLE as usize],
        vec![],
        tester.memory_controller().borrow().range_checker.clone(),
        "EcDouble",
    );
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

    let mut chip = VmChipWrapper::new(adapter, core, tester.memory_controller());

    let r = chip.core.air.expr.execute(vec![p1_x, p1_y], vec![]);
    assert_eq!(r.len(), 3); // lambda, x3, y3
    assert_eq!(r[1], SampleEcPoints[3].0);
    assert_eq!(r[2], SampleEcPoints[3].1);

    let instruction = rv32_write_heap_default(
        &mut tester,
        vec![p1_x_limbs, p1_y_limbs],
        vec![],
        chip.core.air.offset + EccOpcode::EC_DOUBLE as usize,
    );

    tester.execute(&mut chip, instruction);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();

    tester.simple_test().expect("Verification failed");
}
