use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{
            TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
            RANGE_TUPLE_CHECKER_BUS,
        },
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::{
        BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
        SharedBitwiseOperationLookupChip,
    },
    range_tuple::{
        RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip,
        SharedRangeTupleCheckerChip,
    },
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::MulWOpcode::{self, MULW};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::FieldAlgebra,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{MulWCoreAir, MulWFiller, Rv64MulWChip};
use crate::{
    adapters::{
        Rv64MultWAdapterAir, Rv64MultWAdapterCols, Rv64MultWAdapterExecutor, Rv64MultWAdapterFiller,
        RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
    },
    mul::MultiplicationCoreCols,
    test_utils::{get_verification_error, rv64_rand_write_register_or_imm},
    Rv64MulWAir, Rv64MulWExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const MAX_NUM_LIMBS: u32 = 32;
const TUPLE_CHECKER_SIZES: [u32; 2] = [
    (1u32 << RV64_CELL_BITS),
    (MAX_NUM_LIMBS * (1u32 << RV64_CELL_BITS)),
];

type Harness = TestChipHarness<F, Rv64MulWExecutor, Rv64MulWAir, Rv64MulWChip<F>>;
type MulWCoreCols<T> = MultiplicationCoreCols<T, RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>;

#[inline(always)]
fn run_mulw(
    x: &[u8; RV64_WORD_NUM_LIMBS],
    y: &[u8; RV64_WORD_NUM_LIMBS],
) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    let rs1 = u32::from_le_bytes(*x);
    let rs2 = u32::from_le_bytes(*y);
    let rd_word = rs1.wrapping_mul(rs2);
    (rd_word as i32 as i64 as u64).to_le_bytes()
}

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64MulWAir, Rv64MulWExecutor, Rv64MulWChip<F>) {
    let air = Rv64MulWAir::new(
        Rv64MultWAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
        MulWCoreAir::new(*range_tuple_chip.bus(), MulWOpcode::CLASS_OFFSET),
    );
    let executor = Rv64MulWExecutor::new(Rv64MultWAdapterExecutor, MulWOpcode::CLASS_OFFSET);
    let chip = Rv64MulWChip::<F>::new(
        MulWFiller::new(
            Rv64MultWAdapterFiller::new(bitwise_chip.clone()),
            range_tuple_chip,
            MulWOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV64_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    ),
    (RangeTupleCheckerAir<2>, SharedRangeTupleCheckerChip<2>),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);
    let range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        range_tuple_chip.clone(),
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (
        harness,
        (bitwise_chip.air, bitwise_chip),
        (range_tuple_chip.air, range_tuple_chip),
    )
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    let b = b.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));

    let (mut instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        b,
        c,
        None,
        MULW.global_opcode().as_usize(),
        rng,
    );
    instruction.e = F::ZERO;
    tester.execute(executor, arena, &instruction);

    let b_word: [u8; RV64_WORD_NUM_LIMBS] = b[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    let c_word: [u8; RV64_WORD_NUM_LIMBS] = c[..RV64_WORD_NUM_LIMBS].try_into().unwrap();
    let expected = run_mulw(&b_word, &c_word);
    assert_eq!(
        expected.map(F::from_canonical_u8),
        tester.read::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    );
    expected
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_rv64_mulw_rand_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut harness, bitwise, range_tuple) = create_harness(&tester);
    let num_ops = 100;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            None,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .load_periphery(range_tuple)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_mulw_test(
    prank_a: [u32; RV64_WORD_NUM_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_is_valid: bool,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_harness(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        Some(b),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut MulWCoreCols<F> = values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_canonical_u32);
        cols.is_valid = F::from_bool(prank_is_valid);
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .load_periphery(range_tuple)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn rv64_mulw_wrong_negative_test() {
    run_negative_mulw_test(
        [63, 247, 125, 234],
        [51, 109, 78, 142, 0, 0, 0, 0],
        [197, 85, 150, 32, 0, 0, 0, 0],
        true,
        true,
    );
}

#[test]
fn rv64_mulw_is_valid_false_negative_test() {
    run_negative_mulw_test(
        [63, 247, 125, 234],
        [51, 109, 78, 142, 0, 0, 0, 0],
        [197, 85, 150, 32, 0, 0, 0, 0],
        false,
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_mulw_sanity_test() {
    let x: [u8; RV64_WORD_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u8; RV64_WORD_NUM_LIMBS] = [51, 109, 78, 142];
    let (result, carry) = crate::mul::run_mul::<RV64_WORD_NUM_LIMBS, RV64_CELL_BITS>(&x, &y);
    let z: [u8; RV64_WORD_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV64_WORD_NUM_LIMBS] = [39, 100, 126, 205];
    for i in 0..RV64_WORD_NUM_LIMBS {
        assert_eq!(z[i], result[i]);
        assert_eq!(c[i], carry[i]);
    }
}

#[test]
fn run_mulw_sign_extension_test() {
    // MULW of 0x80000000 * 1 = 0x80000000, sign-extended to 0xFFFFFFFF_80000000
    let result = run_mulw(&[0, 0, 0, 128], &[1, 0, 0, 0]);
    assert_eq!(result, [0, 0, 0, 128, 255, 255, 255, 255]);

    // MULW of 1 * 1 = 1, sign-extended to 0x00000000_00000001
    let result = run_mulw(&[1, 0, 0, 0], &[1, 0, 0, 0]);
    assert_eq!(result, [1, 0, 0, 0, 0, 0, 0, 0]);

    // MULW of 0xFFFFFFFF * 0xFFFFFFFF = 1 (wrapping), sign-extended to 0x00000000_00000001
    let result = run_mulw(&[255, 255, 255, 255], &[255, 255, 255, 255]);
    assert_eq!(result, [1, 0, 0, 0, 0, 0, 0, 0]);
}
