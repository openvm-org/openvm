use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{
            TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
            RANGE_TUPLE_CHECKER_BUS,
        },
        Arena, ExecutionBridge, MemoryConfig, Postflight, PreflightExecutor, PreflightHistory,
        PreflightProgramEvent, TraceFiller, BLOCK_FE_WIDTH,
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
use openvm_instructions::{
    instruction::Instruction, program::Program, riscv::RV64_REGISTER_AS, LocalOpcode,
};
use openvm_riscv_transpiler::MulOpcode::{self, MUL};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv64MultAdapterRecord, MultiplicationCoreRecord, Rv64MultiplicationChipGpu},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{core::run_mul, trace::generate_trace_from_postflight};
use crate::{
    adapters::{
        Rv64MultAdapterAir, Rv64MultAdapterExecutor, Rv64MultAdapterFiller, RV64_BYTE_BITS,
        RV64_REGISTER_NUM_LIMBS,
    },
    mul::{MultiplicationCoreCols, Rv64MultiplicationChip},
    test_utils::rv64_rand_write_register_or_imm,
    MultiplicationCoreAir, MultiplicationFiller, Rv64MultiplicationAir, Rv64MultiplicationExecutor,
};

const MAX_INS_CAPACITY: usize = 128;
// the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
const MAX_NUM_LIMBS: u32 = 32;
const TUPLE_CHECKER_SIZES: [u32; 2] = [
    (1u32 << RV64_BYTE_BITS),
    (MAX_NUM_LIMBS * (1u32 << RV64_BYTE_BITS)),
];

type F = BabyBear;
type Harness = TestChipHarness<
    F,
    Rv64MultiplicationExecutor,
    Rv64MultiplicationAir,
    Rv64MultiplicationChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_BYTE_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64MultiplicationAir,
    Rv64MultiplicationExecutor,
    Rv64MultiplicationChip<F>,
) {
    let air = Rv64MultiplicationAir::new(
        Rv64MultAdapterAir::new(execution_bridge, memory_bridge),
        MultiplicationCoreAir::new(
            *range_tuple_chip.bus(),
            bitwise_chip.bus(),
            MulOpcode::CLASS_OFFSET,
        ),
    );
    let executor =
        Rv64MultiplicationExecutor::new(Rv64MultAdapterExecutor, MulOpcode::CLASS_OFFSET);
    let chip = Rv64MultiplicationChip::<F>::new(
        MultiplicationFiller::new(
            Rv64MultAdapterFiller,
            range_tuple_chip,
            bitwise_chip,
            MulOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
    (RangeTupleCheckerAir<2>, SharedRangeTupleCheckerChip<2>),
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
) {
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);
    let range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_tuple_chip.clone(),
        bitwise_chip.clone(),
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (
        harness,
        (range_tuple_chip.air, range_tuple_chip),
        (bitwise_chip.air, bitwise_chip),
    )
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: MulOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));

    let (mut instruction, rd) =
        rv64_rand_write_register_or_imm(tester, b, c, None, opcode.global_opcode().as_usize(), rng);

    instruction.e = F::ZERO;
    tester.execute(executor, arena, &instruction);

    let (a, _) = run_mul::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(&b, &c);
    assert_eq!(
        a.map(F::from_u8),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
    )
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_rv64_mul_rand_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut harness, range_tuple, bitwise) = create_harness(&mut tester);
    let num_ops = 100;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            MUL,
            None,
            None,
        );
    }

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(range_tuple)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn postflight_trace_matches_record_arena_trace() {
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, _, _) = create_harness(&mut tester);
    let mul = Instruction::from_usize(
        MUL.global_opcode(),
        [24, 8, 16, RV64_REGISTER_AS as usize, 0],
    );
    let sentinel = Instruction::from_usize(
        MUL.global_opcode(),
        [32, 8, 16, RV64_REGISTER_AS as usize, 0],
    );
    unsafe {
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_REGISTER_AS,
            4,
            [0x0201, 0x0403, 0x0605, 0x0807],
        );
        tester.memory.memory.data.write::<u16, BLOCK_FE_WIDTH>(
            RV64_REGISTER_AS,
            8,
            [0x0a09, 0x0c0b, 0x0e0d, 0x100f],
        );
    }
    tester.execute_with_pc(&mut harness.executor, &mut harness.arena, &mul, 0);

    let history = PreflightHistory {
        program: vec![
            PreflightProgramEvent {
                pc: 0,
                timestamp: 1,
            },
            PreflightProgramEvent {
                pc: 4,
                timestamp: 4,
            },
        ],
        memory: tester.memory.memory.take_log(),
    };
    let program = Program::new_without_debug_infos(&[mul, sentinel], 0);
    let memory_config = MemoryConfig::default();
    let postflight = Postflight::new(&program, &history, &memory_config, None).unwrap();
    let actual = generate_trace_from_postflight(&harness.chip, &postflight).unwrap();
    let actual_range_tuple = harness.chip.inner.range_tuple_chip.generate_trace::<F>();
    let actual_bitwise = harness.chip.inner.bitwise_lookup_chip.generate_trace::<F>();

    let rows_used = harness.arena.trace_offset / harness.arena.width;
    let mut expected_values = harness.arena.trace_buffer;
    expected_values.truncate(rows_used.next_power_of_two() * harness.arena.width);
    let mut expected = RowMajorMatrix::new(expected_values, harness.arena.width);
    harness.chip.inner.fill_trace(
        &harness.chip.mem_helper.as_borrowed(),
        &mut expected,
        rows_used,
    );
    let expected_range_tuple = harness.chip.inner.range_tuple_chip.generate_trace::<F>();
    let expected_bitwise = harness.chip.inner.bitwise_lookup_chip.generate_trace::<F>();

    assert_eq!(actual.width(), expected.width());
    assert_eq!(actual.height(), expected.height());
    assert_eq!(actual.values, expected.values);
    assert_eq!(actual_range_tuple.values, expected_range_tuple.values);
    assert_eq!(actual_bitwise.values, expected_bitwise.values);
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_mul_test(
    opcode: MulOpcode,
    prank_a: [u32; RV64_REGISTER_NUM_LIMBS],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_is_valid: bool,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, range_tuple, bitwise) = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut MultiplicationCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_u32);
        cols.is_valid = F::from_bool(prank_is_valid);
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(range_tuple)
        .load_periphery(bitwise)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_mul_wrong_negative_test() {
    run_negative_mul_test(
        MUL,
        [63, 247, 125, 232, 252, 163, 203, 218],
        [51, 109, 78, 142, 73, 35, 25, 206],
        [197, 85, 150, 32, 88, 77, 201, 19],
        true,
        true,
    );
}

#[test]
fn rv64_mul_is_valid_false_negative_test() {
    run_negative_mul_test(
        MUL,
        [63, 247, 125, 232, 252, 163, 203, 218],
        [51, 109, 78, 142, 73, 35, 25, 206],
        [197, 85, 150, 32, 88, 77, 201, 19],
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
fn run_mul_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [229, 33, 29, 111, 145, 34, 25, 205];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [51, 109, 78, 142, 73, 35, 25, 206];
    let z: [u8; RV64_REGISTER_NUM_LIMBS] = [159, 65, 2, 228, 66, 204, 249, 3];
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = [45, 104, 90, 171, 169, 159, 160, 366];
    let (result, carry) = run_mul::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(&x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i]);
        assert_eq!(c[i], carry[i]);
    }
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64MultiplicationExecutor,
    Rv64MultiplicationAir,
    Rv64MultiplicationChipGpu,
    Rv64MultiplicationChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);
    let dummy_range_tuple_chip = Arc::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_tuple_chip,
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64MultiplicationChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.range_tuple_checker(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_mul_tracegen() {
    use openvm_circuit::arch::testing::BITWISE_OP_LOOKUP_BUS;
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS))
        .with_range_tuple_checker(RangeTupleCheckerBus::new(
            RANGE_TUPLE_CHECKER_BUS,
            TUPLE_CHECKER_SIZES,
        ));

    let mut harness = create_cuda_harness(&tester);
    let num_ops = 100;
    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            MulOpcode::MUL,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64MultAdapterRecord,
        &'a mut MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record<'_>, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64MultAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
