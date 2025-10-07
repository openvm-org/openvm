use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, RANGE_TUPLE_CHECKER_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::range_tuple::{
    RangeTupleCheckerAir, RangeTupleCheckerBus, RangeTupleCheckerChip, SharedRangeTupleCheckerChip,
};
use openvm_instructions::LocalOpcode;
use openvm_rv32im_transpiler::MulOpcode::{self, MUL};
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
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv32MultAdapterRecord, MultiplicationCoreRecord, Rv32MultiplicationChipGpu},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::core::run_mul;
use crate::{
    adapters::{
        Rv32MultAdapterAir, Rv32MultAdapterExecutor, Rv32MultAdapterFiller, RV32_CELL_BITS,
        RV32_REGISTER_NUM_LIMBS,
    },
    mul::{MultiplicationCoreCols, Rv32MultiplicationChip},
    test_utils::{get_verification_error, rv32_rand_write_register_or_imm},
    MultiplicationCoreAir, MultiplicationFiller, Rv32MultiplicationAir, Rv32MultiplicationExecutor,
};

const MAX_INS_CAPACITY: usize = 128;
// the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
const MAX_NUM_LIMBS: u32 = 32;
const TUPLE_CHECKER_SIZES: [u32; 2] = [
    (1u32 << RV32_CELL_BITS),
    (MAX_NUM_LIMBS * (1u32 << RV32_CELL_BITS)),
];

type F = BabyBear;
type Harness = TestChipHarness<
    F,
    Rv32MultiplicationExecutor,
    Rv32MultiplicationAir,
    Rv32MultiplicationChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv32MultiplicationAir,
    Rv32MultiplicationExecutor,
    Rv32MultiplicationChip<F>,
) {
    let air = Rv32MultiplicationAir::new(
        Rv32MultAdapterAir::new(execution_bridge, memory_bridge),
        MultiplicationCoreAir::new(*range_tuple_chip.bus(), MulOpcode::CLASS_OFFSET),
    );
    let executor =
        Rv32MultiplicationExecutor::new(Rv32MultAdapterExecutor, MulOpcode::CLASS_OFFSET);
    let chip = Rv32MultiplicationChip::<F>::new(
        MultiplicationFiller::new(
            Rv32MultAdapterFiller,
            range_tuple_chip,
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
) {
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);
    let range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_tuple_chip.clone(),
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (range_tuple_chip.air, range_tuple_chip))
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: MulOpcode,
    b: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));

    let (mut instruction, rd) =
        rv32_rand_write_register_or_imm(tester, b, c, None, opcode.global_opcode().as_usize(), rng);

    instruction.e = F::ZERO;
    tester.execute(executor, arena, &instruction);

    let (a, _) = run_mul::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&b, &c);
    assert_eq!(
        a.map(F::from_canonical_u8),
        tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd)
    )
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_rv32_mul_rand_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut harness, range_tuple) = create_harness(&mut tester);
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
fn run_negative_mul_test(
    opcode: MulOpcode,
    prank_a: [u32; RV32_REGISTER_NUM_LIMBS],
    b: [u8; RV32_REGISTER_NUM_LIMBS],
    c: [u8; RV32_REGISTER_NUM_LIMBS],
    prank_is_valid: bool,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, range_tuple) = create_harness(&mut tester);

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
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut MultiplicationCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_canonical_u32);
        cols.is_valid = F::from_bool(prank_is_valid);
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(range_tuple)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn rv32_mul_wrong_negative_test() {
    run_negative_mul_test(
        MUL,
        [63, 247, 125, 234],
        [51, 109, 78, 142],
        [197, 85, 150, 32],
        true,
        true,
    );
}

#[test]
fn rv32_mul_is_valid_false_negative_test() {
    run_negative_mul_test(
        MUL,
        [63, 247, 125, 234],
        [51, 109, 78, 142],
        [197, 85, 150, 32],
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
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 205];
    let (result, carry) = run_mul::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
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
    Rv32MultiplicationExecutor,
    Rv32MultiplicationAir,
    Rv32MultiplicationChipGpu,
    Rv32MultiplicationChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);
    let dummy_range_tuple_chip = Arc::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_tuple_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32MultiplicationChipGpu::new(
        tester.range_checker(),
        tester.range_tuple_checker(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_rand_mul_tracegen() {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default().with_range_tuple_checker(
        RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES),
    );

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
        &'a mut Rv32MultAdapterRecord,
        &'a mut MultiplicationCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record<'_>, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32MultAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
