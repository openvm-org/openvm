#[cfg(feature = "aot")]
use std::collections::HashMap;
use std::{array, borrow::BorrowMut, sync::Arc};

#[cfg(feature = "aot")]
use openvm_circuit::arch::{VmExecutor, VmState};
#[cfg(feature = "aot")]
use openvm_circuit::{
    arch::hasher::poseidon2::vm_poseidon2_hasher, system::memory::merkle::MerkleTree,
};
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
#[cfg(feature = "aot")]
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::Program,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    SystemOpcode,
};
#[cfg(feature = "aot")]
use openvm_rv32im_transpiler::BaseAluOpcode::ADD;
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
#[cfg(feature = "aot")]
use crate::Rv32ImConfig;
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

#[cfg(feature = "aot")]
fn run_mul_program(instructions: Vec<Instruction<F>>) -> (VmState<F>, VmState<F>) {
    let program = Program::from_instructions(&instructions);
    let exe = VmExe::new(program);
    let config = Rv32ImConfig::default();
    let memory_dimensions = config.rv32i.system.memory_config.memory_dimensions();
    let executor = VmExecutor::new(config.clone()).expect("failed to create Rv32IM executor");

    let interpreter = executor
        .interpreter_instance(&exe)
        .expect("interpreter build must succeed");
    let interp_state = interpreter
        .execute(vec![], None)
        .expect("interpreter execution must succeed");

    let aot_instance = executor.aot_instance(&exe).expect("AOT build must succeed");
    let aot_state = aot_instance
        .execute(vec![], None)
        .expect("AOT execution must succeed");

    assert_eq!(interp_state.pc(), aot_state.pc());

    let hasher = vm_poseidon2_hasher::<BabyBear>();
    let tree1 = MerkleTree::from_memory(&interp_state.memory.memory, &memory_dimensions, &hasher);
    let tree2 = MerkleTree::from_memory(&aot_state.memory.memory, &memory_dimensions, &hasher);
    assert_eq!(tree1.root(), tree2.root(), "Memory states differ");

    (interp_state, aot_state)
}

#[cfg(feature = "aot")]
fn read_register(state: &VmState<F>, offset: usize) -> u32 {
    let bytes = unsafe { state.memory.read::<u8, 4>(RV32_REGISTER_AS, offset as u32) };
    u32::from_le_bytes(bytes)
}

#[cfg(feature = "aot")]
fn add_immediate(rd: usize, imm: u32) -> Instruction<F> {
    Instruction::from_usize(
        ADD.global_opcode(),
        [
            rd,
            0,
            imm as usize,
            RV32_REGISTER_AS as usize,
            RV32_IMM_AS as usize,
        ],
    )
}

#[cfg(feature = "aot")]
fn mul_register(rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(
        MulOpcode::MUL.global_opcode(),
        [
            rd,
            rs1,
            rs2,
            RV32_REGISTER_AS as usize,
            RV32_REGISTER_AS as usize,
        ],
    )
}

#[cfg(feature = "aot")]
#[test]
fn test_aot_mul_basic() {
    let instructions = vec![
        add_immediate(4, 7),
        add_immediate(8, 11),
        mul_register(12, 4, 8),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let (interp_state, aot_state) = run_mul_program(instructions);

    let interp_x3 = read_register(&interp_state, 12);
    let aot_x3 = read_register(&aot_state, 12);
    assert_eq!(interp_x3, 77);
    assert_eq!(interp_x3, aot_x3);
}

#[cfg(feature = "aot")]
#[test]
fn test_aot_mul_upper_xmm() {
    let instructions = vec![
        add_immediate(4, 5),
        add_immediate(12, 9),
        mul_register(4, 4, 12),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let (interp_state, aot_state) = run_mul_program(instructions);

    let interp_x1 = read_register(&interp_state, 4);
    let aot_x1 = read_register(&aot_state, 4);
    assert_eq!(interp_x1, 45);
    assert_eq!(interp_x1, aot_x1);
}

#[cfg(feature = "aot")]
#[test]
fn test_aot_mul_randomized_pairs() {
    let offsets: [usize; 12] = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48];
    let mut rng = create_seeded_rng();
    let mut instructions = Vec::new();
    let mut expected = HashMap::new();

    for &offset in &offsets {
        let value_i32 = rng.gen_range(-(1i32 << 11)..(1i32 << 11));
        let imm_field = (value_i32 as u32) & 0x00FF_FFFF;
        instructions.push(add_immediate(offset, imm_field));
        expected.insert(offset, value_i32 as u32);
    }

    for (i, &rd_offset) in offsets.iter().enumerate() {
        let rs1_offset = offsets[i];
        let rs2_offset = offsets[(i + 3) % offsets.len()];
        instructions.push(mul_register(rd_offset, rs1_offset, rs2_offset));

        let rs1_val = *expected.get(&rs1_offset).unwrap();
        let rs2_val = *expected.get(&rs2_offset).unwrap();
        expected.insert(rd_offset, rs1_val.wrapping_mul(rs2_val));
    }

    instructions.push(Instruction::from_isize(
        SystemOpcode::TERMINATE.global_opcode(),
        0,
        0,
        0,
        0,
        0,
    ));

    let (interp_state, aot_state) = run_mul_program(instructions);

    for (offset, expected_val) in expected {
        let interp_val = read_register(&interp_state, offset);
        let aot_val = read_register(&aot_state, offset);
        assert_eq!(
            interp_val, expected_val,
            "unexpected value at offset {offset}"
        );
        assert_eq!(interp_val, aot_val, "AOT mismatch at offset {offset}");
    }
}

#[cfg(feature = "aot")]
#[test]
fn test_aot_mul_chained_dependencies() {
    let instructions = vec![
        add_immediate(4, 3),
        add_immediate(8, 5),
        mul_register(12, 4, 8),
        mul_register(4, 12, 8),
        mul_register(8, 4, 12),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let (interp_state, aot_state) = run_mul_program(instructions);

    let interp_x3 = read_register(&interp_state, 12);
    let aot_x3 = read_register(&aot_state, 12);
    assert_eq!(interp_x3, 15);
    assert_eq!(interp_x3, aot_x3);

    let interp_x1 = read_register(&interp_state, 4);
    let aot_x1 = read_register(&aot_state, 4);
    assert_eq!(interp_x1, 75);
    assert_eq!(interp_x1, aot_x1);

    let interp_x2 = read_register(&interp_state, 8);
    let aot_x2 = read_register(&aot_state, 8);
    assert_eq!(interp_x2, 1125);
    assert_eq!(interp_x2, aot_x2);
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
