#[cfg(feature = "aot")]
use std::collections::HashMap;
use std::{borrow::BorrowMut, sync::Arc};

#[cfg(feature = "aot")]
use openvm_circuit::arch::VirtualMachine;
#[cfg(feature = "aot")]
use openvm_circuit::arch::{VmExecutor, VmState};
#[cfg(feature = "aot")]
use openvm_circuit::{
    arch::hasher::poseidon2::vm_poseidon2_hasher, system::memory::merkle::MerkleTree,
};
use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS, RANGE_TUPLE_CHECKER_BUS,
        },
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::generate_long_number,
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
#[cfg(feature = "aot")]
use openvm_instructions::{
    exe::VmExe,
    program::Program,
    riscv::{RV32_IMM_AS, RV32_REGISTER_AS},
    SystemOpcode,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
#[cfg(feature = "aot")]
use openvm_rv32im_transpiler::BaseAluOpcode::ADD;
use openvm_rv32im_transpiler::MulHOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::FieldAlgebra,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
#[cfg(feature = "aot")]
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;
#[cfg(feature = "aot")]
use openvm_stark_sdk::config::FriParameters;
#[cfg(feature = "aot")]
use openvm_stark_sdk::engine::StarkFriEngine;
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::rngs::StdRng;
#[cfg(feature = "aot")]
use rand::Rng;
use test_case::test_case;
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv32MultAdapterRecord, MulHCoreRecord, Rv32MulHChipGpu},
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::core::run_mulh;
use crate::{
    adapters::{
        Rv32MultAdapterAir, Rv32MultAdapterExecutor, Rv32MultAdapterFiller, RV32_CELL_BITS,
        RV32_REGISTER_NUM_LIMBS,
    },
    mulh::{MulHCoreCols, Rv32MulHChip},
    test_utils::get_verification_error,
    MulHCoreAir, MulHFiller, Rv32MulHAir, Rv32MulHExecutor,
};
#[cfg(feature = "aot")]
use crate::{Rv32ImBuilder, Rv32ImConfig};

const MAX_INS_CAPACITY: usize = 128;
// the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
const MAX_NUM_LIMBS: u32 = 32;
const TUPLE_CHECKER_SIZES: [u32; 2] = [
    (1u32 << RV32_CELL_BITS),
    (MAX_NUM_LIMBS * (1u32 << RV32_CELL_BITS)),
];
type F = BabyBear;
type Harness = TestChipHarness<F, Rv32MulHExecutor, Rv32MulHAir, Rv32MulHChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv32MulHAir, Rv32MulHExecutor, Rv32MulHChip<F>) {
    let air = Rv32MulHAir::new(
        Rv32MultAdapterAir::new(execution_bridge, memory_bridge),
        MulHCoreAir::new(bitwise_chip.bus(), *range_tuple_chip.bus()),
    );
    let executor = Rv32MulHExecutor::new(Rv32MultAdapterExecutor, MulHOpcode::CLASS_OFFSET);
    let chip = Rv32MulHChip::<F>::new(
        MulHFiller::new(Rv32MultAdapterFiller, bitwise_chip, range_tuple_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
    (RangeTupleCheckerAir<2>, SharedRangeTupleCheckerChip<2>),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
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
    opcode: MulHOpcode,
    b: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    c: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(generate_long_number::<
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >(rng));
    let c = c.unwrap_or(generate_long_number::<
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >(rng));

    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);
    let rd = gen_pointer(rng, 4);

    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 0]),
    );

    let (a, _, _, _, _) = run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode, &b, &c);
    assert_eq!(
        a.map(F::from_canonical_u32),
        tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd)
    );
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(MULH, 100)]
#[test_case(MULHSU, 100)]
#[test_case(MULHU, 100)]
fn run_rv32_mulh_rand_test(opcode: MulHOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_harness(&mut tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
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
fn run_negative_mulh_test(
    opcode: MulHOpcode,
    prank_a: [u32; RV32_REGISTER_NUM_LIMBS],
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    c: [u32; RV32_REGISTER_NUM_LIMBS],
    prank_a_mul: [u32; RV32_REGISTER_NUM_LIMBS],
    prank_b_ext: u32,
    prank_c_ext: u32,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_harness(&mut tester);

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
        let cols: &mut MulHCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_canonical_u32);
        cols.a_mul = prank_a_mul.map(F::from_canonical_u32);
        cols.b_ext = F::from_canonical_u32(prank_b_ext);
        cols.c_ext = F::from_canonical_u32(prank_c_ext);
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
fn rv32_mulh_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULH,
        [130, 9, 135, 241],
        [197, 85, 150, 32],
        [51, 109, 78, 142],
        [63, 247, 125, 234],
        0,
        255,
        true,
    );
}

#[test]
fn rv32_mulh_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULH,
        [130, 9, 135, 242],
        [197, 85, 150, 32],
        [51, 109, 78, 142],
        [63, 247, 125, 232],
        0,
        255,
        true,
    );
}

#[test]
fn rv32_mulh_wrong_ext_negative_test() {
    run_negative_mulh_test(
        MULH,
        [1, 0, 0, 0],
        [0, 0, 0, 128],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        0,
        0,
        true,
    );
}

#[test]
fn rv32_mulh_invalid_ext_negative_test() {
    run_negative_mulh_test(
        MULH,
        [3, 2, 2, 2],
        [0, 0, 0, 128],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        1,
        0,
        false,
    );
}

#[test]
fn rv32_mulhsu_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [174, 40, 246, 202],
        [197, 85, 150, 160],
        [51, 109, 78, 142],
        [63, 247, 125, 105],
        255,
        0,
        true,
    );
}

#[test]
fn rv32_mulhsu_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [174, 40, 246, 201],
        [197, 85, 150, 160],
        [51, 109, 78, 142],
        [63, 247, 125, 104],
        255,
        0,
        true,
    );
}

#[test]
fn rv32_mulhsu_wrong_b_ext_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [1, 0, 0, 0],
        [0, 0, 0, 128],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        0,
        0,
        true,
    );
}

#[test]
fn rv32_mulhsu_wrong_c_ext_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [0, 0, 0, 64],
        [0, 0, 0, 128],
        [0, 0, 0, 128],
        [0, 0, 0, 0],
        255,
        255,
        false,
    );
}

#[test]
fn rv32_mulhu_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [130, 9, 135, 241],
        [197, 85, 150, 32],
        [51, 109, 78, 142],
        [63, 247, 125, 234],
        0,
        0,
        true,
    );
}

#[test]
fn rv32_mulhu_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [130, 9, 135, 240],
        [197, 85, 150, 32],
        [51, 109, 78, 142],
        [63, 247, 125, 232],
        0,
        0,
        true,
    );
}

#[test]
fn rv32_mulhu_wrong_ext_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [255, 255, 255, 255],
        [0, 0, 0, 128],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        255,
        0,
        false,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_mulh_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [130, 9, 135, 241];
    let z_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [303, 375, 449, 463];
    let c_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 205];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(MULH, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV32_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 255);
}

#[test]
fn run_mulhu_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [71, 95, 29, 18];
    let z_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [107, 93, 18, 0];
    let c_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 205];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(MULHU, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV32_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 0);
}

#[test]
fn run_mulhsu_pos_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [71, 95, 29, 18];
    let z_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [107, 93, 18, 0];
    let c_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 205];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(MULHSU, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV32_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 0);
}

#[test]
fn run_mulhsu_neg_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 160];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [174, 40, 246, 202];
    let z_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 104];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [212, 292, 326, 379];
    let c_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 231];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(MULHSU, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV32_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 255);
    assert_eq!(y_ext, 0);
}

//////////////////////////////////////////////////////////////////////////////////////
// AOT TESTS
//////////////////////////////////////////////////////////////////////////////////////

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

    // Also test metered execution (interpreter and AOT) produce identical final state
    let engine = BabyBearPoseidon2Engine::new(FriParameters::new_for_testing(3));
    let (vm, _) =
        VirtualMachine::new_with_keygen(engine, Rv32ImBuilder, config.clone()).expect("vm init");
    let executor_idx_to_air_idx = vm.executor_idx_to_air_idx();
    let metered_ctx = vm.build_metered_ctx(&exe);

    let metered_interp = vm
        .executor()
        .metered_interpreter_instance(&exe, &executor_idx_to_air_idx)
        .expect("metered interpreter build must succeed");
    let (_, metered_interp_state) = metered_interp
        .execute_metered(vec![], metered_ctx.clone())
        .expect("metered interpreter execution must succeed");

    let metered_aot = vm
        .executor()
        .metered_aot_instance(&exe, &executor_idx_to_air_idx)
        .expect("metered AOT build must succeed");
    let (_, metered_aot_state) = metered_aot
        .execute_metered(vec![], metered_ctx.clone())
        .expect("metered AOT execution must succeed");
    println!(
        "interp_state.pc(): {}, metered_interp_state.pc(): {}",
        interp_state.pc(),
        metered_interp_state.pc()
    );
    assert_eq!(metered_aot_state.pc(), metered_interp_state.pc());
    let tree_mi = MerkleTree::from_memory(
        &metered_interp_state.memory.memory,
        &memory_dimensions,
        &hasher,
    );
    let tree_ma = MerkleTree::from_memory(
        &metered_aot_state.memory.memory,
        &memory_dimensions,
        &hasher,
    );

    assert_eq!(
        tree_ma.root(),
        tree_mi.root(),
        "Metered interpreter memory differs"
    );
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
fn mulh_register(op: MulHOpcode, rd: usize, rs1: usize, rs2: usize) -> Instruction<F> {
    Instruction::from_usize(
        op.global_opcode(),
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
fn mulh_signed(rs1: u32, rs2: u32) -> u32 {
    let prod = (rs1 as i32 as i64) * (rs2 as i32 as i64); // have to cast in this order, to sign extend properly
    (prod >> 32) as u32
}

#[cfg(feature = "aot")]
fn mulh_signed_unsigned(rs1: u32, rs2: u32) -> u32 {
    let prod = (rs1 as i32 as i128) * (rs2 as u64 as i128);
    (prod >> 32) as u32
}

#[cfg(feature = "aot")]
fn mulh_unsigned(rs1: u32, rs2: u32) -> u32 {
    let prod = (rs1 as u64) * (rs2 as u64);
    (prod >> 32) as u32
}

#[cfg(feature = "aot")]
#[test]
fn test_aot_mulh_variants_basic() {
    let instructions = vec![
        add_immediate(4, 1234),
        add_immediate(8, 200),
        mulh_register(MulHOpcode::MULH, 12, 4, 8),
        add_immediate(16, 800),
        add_immediate(20, 12345),
        mulh_register(MulHOpcode::MULHSU, 24, 16, 20),
        add_immediate(28, 1200),
        add_immediate(32, 200),
        mulh_register(MulHOpcode::MULHU, 36, 28, 32),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let (interp_state, aot_state) = run_mul_program(instructions);

    let x3 = read_register(&interp_state, 12);
    assert_eq!(x3, mulh_signed(1234, 200));
    assert_eq!(x3, read_register(&aot_state, 12));

    let x6 = read_register(&interp_state, 24);
    assert_eq!(x6, mulh_signed_unsigned(800, 12345));
    assert_eq!(x6, read_register(&aot_state, 24));

    let x9 = read_register(&interp_state, 36);
    assert_eq!(x9, mulh_unsigned(1200, 200));
    assert_eq!(x9, read_register(&aot_state, 36));
}

#[cfg(feature = "aot")]
#[test]
fn test_aot_mulh_upper_lane() {
    let instructions = vec![
        add_immediate(4, 0x0000_000F),
        add_immediate(12, 0x0000_0002),
        mulh_register(MulHOpcode::MULH, 16, 4, 12),
        Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0),
    ];

    let (interp_state, aot_state) = run_mul_program(instructions);

    let expected = mulh_signed(0x0000_000F, 0x0000_0002);
    let interp_val = read_register(&interp_state, 16);
    let aot_val = read_register(&aot_state, 16);
    assert_eq!(interp_val, expected);
    assert_eq!(interp_val, aot_val);
}

#[cfg(feature = "aot")]
#[test]
fn test_aot_mulh_randomized() {
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
        let rs2_offset = offsets[(i + 4) % offsets.len()];
        let opcode = match i % 3 {
            0 => MulHOpcode::MULH,
            1 => MulHOpcode::MULHSU,
            _ => MulHOpcode::MULHU,
        };
        instructions.push(mulh_register(opcode, rd_offset, rs1_offset, rs2_offset));

        let rs1_val = *expected.get(&rs1_offset).unwrap();
        let rs2_val = *expected.get(&rs2_offset).unwrap();
        let result = match opcode {
            MulHOpcode::MULH => mulh_signed(rs1_val, rs2_val),
            MulHOpcode::MULHSU => mulh_signed_unsigned(rs1_val, rs2_val),
            MulHOpcode::MULHU => mulh_unsigned(rs1_val, rs2_val),
        };
        expected.insert(rd_offset, result);
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

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv32MulHExecutor, Rv32MulHAir, Rv32MulHChipGpu, Rv32MulHChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let dummy_range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        dummy_range_tuple_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32MulHChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.range_tuple_checker(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(MulHOpcode::MULH, 100)]
#[test_case(MulHOpcode::MULHSU, 100)]
#[test_case(MulHOpcode::MULHU, 100)]
fn test_cuda_rand_mulh_tracegen(opcode: MulHOpcode, num_ops: usize) {
    let mut tester = GpuChipTestBuilder::default()
        .with_bitwise_op_lookup(default_bitwise_lookup_bus())
        .with_range_tuple_checker(RangeTupleCheckerBus::new(
            RANGE_TUPLE_CHECKER_BUS,
            TUPLE_CHECKER_SIZES,
        ));
    let mut rng = create_seeded_rng();

    let mut harness = create_cuda_harness(&tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            None,
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv32MultAdapterRecord,
        &'a mut MulHCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
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
