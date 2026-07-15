use std::{borrow::BorrowMut, sync::Arc};

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
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_riscv_transpiler::MulHOpcode::{self, *};
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
use rand::rngs::StdRng;
use test_case::test_case;
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv64MultAdapterRecord, MulHCoreRecord, Rv64MulHChipGpu},
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::core::run_mulh;
use crate::{
    adapters::{
        Rv64MultAdapterAir, Rv64MultAdapterExecutor, Rv64MultAdapterFiller, RV64_BYTE_BITS,
        RV64_REGISTER_NUM_LIMBS,
    },
    mulh::{MulHCoreCols, Rv64MulHChip},
    MulHCoreAir, MulHFiller, Rv64MulHAir, Rv64MulHExecutor,
};
const MAX_INS_CAPACITY: usize = 128;
// the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
const MAX_NUM_LIMBS: u32 = 32;
const TUPLE_CHECKER_SIZES: [u32; 2] = [
    (1u32 << RV64_BYTE_BITS),
    (MAX_NUM_LIMBS * (1u32 << RV64_BYTE_BITS)),
];
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64MulHExecutor, Rv64MulHAir, Rv64MulHChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_BYTE_BITS>>,
    range_tuple_chip: Arc<RangeTupleCheckerChip<2>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64MulHAir, Rv64MulHExecutor, Rv64MulHChip<F>) {
    let air = Rv64MulHAir::new(
        Rv64MultAdapterAir::new(execution_bridge, memory_bridge),
        MulHCoreAir::new(bitwise_chip.bus(), *range_tuple_chip.bus()),
    );
    let executor = Rv64MulHExecutor::new(Rv64MultAdapterExecutor, MulHOpcode::CLASS_OFFSET);
    let chip = Rv64MulHChip::<F>::new(
        MulHFiller::new(Rv64MultAdapterFiller, bitwise_chip, range_tuple_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV64_BYTE_BITS>,
        SharedBitwiseOperationLookupChip<RV64_BYTE_BITS>,
    ),
    (RangeTupleCheckerAir<2>, SharedRangeTupleCheckerChip<2>),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
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
    b: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u32; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(generate_long_number::<
        RV64_REGISTER_NUM_LIMBS,
        RV64_BYTE_BITS,
    >(rng));
    let c = c.unwrap_or(generate_long_number::<
        RV64_REGISTER_NUM_LIMBS,
        RV64_BYTE_BITS,
    >(rng));

    let rs1 = gen_pointer(rng, 8);
    let rs2 = gen_pointer(rng, 8);
    let rd = gen_pointer(rng, 8);

    tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_u32));
    tester.write_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_u32));

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 0]),
    );

    let (a, _, _, _, _) = run_mulh::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(opcode, &b, &c);
    assert_eq!(
        a.map(F::from_u32),
        tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd)
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
fn run_rv64_mulh_rand_test(opcode: MulHOpcode, num_ops: usize) {
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
    prank_a: [u32; RV64_REGISTER_NUM_LIMBS],
    b: [u32; RV64_REGISTER_NUM_LIMBS],
    c: [u32; RV64_REGISTER_NUM_LIMBS],
    prank_a_mul: [u32; RV64_REGISTER_NUM_LIMBS],
    prank_b_ext: u32,
    prank_c_ext: u32,
    _interaction_error: bool,
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
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut MulHCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_u32);
        cols.a_mul = prank_a_mul.map(F::from_u32);
        cols.b_ext = F::from_u32(prank_b_ext);
        cols.c_ext = F::from_u32(prank_c_ext);
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .load_periphery(range_tuple)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_mulh_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULH,
        [227, 221, 103, 215, 175, 23, 236, 9],
        [197, 85, 150, 32, 145, 34, 25, 205],
        [51, 109, 78, 142, 73, 35, 25, 206],
        [63, 247, 125, 232, 87, 99, 218, 88],
        255,
        255,
        true,
    );
}

#[test]
fn rv64_mulh_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULH,
        [227, 221, 103, 215, 175, 23, 236, 10],
        [197, 85, 150, 32, 145, 34, 25, 205],
        [51, 109, 78, 142, 73, 35, 25, 206],
        [63, 247, 125, 232, 87, 99, 218, 86],
        255,
        255,
        true,
    );
}

#[test]
fn rv64_mulh_wrong_ext_negative_test() {
    run_negative_mulh_test(
        MULH,
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 128],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        0,
        0,
        true,
    );
}

#[test]
fn rv64_mulh_invalid_ext_negative_test() {
    run_negative_mulh_test(
        MULH,
        [3, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 128],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        1,
        0,
        false,
    );
}

#[test]
fn rv64_mulhsu_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [171, 1, 123, 29, 90, 68, 133, 207],
        [197, 85, 150, 160, 41, 77, 200, 195],
        [51, 109, 78, 142, 73, 35, 25, 206],
        [63, 247, 125, 104, 57, 206, 89, 35],
        255,
        0,
        true,
    );
}

#[test]
fn rv64_mulhsu_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [171, 1, 123, 29, 90, 68, 133, 206],
        [197, 85, 150, 160, 41, 77, 200, 195],
        [51, 109, 78, 142, 73, 35, 25, 206],
        [63, 247, 125, 104, 57, 206, 89, 34],
        255,
        0,
        true,
    );
}

#[test]
fn rv64_mulhsu_wrong_b_ext_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 128],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        0,
        0,
        true,
    );
}

#[test]
fn rv64_mulhsu_wrong_c_ext_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [0, 0, 0, 0, 0, 0, 0, 64],
        [0, 0, 0, 0, 0, 0, 0, 128],
        [0, 0, 0, 0, 0, 0, 0, 128],
        [0, 0, 0, 0, 0, 0, 0, 0],
        255,
        255,
        false,
    );
}

#[test]
fn rv64_mulhu_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [219, 160, 76, 134, 138, 93, 30, 165],
        [197, 85, 150, 32, 145, 34, 25, 205],
        [51, 109, 78, 142, 73, 35, 25, 206],
        [63, 247, 125, 232, 87, 99, 218, 88],
        0,
        0,
        true,
    );
}

#[test]
fn rv64_mulhu_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [219, 160, 76, 134, 138, 93, 30, 164],
        [197, 85, 150, 32, 145, 34, 25, 205],
        [51, 109, 78, 142, 73, 35, 25, 206],
        [63, 247, 125, 232, 87, 99, 218, 86],
        0,
        0,
        true,
    );
}

#[test]
fn rv64_mulhu_wrong_ext_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [255, 255, 255, 255, 255, 255, 255, 255],
        [0, 0, 0, 0, 0, 0, 0, 128],
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
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
    let x: [u32; RV64_REGISTER_NUM_LIMBS] = [197, 85, 150, 32, 145, 34, 25, 205];
    let y: [u32; RV64_REGISTER_NUM_LIMBS] = [51, 109, 78, 142, 73, 35, 25, 206];
    let z: [u32; RV64_REGISTER_NUM_LIMBS] = [227, 221, 103, 215, 175, 23, 236, 9];
    let z_mul: [u32; RV64_REGISTER_NUM_LIMBS] = [63, 247, 125, 232, 87, 99, 218, 86];
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = [490, 671, 835, 1025, 1119, 1171, 1345, 1591];
    let c_mul: [u32; RV64_REGISTER_NUM_LIMBS] = [39, 100, 126, 205, 192, 213, 155, 339];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(MULH, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV64_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 255);
    assert_eq!(y_ext, 255);
}

#[test]
fn run_mulhu_sanity_test() {
    let x: [u32; RV64_REGISTER_NUM_LIMBS] = [197, 85, 150, 32, 145, 34, 25, 205];
    let y: [u32; RV64_REGISTER_NUM_LIMBS] = [51, 109, 78, 142, 73, 35, 25, 206];
    let z: [u32; RV64_REGISTER_NUM_LIMBS] = [219, 160, 76, 134, 138, 93, 30, 165];
    let z_mul: [u32; RV64_REGISTER_NUM_LIMBS] = [63, 247, 125, 232, 87, 99, 218, 86];
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = [243, 230, 166, 182, 58, 40, 165, 0];
    let c_mul: [u32; RV64_REGISTER_NUM_LIMBS] = [39, 100, 126, 205, 192, 213, 155, 339];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(MULHU, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV64_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 0);
}

#[test]
fn run_mulhsu_pos_sanity_test() {
    let x: [u32; RV64_REGISTER_NUM_LIMBS] = [197, 85, 150, 32, 145, 34, 25, 5];
    let y: [u32; RV64_REGISTER_NUM_LIMBS] = [51, 109, 78, 142, 73, 35, 25, 206];
    let z: [u32; RV64_REGISTER_NUM_LIMBS] = [139, 91, 31, 15, 249, 185, 26, 4];
    let z_mul: [u32; RV64_REGISTER_NUM_LIMBS] = [63, 247, 125, 232, 87, 99, 218, 126];
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = [158, 169, 55, 125, 30, 20, 4, 0];
    let c_mul: [u32; RV64_REGISTER_NUM_LIMBS] = [39, 100, 126, 205, 192, 213, 155, 299];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(MULHSU, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV64_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 0);
}

#[test]
fn run_mulhsu_neg_sanity_test() {
    let x: [u32; RV64_REGISTER_NUM_LIMBS] = [197, 85, 150, 32, 145, 34, 25, 205];
    let y: [u32; RV64_REGISTER_NUM_LIMBS] = [51, 109, 78, 142, 73, 35, 25, 206];
    let z: [u32; RV64_REGISTER_NUM_LIMBS] = [168, 51, 254, 247, 64, 58, 5, 215];
    let z_mul: [u32; RV64_REGISTER_NUM_LIMBS] = [63, 247, 125, 232, 87, 99, 218, 86];
    let c: [u32; RV64_REGISTER_NUM_LIMBS] = [294, 390, 403, 561, 511, 528, 678, 718];
    let c_mul: [u32; RV64_REGISTER_NUM_LIMBS] = [39, 100, 126, 205, 192, 213, 155, 339];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(MULHSU, &x, &y);
    for i in 0..RV64_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV64_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 255);
    assert_eq!(y_ext, 0);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv64MulHExecutor, Rv64MulHAir, Rv64MulHChipGpu, Rv64MulHChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let range_tuple_bus = RangeTupleCheckerBus::new(RANGE_TUPLE_CHECKER_BUS, TUPLE_CHECKER_SIZES);

    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_BYTE_BITS>::new(
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
    let gpu_chip = Rv64MulHChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.range_tuple_checker(),
        tester.timestamp_max_bits(),
        #[cfg(feature = "rvr")]
        Default::default(),
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
        &'a mut Rv64MultAdapterRecord,
        &'a mut MulHCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
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
