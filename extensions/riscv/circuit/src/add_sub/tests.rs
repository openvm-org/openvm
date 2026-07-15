use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::BaseAluOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;
#[cfg(feature = "cuda")]
use {
    crate::{adapters::Rv64BaseAluRegU16AdapterRecord, AddSubCoreRecord, Rv64AddSubChipGpu},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
    openvm_circuit_primitives::var_range::VariableRangeCheckerChip,
    std::sync::Arc,
};

use super::{core::run_add_sub, AddSubCoreAir, Rv64AddSubChip, Rv64AddSubExecutor};
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, Rv64BaseAluRegU16AdapterAir,
        Rv64BaseAluRegU16AdapterExecutor, Rv64BaseAluRegU16AdapterFiller, RV64_REGISTER_NUM_LIMBS,
        U16_BITS,
    },
    add_sub::AddSubCoreCols,
    test_utils::rv64_rand_write_register_or_imm,
    AddSubFiller, Rv64AddSubAir,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv64AddSubExecutor, Rv64AddSubAir, Rv64AddSubChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64AddSubAir, Rv64AddSubExecutor, Rv64AddSubChip<F>) {
    let air = Rv64AddSubAir::new(
        Rv64BaseAluRegU16AdapterAir::new(execution_bridge, memory_bridge),
        AddSubCoreAir::new(range_checker_chip.bus(), BaseAluOpcode::CLASS_OFFSET),
    );
    let executor = Rv64AddSubExecutor::new(
        Rv64BaseAluRegU16AdapterExecutor,
        BaseAluOpcode::CLASS_OFFSET,
    );
    let chip = Rv64AddSubChip::new(
        AddSubFiller::new(Rv64BaseAluRegU16AdapterFiller, range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: BaseAluOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.random_range(0..=u8::MAX)));

    let (instruction, rd) =
        rv64_rand_write_register_or_imm(tester, b, c, None, opcode.global_opcode().as_usize(), rng);
    tester.execute(executor, arena, &instruction);

    let b_u16 = rv64_bytes_to_u16_block(b);
    let c_u16 = rv64_bytes_to_u16_block(c);
    let a = run_add_sub::<BLOCK_FE_WIDTH, U16_BITS>(opcode, &b_u16, &c_u16);
    let a_bytes = rv64_u16_block_to_bytes(a).map(F::from_u8);
    assert_eq!(a_bytes, tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd))
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(ADD, 100)]
#[test_case(SUB, 100)]
fn rand_rv64_add_sub_test(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

    // TODO(AG): make a more meaningful test for memory accesses
    tester.write_bytes(2, 1024, [F::ONE; 8]);
    tester.write_bytes(2, 1032, [F::ONE; 8]);
    let sm_lo: [F; 8] = tester.read_bytes(2, 1024);
    let sm_hi: [F; 8] = tester.read_bytes(2, 1032);
    assert_eq!(sm_lo, [F::ONE; 8]);
    assert_eq!(sm_hi, [F::ONE; 8]);

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

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

fn run_negative_add_sub_test(
    opcode: BaseAluOpcode,
    prank_a: [u32; BLOCK_FE_WIDTH],
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_b: Option<[u32; BLOCK_FE_WIDTH]>,
    prank_c: Option<[u32; BLOCK_FE_WIDTH]>,
    prank_opcode_flags: Option<[bool; 2]>,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_harness(&tester);

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
        let cols: &mut AddSubCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_u32);
        if let Some(prank_b) = prank_b {
            cols.b = prank_b.map(F::from_u32);
        }
        if let Some(prank_c) = prank_c {
            cols.c = prank_c.map(F::from_u32);
        }
        if let Some(prank_opcode_flags) = prank_opcode_flags {
            cols.opcode_add_flag = F::from_bool(prank_opcode_flags[0]);
            cols.opcode_sub_flag = F::from_bool(prank_opcode_flags[1]);
        }
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester
        .simple_test()
        .expect_err("Expected verification to fail, but it passed");
}

#[test]
fn rv64_add_sub_add_wrong_negative_test() {
    run_negative_add_sub_test(
        ADD,
        [499, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        [250, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
    );
}

#[test]
fn rv64_add_sub_add_out_of_range_negative_test() {
    // b[0] = c[0] = 65535; the correct result is [65534, 1, 0, 0]. Pranking
    // a = [131070, 0, 0, 0] satisfies every carry constraint (all carries 0),
    // so only the 16-bit range check on a[0] can catch it.
    run_negative_add_sub_test(
        ADD,
        [131070, 0, 0, 0],
        [255, 255, 0, 0, 0, 0, 0, 0],
        [255, 255, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
    );
}

#[test]
fn rv64_add_sub_sub_wrong_negative_test() {
    run_negative_add_sub_test(
        SUB,
        [65535, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
    );
}

#[test]
fn rv64_add_sub_sub_out_of_range_negative_test() {
    // a[0] = -1 in the field satisfies the SUB carry constraints for 1 - 2, but is
    // not a canonical u16, so only the range check on a[0] can catch it.
    run_negative_add_sub_test(
        SUB,
        [F::NEG_ONE.as_canonical_u32(), 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 0, 0, 0],
        None,
        None,
        None,
    );
}

#[test]
fn rv64_add_sub_adapter_unconstrained_rs2_read_test() {
    run_negative_add_sub_test(
        ADD,
        [514, 514, 514, 514],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        None,
        None,
        Some([false, false]),
    );
}

#[test]
fn rv64_add_sub_noncanonical_b_negative_test() {
    // Prank b[0] = 2^16 with a = [0, 1, 0, 0] as compensation: every carry constraint
    // holds (carry[0] = 1) and all a cells are canonical, so only the memory bus read
    // interaction can catch that b doesn't match the rs1 register contents. This is
    // the binding that lets the core skip range checks on b and c entirely.
    run_negative_add_sub_test(
        ADD,
        [0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        Some([1 << U16_BITS, 0, 0, 0]),
        None,
        None,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_add_sanity_test() {
    let x = rv64_bytes_to_u16_block([229, 33, 29, 111, 145, 34, 25, 205]);
    let y = rv64_bytes_to_u16_block([50, 171, 44, 194, 73, 35, 25, 206]);
    let z = rv64_bytes_to_u16_block([23, 205, 73, 49, 219, 69, 50, 155]);
    let result = run_add_sub::<BLOCK_FE_WIDTH, U16_BITS>(ADD, &x, &y);
    for i in 0..BLOCK_FE_WIDTH {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_sub_sanity_test() {
    let x = rv64_bytes_to_u16_block([229, 33, 29, 111, 145, 34, 25, 205]);
    let y = rv64_bytes_to_u16_block([50, 171, 44, 194, 73, 35, 25, 206]);
    let z = rv64_bytes_to_u16_block([179, 118, 240, 172, 71, 255, 255, 254]);
    let result = run_add_sub::<BLOCK_FE_WIDTH, U16_BITS>(SUB, &x, &y);
    for i in 0..BLOCK_FE_WIDTH {
        assert_eq!(z[i], result[i])
    }
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness =
    GpuTestChipHarness<F, Rv64AddSubExecutor, Rv64AddSubAir, Rv64AddSubChipGpu, Rv64AddSubChip<F>>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let dummy_range_checker_chip = Arc::new(VariableRangeCheckerChip::new(
        openvm_circuit::arch::testing::default_var_range_checker_bus(),
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_range_checker_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64AddSubChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(BaseAluOpcode::ADD, 100)]
#[test_case(BaseAluOpcode::SUB, 100)]
fn test_cuda_rand_add_sub_tracegen(opcode: BaseAluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();

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
        &'a mut Rv64BaseAluRegU16AdapterRecord,
        &'a mut AddSubCoreRecord<BLOCK_FE_WIDTH>,
    );

    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluRegU16AdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
