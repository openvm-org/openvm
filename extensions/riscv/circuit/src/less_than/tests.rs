use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor, BLOCK_FE_WIDTH,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::i32_to_f,
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LessThanOpcode::{self, *};
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
    crate::{adapters::Rv64BaseAluRegU16AdapterRecord, LessThanCoreRecord, Rv64LessThanChipGpu},
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};
#[cfg(feature = "cuda")]
use {openvm_circuit_primitives::var_range::VariableRangeCheckerChip, std::sync::Arc};

use super::{core::run_less_than, LessThanCoreAir, Rv64LessThanChip};
use crate::{
    adapters::{
        rv64_bytes_to_u16_block, Rv64BaseAluRegU16AdapterAir, Rv64BaseAluRegU16AdapterExecutor,
        Rv64BaseAluRegU16AdapterFiller, RV64_REGISTER_NUM_LIMBS, U16_BITS,
    },
    less_than::LessThanCoreCols,
    test_utils::{
        rv64_marker_bytes_to_u16_marker, rv64_msb_byte_prank_to_u16_limb,
        rv64_rand_write_register_or_imm,
    },
    LessThanFiller, Rv64LessThanAir, Rv64LessThanExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
type Harness = TestChipHarness<F, Rv64LessThanExecutor, Rv64LessThanAir, Rv64LessThanChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64LessThanAir, Rv64LessThanExecutor, Rv64LessThanChip<F>) {
    let air = Rv64LessThanAir::new(
        Rv64BaseAluRegU16AdapterAir::new(execution_bridge, memory_bridge),
        LessThanCoreAir::new(range_checker_chip.bus(), LessThanOpcode::CLASS_OFFSET),
    );
    let executor = Rv64LessThanExecutor::new(
        Rv64BaseAluRegU16AdapterExecutor,
        LessThanOpcode::CLASS_OFFSET,
    );
    let chip = Rv64LessThanChip::<F>::new(
        LessThanFiller::new(Rv64BaseAluRegU16AdapterFiller::new(), range_checker_chip),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> Harness {
    let range_checker = tester.range_checker();
    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        range_checker,
        tester.memory_helper(),
    );
    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: LessThanOpcode,
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
    let (cmp, _, _, _) = run_less_than::<BLOCK_FE_WIDTH, U16_BITS>(opcode == SLT, &b_u16, &c_u16);
    let mut a = [F::ZERO; RV64_REGISTER_NUM_LIMBS];
    a[0] = F::from_bool(cmp);
    assert_eq!(a, tester.read_bytes::<RV64_REGISTER_NUM_LIMBS>(1, rd));
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(SLT, 100)]
#[test_case(SLTU, 100)]
fn run_rv64_lt_rand_test(opcode: LessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&tester);

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

    // Test special case where b = c
    let b = [101, 128, 202, 255, 255, 255, 255, 255];
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(b),
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct LessThanPrankValues<const NUM_LIMBS: usize> {
    pub b_msb: Option<i32>,
    pub c_msb: Option<i32>,
    pub diff_marker: Option<[u32; NUM_LIMBS]>,
    pub diff_val: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_less_than_test(
    opcode: LessThanOpcode,
    b: [u8; RV64_REGISTER_NUM_LIMBS],
    c: [u8; RV64_REGISTER_NUM_LIMBS],
    prank_cmp_result: bool,
    prank_vals: LessThanPrankValues<BLOCK_FE_WIDTH>,
    _interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut harness = create_test_chip(&tester);

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
        let cols: &mut LessThanCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(b_msb) = prank_vals.b_msb {
            cols.b_msb_f = i32_to_f(b_msb);
        }
        if let Some(c_msb) = prank_vals.c_msb {
            cols.c_msb_f = i32_to_f(c_msb);
        }
        if let Some(diff_marker) = prank_vals.diff_marker {
            cols.diff_marker = diff_marker.map(F::from_u32);
        }
        if let Some(diff_val) = prank_vals.diff_val {
            cols.diff_val = F::from_u32(diff_val);
        }
        cols.cmp_result = F::from_bool(prank_cmp_result);

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
fn rv64_lt_wrong_false_cmp_negative_test() {
    let b = [145, 34, 25, 205, 255, 255, 255, 255];
    let c = [73, 35, 25, 205, 255, 255, 255, 255];
    let prank_vals = Default::default();
    run_negative_less_than_test(SLT, b, c, false, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv64_lt_wrong_true_cmp_negative_test() {
    let b = [73, 35, 25, 205, 255, 255, 255, 255];
    let c = [145, 34, 25, 205, 255, 255, 255, 255];
    let prank_vals = Default::default();
    run_negative_less_than_test(SLT, b, c, true, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, true, prank_vals, false);
}

#[test]
fn rv64_lt_wrong_eq_negative_test() {
    let b = [73, 35, 25, 205, 255, 255, 255, 255];
    let c = [73, 35, 25, 205, 255, 255, 255, 255];
    let prank_vals = Default::default();
    run_negative_less_than_test(SLT, b, c, true, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, true, prank_vals, false);
}

#[test]
fn rv64_lt_fake_diff_val_negative_test() {
    let b = [145, 34, 25, 205, 255, 255, 255, 255];
    let c = [73, 35, 25, 205, 255, 255, 255, 255];
    let prank_vals = LessThanPrankValues {
        diff_val: Some(F::NEG_ONE.as_canonical_u32()),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, false, prank_vals, true);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, true);
}

#[test]
fn rv64_lt_zero_diff_val_negative_test() {
    let b = [145, 34, 25, 205, 255, 255, 255, 255];
    let c = [73, 35, 25, 205, 255, 255, 255, 255];
    let prank_vals = LessThanPrankValues {
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 1, 0, 0, 0, 0, 0])),
        diff_val: Some(0),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, false, prank_vals, true);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, true);
}

#[test]
fn rv64_lt_fake_diff_marker_negative_test() {
    let b = [145, 34, 25, 205, 255, 255, 255, 255];
    let c = [73, 35, 25, 205, 255, 255, 255, 255];
    let prank_vals = LessThanPrankValues {
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([1, 0, 0, 0, 0, 0, 0, 0])),
        diff_val: Some(72),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, false, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv64_lt_zero_diff_marker_negative_test() {
    let b = [145, 34, 25, 205, 255, 255, 255, 255];
    let c = [73, 35, 25, 205, 255, 255, 255, 255];
    let prank_vals = LessThanPrankValues {
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 0])),
        diff_val: Some(0),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, false, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv64_slt_wrong_b_msb_negative_test() {
    // b[7]=c[7]=205, actual diff at byte 1. Prank b_msb to 206 → b_diff constraint fails.
    let b = [145, 34, 25, 0, 0, 0, 0, 205];
    let c = [73, 35, 25, 0, 0, 0, 0, 205];
    let prank_vals = LessThanPrankValues {
        b_msb: Some(rv64_msb_byte_prank_to_u16_limb(b, 206)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, false, prank_vals, false);
}

#[test]
fn rv64_slt_wrong_b_msb_sign_negative_test() {
    // b[7]=c[7]=205 (negative). Prank b_msb_f to 205 (raw byte instead of 205-256=-51).
    // b_diff=0 so constraint passes, but range check sends 205+128=333 → interaction error.
    let b = [145, 34, 25, 0, 0, 0, 0, 205];
    let c = [73, 35, 25, 0, 0, 0, 0, 205];
    let prank_vals = LessThanPrankValues {
        b_msb: Some(rv64_msb_byte_prank_to_u16_limb(b, 205)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, false, prank_vals, true);
}

#[test]
fn rv64_slt_wrong_c_msb_negative_test() {
    // b[7]=c[7]=205, actual diff at byte 1. Prank c_msb to 204 → c_diff constraint fails.
    let b = [145, 36, 25, 0, 0, 0, 0, 205];
    let c = [73, 35, 25, 0, 0, 0, 0, 205];
    let prank_vals = LessThanPrankValues {
        c_msb: Some(rv64_msb_byte_prank_to_u16_limb(c, 204)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, true, prank_vals, false);
}

#[test]
fn rv64_slt_wrong_c_msb_sign_negative_test() {
    // c[7]=205 (negative). Prank c_msb_f to 205 (raw byte instead of -51).
    // c_diff=0 so constraint passes, but range check sends 205+128=333 → interaction error.
    let b = [145, 36, 25, 0, 0, 0, 0, 205];
    let c = [73, 35, 25, 0, 0, 0, 0, 205];
    let prank_vals = LessThanPrankValues {
        c_msb: Some(rv64_msb_byte_prank_to_u16_limb(c, 205)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, true, prank_vals, true);
}

#[test]
fn rv64_sltu_wrong_b_msb_negative_test() {
    // b[7]=c[7]=205. Prank b_msb to 204 → b_diff constraint fails.
    let b = [145, 36, 25, 0, 0, 0, 0, 205];
    let c = [73, 35, 25, 0, 0, 0, 0, 205];
    let prank_vals = LessThanPrankValues {
        b_msb: Some(rv64_msb_byte_prank_to_u16_limb(b, 204)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_less_than_test(SLTU, b, c, true, prank_vals, false);
}

#[test]
fn rv64_sltu_wrong_b_msb_sign_negative_test() {
    // b[7]=205. Prank b_msb_f to -51 (=205-256). b_diff=205-(-51)=256, 256*(256-256)=0
    // so constraint passes, but range check sends -51 which is out of range → interaction error.
    let b = [145, 36, 25, 0, 0, 0, 0, 205];
    let c = [73, 35, 25, 0, 0, 0, 0, 205];
    let prank_vals = LessThanPrankValues {
        b_msb: Some(rv64_msb_byte_prank_to_u16_limb(b, -51)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_less_than_test(SLTU, b, c, true, prank_vals, true);
}

#[test]
fn rv64_sltu_wrong_c_msb_negative_test() {
    // c[7]=205. Prank c_msb to 204 → c_diff constraint fails.
    let b = [145, 34, 25, 0, 0, 0, 0, 205];
    let c = [73, 35, 25, 0, 0, 0, 0, 205];
    let prank_vals = LessThanPrankValues {
        c_msb: Some(rv64_msb_byte_prank_to_u16_limb(c, 204)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv64_sltu_wrong_c_msb_sign_negative_test() {
    // c[7]=205. Prank c_msb_f to -51 (=205-256). c_diff=205-(-51)=256, 256*(256-256)=0
    // so constraint passes, but range check sends -51 which is out of range → interaction error.
    let b = [145, 34, 25, 0, 0, 0, 0, 205];
    let c = [73, 35, 25, 0, 0, 0, 0, 205];
    let prank_vals = LessThanPrankValues {
        c_msb: Some(rv64_msb_byte_prank_to_u16_limb(c, -51)),
        diff_marker: Some(rv64_marker_bytes_to_u16_marker([0, 0, 0, 0, 0, 0, 0, 1])),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, true);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sltu_sanity_test() {
    let x = rv64_bytes_to_u16_block([145, 34, 25, 205, 91, 77, 88, 120]);
    let y = rv64_bytes_to_u16_block([73, 35, 25, 205, 91, 77, 88, 120]);
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<BLOCK_FE_WIDTH, U16_BITS>(false, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 0);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn run_slt_same_sign_sanity_test() {
    let x = rv64_bytes_to_u16_block([145, 34, 25, 205, 91, 77, 88, 205]);
    let y = rv64_bytes_to_u16_block([73, 35, 25, 205, 91, 77, 88, 205]);
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<BLOCK_FE_WIDTH, U16_BITS>(true, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 0);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn run_slt_diff_sign_sanity_test() {
    let x = rv64_bytes_to_u16_block([45, 35, 25, 55, 61, 90, 77, 74]);
    let y = rv64_bytes_to_u16_block([173, 34, 25, 205, 61, 90, 77, 182]);
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<BLOCK_FE_WIDTH, U16_BITS>(true, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, BLOCK_FE_WIDTH - 1);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}

#[test]
fn run_less_than_equal_sanity_test() {
    let x = rv64_bytes_to_u16_block([45, 35, 25, 55, 61, 90, 77, 74]);
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<BLOCK_FE_WIDTH, U16_BITS>(true, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_idx, BLOCK_FE_WIDTH);
    assert_eq!(x_sign, y_sign);
}

// ////////////////////////////////////////////////////////////////////////////////////
//  CUDA TESTS
//
//  Ensure GPU tracegen is equivalent to CPU tracegen
// ////////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    Rv64LessThanExecutor,
    Rv64LessThanAir,
    Rv64LessThanChipGpu,
    Rv64LessThanChip<F>,
>;

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
    let gpu_chip = Rv64LessThanChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LessThanOpcode::SLT, 100)]
#[test_case(LessThanOpcode::SLTU, 100)]
fn test_cuda_rand_less_than_tracegen(opcode: LessThanOpcode, num_ops: usize) {
    let mut tester = GpuChipTestBuilder::default();
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
        &'a mut Rv64BaseAluRegU16AdapterRecord,
        &'a mut LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>,
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
