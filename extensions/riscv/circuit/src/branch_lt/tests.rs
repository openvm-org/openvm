use std::{array, borrow::BorrowMut};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::i32_to_f,
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::var_range::VariableRangeCheckerChip;
#[cfg(feature = "cuda")]
use std::sync::Arc;
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_riscv_transpiler::BranchLessThanOpcode;
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
    crate::{
        adapters::Rv64BranchAdapterRecord, BranchLessThanCoreRecord, Rv64BranchLessThanChipGpu,
    },
    openvm_circuit::arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{run_cmp, Rv64BranchLessThanChip};
use crate::{
    adapters::{
        Rv64BranchAdapterAir, Rv64BranchAdapterExecutor, Rv64BranchAdapterFiller,
        RV64_REGISTER_NUM_LIMBS, RV_B_TYPE_IMM_BITS,
    },
    branch_eq::{RV64_BRANCH_LIMB_BITS, RV64_BRANCH_NUM_LIMBS},
    branch_lt::BranchLessThanCoreCols,
    BranchLessThanCoreAir, BranchLessThanFiller, Rv64BranchLessThanAir, Rv64BranchLessThanExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_IMM: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);

/// Convert a `[u16; 4]` register value to its little-endian 8-byte representation.
#[inline]
fn u16_array_to_bytes_le(arr: &[u16; RV64_BRANCH_NUM_LIMBS]) -> [u8; RV64_REGISTER_NUM_LIMBS] {
    let mut out = [0u8; RV64_REGISTER_NUM_LIMBS];
    for (i, &v) in arr.iter().enumerate() {
        let [lo, hi] = v.to_le_bytes();
        out[2 * i] = lo;
        out[2 * i + 1] = hi;
    }
    out
}
type Harness = TestChipHarness<
    F,
    Rv64BranchLessThanExecutor,
    Rv64BranchLessThanAir,
    Rv64BranchLessThanChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    range_checker_chip: SharedVariableRangeCheckerChip,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv64BranchLessThanAir,
    Rv64BranchLessThanExecutor,
    Rv64BranchLessThanChip<F>,
) {
    let air = Rv64BranchLessThanAir::new(
        Rv64BranchAdapterAir::new(execution_bridge, memory_bridge),
        BranchLessThanCoreAir::new(range_checker_chip.bus(), BranchLessThanOpcode::CLASS_OFFSET),
    );
    let executor = Rv64BranchLessThanExecutor::new(
        Rv64BranchAdapterExecutor::new(),
        BranchLessThanOpcode::CLASS_OFFSET,
    );
    let chip = Rv64BranchLessThanChip::new(
        BranchLessThanFiller::new(
            Rv64BranchAdapterFiller,
            range_checker_chip,
            BranchLessThanOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_harness(tester: &mut VmChipTestBuilder<F>) -> Harness {
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
    opcode: BranchLessThanOpcode,
    a: Option<[u16; RV64_BRANCH_NUM_LIMBS]>,
    b: Option<[u16; RV64_BRANCH_NUM_LIMBS]>,
    imm: Option<i32>,
) {
    let a = a.unwrap_or(array::from_fn(|_| rng.random_range(0..=u16::MAX)));
    let b = b.unwrap_or(if rng.random_bool(0.5) {
        a
    } else {
        array::from_fn(|_| rng.random_range(0..=u16::MAX))
    });

    let imm = imm.unwrap_or(rng.random_range((-ABS_MAX_IMM)..ABS_MAX_IMM));
    let rs1 = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let rs2 = gen_pointer(rng, RV64_REGISTER_NUM_LIMBS);
    let a_bytes: [F; RV64_REGISTER_NUM_LIMBS] = u16_array_to_bytes_le(&a).map(F::from_u8);
    let b_bytes: [F; RV64_REGISTER_NUM_LIMBS] = u16_array_to_bytes_le(&b).map(F::from_u8);
    tester.write::<RV64_REGISTER_NUM_LIMBS>(1, rs1, a_bytes);
    tester.write::<RV64_REGISTER_NUM_LIMBS>(1, rs2, b_bytes);

    tester.execute_with_pc(
        executor,
        arena,
        &Instruction::from_isize(
            opcode.global_opcode(),
            rs1 as isize,
            rs2 as isize,
            imm as isize,
            1,
            1,
        ),
        rng.random_range(imm.unsigned_abs()..(1 << (PC_BITS - 1))),
    );

    let (cmp_result, _, _, _) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(opcode.local_usize() as u8, &a, &b);
    let from_pc = tester.last_from_pc().as_canonical_u32() as i32;
    let to_pc = tester.last_to_pc().as_canonical_u32() as i32;
    let pc_inc = if cmp_result { imm } else { 4 };

    assert_eq!(to_pc, from_pc + pc_inc);
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(BranchLessThanOpcode::BLT, 100)]
#[test_case(BranchLessThanOpcode::BLTU, 100)]
#[test_case(BranchLessThanOpcode::BGE, 100)]
#[test_case(BranchLessThanOpcode::BGEU, 100)]
fn rand_branch_lt_test(opcode: BranchLessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    // Test special case where b = c
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some([0xff01, 0xfe80, 0xcaca, 0xffff]),
        Some([0xff01, 0xfe80, 0xcaca, 0xffff]),
        Some(24),
    );
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some([36, 0, 0, 0]),
        Some([36, 0, 0, 0]),
        Some(24),
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
struct BranchLessThanPrankValues<const NUM_LIMBS: usize> {
    pub a_msb: Option<i32>,
    pub b_msb: Option<i32>,
    pub diff_marker: Option<[u32; NUM_LIMBS]>,
    pub diff_val: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_branch_lt_test(
    opcode: BranchLessThanOpcode,
    a: [u16; RV64_BRANCH_NUM_LIMBS],
    b: [u16; RV64_BRANCH_NUM_LIMBS],
    prank_cmp_result: bool,
    prank_vals: BranchLessThanPrankValues<RV64_BRANCH_NUM_LIMBS>,
    _interaction_error: bool,
) {
    let imm = 16i32;
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut harness = create_harness(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(a),
        Some(b),
        Some(imm),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let ge_opcode = opcode == BranchLessThanOpcode::BGE || opcode == BranchLessThanOpcode::BGEU;

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).unwrap().to_vec();
        let cols: &mut BranchLessThanCoreCols<F, RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(a_msb) = prank_vals.a_msb {
            cols.a_msb_f = i32_to_f(a_msb);
        }
        if let Some(b_msb) = prank_vals.b_msb {
            cols.b_msb_f = i32_to_f(b_msb);
        }
        if let Some(diff_marker) = prank_vals.diff_marker {
            cols.diff_marker = diff_marker.map(F::from_u32);
        }
        if let Some(diff_val) = prank_vals.diff_val {
            cols.diff_val = F::from_u32(diff_val);
        }
        cols.cmp_result = F::from_bool(prank_cmp_result);
        cols.cmp_lt = F::from_bool(ge_opcode ^ prank_cmp_result);

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

// Canonical relation: a < b (unsigned). The lowest u16 differs (0x4900 < 0x9100); higher limbs
// match. MSB = 0xcd05 has bit 15 set, so both are negative under signed comparison; sign matches
// → signed BLT also gives a < b.
const A_LT_B_LOWER: [u16; RV64_BRANCH_NUM_LIMBS] = [0x4900, 0x5638, 0x6459, 0xcd05];
const A_LT_B_HIGHER: [u16; RV64_BRANCH_NUM_LIMBS] = [0x9100, 0x5638, 0x6459, 0xcd05];

#[test]
fn rv64_blt_wrong_lt_cmp_negative_test() {
    let a = A_LT_B_LOWER;
    let b = A_LT_B_HIGHER;
    let prank_vals = Default::default();
    // Canonical (a<b) cmp_result is true for BLT/BLTU and false for BGE/BGEU; prank to opposite.
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

#[test]
fn rv64_blt_wrong_eq_cmp_negative_test() {
    let a = A_LT_B_LOWER;
    let b = a;
    let prank_vals = Default::default();
    // Canonical (a==b) cmp_result is false for BLT/BLTU and true for BGE/BGEU; prank to opposite.
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, false);
}

#[test]
fn rv64_blt_fake_diff_val_negative_test() {
    let a = A_LT_B_LOWER;
    let b = A_LT_B_HIGHER;
    let prank_vals = BranchLessThanPrankValues {
        diff_val: Some(F::NEG_ONE.as_canonical_u32()),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, true);
}

#[test]
fn rv64_blt_zero_diff_val_negative_test() {
    let a = A_LT_B_LOWER;
    let b = A_LT_B_HIGHER;
    let prank_vals = BranchLessThanPrankValues {
        diff_marker: Some([1, 0, 0, 0]),
        diff_val: Some(0),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, true);
}

#[test]
fn rv64_blt_zero_diff_marker_negative_test() {
    let a = A_LT_B_LOWER;
    let b = A_LT_B_HIGHER;
    let prank_vals = BranchLessThanPrankValues {
        diff_marker: Some([0, 0, 0, 0]),
        diff_val: Some(0),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn execute_roundtrip_sanity_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let mut chip = create_harness(&mut tester);

    let x = [0x9100, 0x5638, 0x6459, 0xcd05];
    set_and_execute(
        &mut tester,
        &mut chip.executor,
        &mut chip.arena,
        &mut rng,
        BranchLessThanOpcode::BLT,
        Some(x),
        Some(x),
        Some(8),
    );

    set_and_execute(
        &mut tester,
        &mut chip.executor,
        &mut chip.arena,
        &mut rng,
        BranchLessThanOpcode::BGE,
        Some(x),
        Some(x),
        Some(8),
    );
}

#[test]
fn run_cmp_unsigned_sanity_test() {
    let x: [u16; RV64_BRANCH_NUM_LIMBS] = [0x9100, 0x5638, 0x6459, 0xcd05];
    let y: [u16; RV64_BRANCH_NUM_LIMBS] = [0x4900, 0x5638, 0x6459, 0xcd05];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BLTU as u8,
            &x,
            &y,
        );
    assert!(!cmp_result);
    assert_eq!(diff_idx, 0);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BGEU as u8,
            &x,
            &y,
        );
    assert!(cmp_result);
    assert_eq!(diff_idx, 0);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn run_cmp_same_sign_sanity_test() {
    let x: [u16; RV64_BRANCH_NUM_LIMBS] = [0x9100, 0x5638, 0x6459, 0xcd05];
    let y: [u16; RV64_BRANCH_NUM_LIMBS] = [0x4900, 0x5638, 0x6459, 0xcd05];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BLT as u8,
            &x,
            &y,
        );
    assert!(!cmp_result);
    assert_eq!(diff_idx, 0);
    assert!(x_sign); // negative
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BGE as u8,
            &x,
            &y,
        );
    assert!(cmp_result);
    assert_eq!(diff_idx, 0);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn run_cmp_diff_sign_sanity_test() {
    let x: [u16; RV64_BRANCH_NUM_LIMBS] = [0x232d, 0x3719, 0x0000, 0x3700];
    let y: [u16; RV64_BRANCH_NUM_LIMBS] = [0x22ad, 0xcd19, 0xffff, 0xcd00];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BLT as u8,
            &x,
            &y,
        );
    assert!(!cmp_result);
    assert_eq!(diff_idx, RV64_BRANCH_NUM_LIMBS - 1);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BGE as u8,
            &x,
            &y,
        );
    assert!(cmp_result);
    assert_eq!(diff_idx, RV64_BRANCH_NUM_LIMBS - 1);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}

#[test]
fn run_cmp_eq_sanity_test() {
    let x: [u16; RV64_BRANCH_NUM_LIMBS] = [0x232d, 0x3719, 0x0000, 0x3700];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BLT as u8,
            &x,
            &x,
        );
    assert!(!cmp_result);
    assert_eq!(diff_idx, RV64_BRANCH_NUM_LIMBS);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BLTU as u8,
            &x,
            &x,
        );
    assert!(!cmp_result);
    assert_eq!(diff_idx, RV64_BRANCH_NUM_LIMBS);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BGE as u8,
            &x,
            &x,
        );
    assert!(cmp_result);
    assert_eq!(diff_idx, RV64_BRANCH_NUM_LIMBS);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>(
            BranchLessThanOpcode::BGEU as u8,
            &x,
            &x,
        );
    assert!(cmp_result);
    assert_eq!(diff_idx, RV64_BRANCH_NUM_LIMBS);
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
    Rv64BranchLessThanExecutor,
    Rv64BranchLessThanAir,
    Rv64BranchLessThanChipGpu,
    Rv64BranchLessThanChip<F>,
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
    let gpu_chip =
        Rv64BranchLessThanChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());
    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(BranchLessThanOpcode::BLT, 100)]
#[test_case(BranchLessThanOpcode::BLTU, 100)]
#[test_case(BranchLessThanOpcode::BGE, 100)]
#[test_case(BranchLessThanOpcode::BGEU, 100)]
fn test_cuda_rand_branch_lt_tracegen(opcode: BranchLessThanOpcode, num_ops: usize) {
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
            None,
        );
    }

    type Record<'a> = (
        &'a mut Rv64BranchAdapterRecord,
        &'a mut BranchLessThanCoreRecord<RV64_BRANCH_NUM_LIMBS, RV64_BRANCH_LIMB_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BranchAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
