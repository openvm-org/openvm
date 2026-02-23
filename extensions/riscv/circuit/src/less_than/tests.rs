use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{TestBuilder, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::i32_to_f,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LessThanOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
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
    crate::{adapters::Rv64BaseAluAdapterRecord, LessThanCoreRecord, Rv64LessThanChipGpu},
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{core::run_less_than, LessThanCoreAir, Rv64LessThanChip};
use crate::{
    adapters::{
        Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller,
        RV64_CELL_BITS, RV64_REGISTER_NUM_LIMBS,
    },
    less_than::LessThanCoreCols,
    test_utils::{
        generate_rv64_is_type_immediate, get_verification_error, rv64_rand_write_register_or_imm,
    },
    LessThanFiller, Rv64LessThanAir, Rv64LessThanExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
type Harness = TestChipHarness<F, Rv64LessThanExecutor, Rv64LessThanAir, Rv64LessThanChip<F>>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV64_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (Rv64LessThanAir, Rv64LessThanExecutor, Rv64LessThanChip<F>) {
    let air = Rv64LessThanAir::new(
        Rv64BaseAluAdapterAir::new(execution_bridge, memory_bridge, bitwise_chip.bus()),
        LessThanCoreAir::new(bitwise_chip.bus(), LessThanOpcode::CLASS_OFFSET),
    );
    let executor =
        Rv64LessThanExecutor::new(Rv64BaseAluAdapterExecutor, LessThanOpcode::CLASS_OFFSET);
    let chip = Rv64LessThanChip::<F>::new(
        LessThanFiller::new(
            Rv64BaseAluAdapterFiller::new(bitwise_chip.clone()),
            bitwise_chip,
            LessThanOpcode::CLASS_OFFSET,
        ),
        memory_helper,
    );
    (air, executor, chip)
}

fn create_test_chip(
    tester: &VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV64_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV64_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        bitwise_chip.clone(),
        tester.memory_helper(),
    );
    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);
    (harness, (bitwise_chip.air, bitwise_chip))
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<RA: Arena, E: PreflightExecutor<F, RA>>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: LessThanOpcode,
    b: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
    is_imm: Option<bool>,
    c: Option<[u8; RV64_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));
    let (c_imm, c) = if is_imm.unwrap_or(rng.gen_bool(0.5)) {
        let (imm, c) = if let Some(c) = c {
            ((u64::from_le_bytes(c) & 0xFFFFFF) as usize, c)
        } else {
            generate_rv64_is_type_immediate(rng)
        };
        (Some(imm), c)
    } else {
        (
            None,
            c.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX))),
        )
    };

    let (instruction, rd) = rv64_rand_write_register_or_imm(
        tester,
        b,
        c,
        c_imm,
        opcode.global_opcode().as_usize(),
        rng,
    );
    tester.execute(executor, arena, &instruction);

    let (cmp, _, _, _) =
        run_less_than::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(opcode == SLT, &b, &c);
    let mut a = [F::ZERO; RV64_REGISTER_NUM_LIMBS];
    a[0] = F::from_bool(cmp);
    assert_eq!(a, tester.read::<RV64_REGISTER_NUM_LIMBS>(1, rd));
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
    let (mut harness, bitwise) = create_test_chip(&tester);

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
    let b = [101, 128, 202, 255, 255, 255, 255, 255];
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(false),
        Some(b),
    );

    let b = [36, 0, 0, 0, 0, 0, 0, 0];
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(true),
        Some(b),
    );

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
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
    prank_vals: LessThanPrankValues<RV64_REGISTER_NUM_LIMBS>,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        Some(b),
        Some(false),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut LessThanCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(b_msb) = prank_vals.b_msb {
            cols.b_msb_f = i32_to_f(b_msb);
        }
        if let Some(c_msb) = prank_vals.c_msb {
            cols.c_msb_f = i32_to_f(c_msb);
        }
        if let Some(diff_marker) = prank_vals.diff_marker {
            cols.diff_marker = diff_marker.map(F::from_canonical_u32);
        }
        if let Some(diff_val) = prank_vals.diff_val {
            cols.diff_val = F::from_canonical_u32(diff_val);
        }
        cols.cmp_result = F::from_bool(prank_cmp_result);

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
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
        diff_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
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
        diff_marker: Some([1, 0, 0, 0, 0, 0, 0, 0]),
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
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 0]),
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
        b_msb: Some(206),
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 1]),
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
        b_msb: Some(205),
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 1]),
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
        c_msb: Some(204),
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 1]),
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
        c_msb: Some(205),
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 1]),
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
        b_msb: Some(204),
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 1]),
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
        b_msb: Some(-51),
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 1]),
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
        c_msb: Some(204),
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 1]),
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
        c_msb: Some(-51),
        diff_marker: Some([0, 0, 0, 0, 0, 0, 0, 1]),
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
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [145, 34, 25, 205, 255, 255, 255, 255];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [73, 35, 25, 205, 255, 255, 255, 255];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(false, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn run_slt_same_sign_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [145, 34, 25, 205, 255, 255, 255, 255];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [73, 35, 25, 205, 255, 255, 255, 255];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(true, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn run_slt_diff_sign_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [45, 35, 25, 55, 0, 0, 0, 0];
    let y: [u8; RV64_REGISTER_NUM_LIMBS] = [173, 34, 25, 205, 255, 255, 255, 255];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(true, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 7);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}

#[test]
fn run_less_than_equal_sanity_test() {
    let x: [u8; RV64_REGISTER_NUM_LIMBS] = [45, 35, 25, 55, 0, 0, 0, 0];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>(true, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_idx, RV64_REGISTER_NUM_LIMBS);
    assert!(!x_sign); // positive
    assert!(!y_sign); // negative
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
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV64_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv64LessThanChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(LessThanOpcode::SLT, 100)]
#[test_case(LessThanOpcode::SLTU, 100)]
fn test_cuda_rand_less_than_tracegen(opcode: LessThanOpcode, num_ops: usize) {
    let mut tester =
        GpuChipTestBuilder::default().with_bitwise_op_lookup(default_bitwise_lookup_bus());
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
        &'a mut Rv64BaseAluAdapterRecord,
        &'a mut LessThanCoreRecord<RV64_REGISTER_NUM_LIMBS, RV64_CELL_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv64BaseAluAdapterExecutor<RV64_CELL_BITS>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
