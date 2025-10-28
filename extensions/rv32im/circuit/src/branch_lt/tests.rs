use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{
            memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder,
            BITWISE_OP_LOOKUP_BUS,
        },
        Arena, ExecutionBridge, PreflightExecutor,
    },
    system::memory::{offline_checker::MemoryBridge, SharedMemoryHelper},
    utils::i32_to_f,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_rv32im_transpiler::BranchLessThanOpcode;
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
    crate::{
        adapters::Rv32BranchAdapterRecord, BranchLessThanCoreRecord, Rv32BranchLessThanChipGpu,
    },
    openvm_circuit::arch::{
        testing::{default_bitwise_lookup_bus, GpuChipTestBuilder, GpuTestChipHarness},
        EmptyAdapterCoreLayout,
    },
};

use super::{run_cmp, Rv32BranchLessThanChip};
use crate::{
    adapters::{
        Rv32BranchAdapterAir, Rv32BranchAdapterExecutor, Rv32BranchAdapterFiller, RV32_CELL_BITS,
        RV32_REGISTER_NUM_LIMBS, RV_B_TYPE_IMM_BITS,
    },
    branch_lt::BranchLessThanCoreCols,
    test_utils::get_verification_error,
    BranchLessThanCoreAir, BranchLessThanFiller, Rv32BranchLessThanAir, Rv32BranchLessThanExecutor,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
const ABS_MAX_IMM: i32 = 1 << (RV_B_TYPE_IMM_BITS - 1);
type Harness = TestChipHarness<
    F,
    Rv32BranchLessThanExecutor,
    Rv32BranchLessThanAir,
    Rv32BranchLessThanChip<F>,
>;

fn create_harness_fields(
    memory_bridge: MemoryBridge,
    execution_bridge: ExecutionBridge,
    bitwise_chip: Arc<BitwiseOperationLookupChip<RV32_CELL_BITS>>,
    memory_helper: SharedMemoryHelper<F>,
) -> (
    Rv32BranchLessThanAir,
    Rv32BranchLessThanExecutor,
    Rv32BranchLessThanChip<F>,
) {
    let air = Rv32BranchLessThanAir::new(
        Rv32BranchAdapterAir::new(execution_bridge, memory_bridge),
        BranchLessThanCoreAir::new(bitwise_chip.bus(), BranchLessThanOpcode::CLASS_OFFSET),
    );
    let executor = Rv32BranchLessThanExecutor::new(
        Rv32BranchAdapterExecutor::new(),
        BranchLessThanOpcode::CLASS_OFFSET,
    );
    let chip = Rv32BranchLessThanChip::new(
        BranchLessThanFiller::new(
            Rv32BranchAdapterFiller,
            bitwise_chip,
            BranchLessThanOpcode::CLASS_OFFSET,
        ),
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
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
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
    opcode: BranchLessThanOpcode,
    a: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    b: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<i32>,
) {
    let a = a.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));
    let b = b.unwrap_or(if rng.gen_bool(0.5) {
        a
    } else {
        array::from_fn(|_| rng.gen_range(0..=u8::MAX))
    });

    let imm = imm.unwrap_or(rng.gen_range((-ABS_MAX_IMM)..ABS_MAX_IMM));
    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, a.map(F::from_canonical_u8));
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, b.map(F::from_canonical_u8));

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
        rng.gen_range(imm.unsigned_abs()..(1 << (PC_BITS - 1))),
    );

    let (cmp_result, _, _, _) =
        run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode.local_usize() as u8, &a, &b);
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
    let (mut harness, bitwise_chip) = create_harness(&mut tester);

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
        Some([101, 128, 202, 255]),
        Some([101, 128, 202, 255]),
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

    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise_chip)
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
struct BranchLessThanPrankValues<const NUM_LIMBS: usize> {
    pub a_msb: Option<i32>,
    pub b_msb: Option<i32>,
    pub diff_marker: Option<[u32; NUM_LIMBS]>,
    pub diff_val: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_branch_lt_test(
    opcode: BranchLessThanOpcode,
    a: [u8; RV32_REGISTER_NUM_LIMBS],
    b: [u8; RV32_REGISTER_NUM_LIMBS],
    prank_cmp_result: bool,
    prank_vals: BranchLessThanPrankValues<RV32_REGISTER_NUM_LIMBS>,
    interaction_error: bool,
) {
    let imm = 16i32;
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_harness(&mut tester);

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
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut BranchLessThanCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(a_msb) = prank_vals.a_msb {
            cols.a_msb_f = i32_to_f(a_msb);
        }
        if let Some(b_msb) = prank_vals.b_msb {
            cols.b_msb_f = i32_to_f(b_msb);
        }
        if let Some(diff_marker) = prank_vals.diff_marker {
            cols.diff_marker = diff_marker.map(F::from_canonical_u32);
        }
        if let Some(diff_val) = prank_vals.diff_val {
            cols.diff_val = F::from_canonical_u32(diff_val);
        }
        cols.cmp_result = F::from_bool(prank_cmp_result);
        cols.cmp_lt = F::from_bool(ge_opcode ^ prank_cmp_result);

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
fn rv32_blt_wrong_lt_cmp_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = Default::default();
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

#[test]
fn rv32_blt_wrong_ge_cmp_negative_test() {
    let a = [73, 35, 25, 205];
    let b = [145, 34, 25, 205];
    let prank_vals = Default::default();
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, false);
}

#[test]
fn rv32_blt_wrong_eq_cmp_negative_test() {
    let a = [73, 35, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = Default::default();
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, false);
}

#[test]
fn rv32_blt_fake_diff_val_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
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
fn rv32_blt_zero_diff_val_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        diff_marker: Some([0, 0, 1, 0]),
        diff_val: Some(0),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, true);
}

#[test]
fn rv32_blt_fake_diff_marker_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        diff_marker: Some([1, 0, 0, 0]),
        diff_val: Some(72),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

#[test]
fn rv32_blt_zero_diff_marker_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
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

#[test]
fn rv32_blt_signed_wrong_a_msb_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        a_msb: Some(206),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, false);
}

#[test]
fn rv32_blt_signed_wrong_a_msb_sign_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        a_msb: Some(205),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, true, prank_vals, true);
}

#[test]
fn rv32_blt_signed_wrong_b_msb_negative_test() {
    let a = [145, 36, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        b_msb: Some(206),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, false);
}

#[test]
fn rv32_blt_signed_wrong_b_msb_sign_negative_test() {
    let a = [145, 36, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        b_msb: Some(205),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLT, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGE, a, b, false, prank_vals, true);
}

#[test]
fn rv32_blt_unsigned_wrong_a_msb_negative_test() {
    let a = [145, 36, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        a_msb: Some(204),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, false);
}

#[test]
fn rv32_blt_unsigned_wrong_a_msb_sign_negative_test() {
    let a = [145, 36, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        a_msb: Some(-51),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, true, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, false, prank_vals, true);
}

#[test]
fn rv32_blt_unsigned_wrong_b_msb_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        b_msb: Some(206),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, false);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, false);
}

#[test]
fn rv32_blt_unsigned_wrong_b_msb_sign_negative_test() {
    let a = [145, 34, 25, 205];
    let b = [73, 35, 25, 205];
    let prank_vals = BranchLessThanPrankValues {
        b_msb: Some(-51),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(256),
        ..Default::default()
    };
    run_negative_branch_lt_test(BranchLessThanOpcode::BLTU, a, b, false, prank_vals, true);
    run_negative_branch_lt_test(BranchLessThanOpcode::BGEU, a, b, true, prank_vals, true);
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
    let (mut chip, _) = create_harness(&mut tester);

    let x = [145, 34, 25, 205];
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
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) = run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
        BranchLessThanOpcode::BLTU as u8,
        &x,
        &y,
    );
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned

    let (cmp_result, diff_idx, x_sign, y_sign) = run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
        BranchLessThanOpcode::BGEU as u8,
        &x,
        &y,
    );
    assert!(!cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn run_cmp_same_sign_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BranchLessThanOpcode::BLT as u8, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BranchLessThanOpcode::BGE as u8, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn run_cmp_diff_sign_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [45, 35, 25, 55];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [173, 34, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BranchLessThanOpcode::BLT as u8, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BranchLessThanOpcode::BGE as u8, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}

#[test]
fn run_cmp_eq_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [45, 35, 25, 55];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BranchLessThanOpcode::BLT as u8, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_idx, RV32_REGISTER_NUM_LIMBS);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) = run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
        BranchLessThanOpcode::BLTU as u8,
        &x,
        &x,
    );
    assert!(!cmp_result);
    assert_eq!(diff_idx, RV32_REGISTER_NUM_LIMBS);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(BranchLessThanOpcode::BGE as u8, &x, &x);
    assert!(cmp_result);
    assert_eq!(diff_idx, RV32_REGISTER_NUM_LIMBS);
    assert_eq!(x_sign, y_sign);

    let (cmp_result, diff_idx, x_sign, y_sign) = run_cmp::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
        BranchLessThanOpcode::BGEU as u8,
        &x,
        &x,
    );
    assert!(cmp_result);
    assert_eq!(diff_idx, RV32_REGISTER_NUM_LIMBS);
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
    Rv32BranchLessThanExecutor,
    Rv32BranchLessThanAir,
    Rv32BranchLessThanChipGpu,
    Rv32BranchLessThanChip<F>,
>;

#[cfg(feature = "cuda")]
fn create_cuda_harness(tester: &GpuChipTestBuilder) -> GpuHarness {
    let bitwise_bus = default_bitwise_lookup_bus();
    let dummy_bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let (air, executor, cpu_chip) = create_harness_fields(
        tester.memory_bridge(),
        tester.execution_bridge(),
        dummy_bitwise_chip,
        tester.dummy_memory_helper(),
    );
    let gpu_chip = Rv32BranchLessThanChipGpu::new(
        tester.range_checker(),
        tester.bitwise_op_lookup(),
        tester.timestamp_max_bits(),
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
#[test_case(BranchLessThanOpcode::BLT, 100)]
#[test_case(BranchLessThanOpcode::BLTU, 100)]
#[test_case(BranchLessThanOpcode::BGE, 100)]
#[test_case(BranchLessThanOpcode::BGEU, 100)]
fn test_cuda_rand_branch_lt_tracegen(opcode: BranchLessThanOpcode, num_ops: usize) {
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
        &'a mut Rv32BranchAdapterRecord,
        &'a mut BranchLessThanCoreRecord<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, Rv32BranchAdapterExecutor>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}
