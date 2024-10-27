use std::{borrow::BorrowMut, sync::Arc};

use ax_circuit_primitives::xor::XorLookupChip;
use ax_sdk::utils::create_seeded_rng;
use ax_stark_backend::{
    utils::disable_debug_builder, verifier::VerificationError, ChipUsageGetter,
};
use axvm_instructions::instruction::Instruction;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};
use rand::{rngs::StdRng, Rng};

use super::{core::run_less_than, LessThanCoreChip, Rv32LessThanChip};
use crate::{
    arch::{
        instructions::LessThanOpcode,
        testing::{memory::gen_pointer, TestAdapterChip, VmChipTestBuilder},
        ExecutionBridge, InstructionExecutor, VmAdapterChip, VmChipWrapper, BYTE_XOR_BUS,
    },
    rv32im::{
        adapters::{Rv32BaseAluAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
        less_than::LessThanCoreCols,
    },
    utils::{generate_long_number, generate_rv32_is_type_immediate, i32_to_f},
};

type F = BabyBear;

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_rv32_lt_rand_write_execute<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    opcode: LessThanOpcode,
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    c: [u32; RV32_REGISTER_NUM_LIMBS],
    c_imm: Option<usize>,
    rng: &mut StdRng,
) {
    let is_imm = c_imm.is_some();

    let rs1 = gen_pointer(rng, 4);
    let rs2 = c_imm.unwrap_or_else(|| gen_pointer(rng, 4));
    let rd = gen_pointer(rng, 4);

    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
    if !is_imm {
        tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));
    }

    let (cmp, _, _, _) = run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode, &b, &c);
    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize,
            [rd, rs1, rs2, 1, if is_imm { 0 } else { 1 }],
        ),
    );
    let mut a = [0; RV32_REGISTER_NUM_LIMBS];
    a[0] = cmp as u32;

    assert_eq!(
        a.map(F::from_canonical_u32),
        tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd)
    );
}

fn run_rv32_lt_rand_test(opcode: LessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32LessThanChip::<F>::new(
        Rv32BaseAluAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        LessThanCoreChip::new(xor_lookup_chip.clone(), 0),
        tester.memory_controller(),
    );

    for _ in 0..num_ops {
        let b = generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);
        let (c_imm, c) = if rng.gen_bool(0.5) {
            (
                None,
                generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng),
            )
        } else {
            let (imm, c) = generate_rv32_is_type_immediate(&mut rng);
            (Some(imm), c)
        };
        run_rv32_lt_rand_write_execute(&mut tester, &mut chip, opcode, b, c, c_imm, &mut rng);
    }

    // Test special case where b = c
    run_rv32_lt_rand_write_execute(
        &mut tester,
        &mut chip,
        opcode,
        [101, 128, 202, 255],
        [101, 128, 202, 255],
        None,
        &mut rng,
    );
    run_rv32_lt_rand_write_execute(
        &mut tester,
        &mut chip,
        opcode,
        [36, 0, 0, 0],
        [36, 0, 0, 0],
        Some(36),
        &mut rng,
    );

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv32_slt_rand_test() {
    run_rv32_lt_rand_test(LessThanOpcode::SLT, 100);
}

#[test]
fn rv32_sltu_rand_test() {
    run_rv32_lt_rand_test(LessThanOpcode::SLTU, 100);
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// Given a fake trace of a single operation, setup a chip and run the test. We replace
/// the write part of the trace and check that the core chip throws the expected error.
/// A dummy adapter is used so memory interactions don't indirectly cause false passes.
///////////////////////////////////////////////////////////////////////////////////////

type Rv32LessThanTestChip<F> =
    VmChipWrapper<F, TestAdapterChip<F>, LessThanCoreChip<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>>;

#[derive(Clone, Copy, Default, PartialEq)]
struct LessThanPrankValues<const NUM_LIMBS: usize> {
    pub b_msb: Option<i32>,
    pub c_msb: Option<i32>,
    pub diff_marker: Option<[u32; NUM_LIMBS]>,
    pub diff_val: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_rv32_lt_negative_test(
    opcode: LessThanOpcode,
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    c: [u32; RV32_REGISTER_NUM_LIMBS],
    cmp_result: bool,
    prank_vals: LessThanPrankValues<RV32_REGISTER_NUM_LIMBS>,
    interaction_error: bool,
) {
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut chip = Rv32LessThanTestChip::<F>::new(
        TestAdapterChip::new(
            vec![[b.map(F::from_canonical_u32), c.map(F::from_canonical_u32)].concat()],
            vec![None],
            ExecutionBridge::new(tester.execution_bus(), tester.program_bus()),
        ),
        LessThanCoreChip::new(xor_lookup_chip.clone(), 0),
        tester.memory_controller(),
    );

    tester.execute(
        &mut chip,
        Instruction::from_usize(opcode as usize, [0, 0, 0, 1, 1]),
    );

    let trace_width = chip.trace_width();
    let adapter_width = BaseAir::<F>::width(chip.adapter.air());
    let (_, _, b_sign, c_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode, &b, &c);

    let xor_res = if prank_vals != LessThanPrankValues::default() {
        debug_assert!(prank_vals.diff_val.is_some());
        let b_msb = prank_vals.b_msb.unwrap_or(
            b[RV32_REGISTER_NUM_LIMBS - 1] as i32 - if b_sign { 1 << RV32_CELL_BITS } else { 0 },
        );
        let c_msb = prank_vals.c_msb.unwrap_or(
            c[RV32_REGISTER_NUM_LIMBS - 1] as i32 - if c_sign { 1 << RV32_CELL_BITS } else { 0 },
        );
        let xor_offset = if opcode == LessThanOpcode::SLT {
            1 << (RV32_CELL_BITS - 1)
        } else {
            0
        };
        let diff_val = prank_vals
            .diff_val
            .unwrap()
            .clamp(0, (1 << RV32_CELL_BITS) - 1);
        xor_lookup_chip.clear();
        if diff_val > 0 {
            xor_lookup_chip.request(diff_val - 1, diff_val - 1);
        }
        Some(xor_lookup_chip.request(
            (b_msb + xor_offset) as u8 as u32,
            (c_msb + xor_offset) as u8 as u32,
        ))
    } else {
        None
    };

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut LessThanCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(b_msb) = prank_vals.b_msb {
            cols.b_msb_f = i32_to_f(b_msb);
        }
        if let Some(c_msb) = prank_vals.c_msb {
            cols.c_msb_f = i32_to_f(c_msb);
        }
        if let Some(xor_res) = xor_res {
            cols.xor_res = F::from_canonical_u32(xor_res);
        }
        if let Some(diff_marker) = prank_vals.diff_marker {
            cols.diff_marker = diff_marker.map(F::from_canonical_u32);
        }
        if let Some(diff_val) = prank_vals.diff_val {
            cols.diff_val = F::from_canonical_u32(diff_val);
        }
        cols.cmp_result = F::from_bool(cmp_result);

        *trace = RowMajorMatrix::new(values, trace_width);
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(xor_lookup_chip)
        .finalize();
    tester.simple_test_with_expected_error(if interaction_error {
        VerificationError::NonZeroCumulativeSum
    } else {
        VerificationError::OodEvaluationMismatch
    });
}

#[test]
fn rv32_lt_wrong_false_cmp_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = Default::default();
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, false, prank_vals, false);
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv32_lt_wrong_true_cmp_negative_test() {
    let b = [73, 35, 25, 205];
    let c = [145, 34, 25, 205];
    let prank_vals = Default::default();
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, true, prank_vals, false);
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, true, prank_vals, false);
}

#[test]
fn rv32_lt_wrong_eq_negative_test() {
    let b = [73, 35, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = Default::default();
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, true, prank_vals, false);
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, true, prank_vals, false);
}

#[test]
fn rv32_lt_fake_diff_val_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        diff_val: Some(F::neg_one().as_canonical_u32()),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, false, prank_vals, true);
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, false, prank_vals, true);
}

#[test]
fn rv32_lt_zero_diff_val_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        diff_marker: Some([0, 0, 1, 0]),
        diff_val: Some(0),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, false, prank_vals, true);
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, false, prank_vals, true);
}

#[test]
fn rv32_lt_fake_diff_marker_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        diff_marker: Some([1, 0, 0, 0]),
        diff_val: Some(72),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, false, prank_vals, false);
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv32_lt_zero_diff_marker_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        diff_marker: Some([0, 0, 0, 0]),
        diff_val: Some(0),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, false, prank_vals, false);
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv32_slt_wrong_b_msb_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        b_msb: Some(206),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, false, prank_vals, false);
}

#[test]
fn rv32_slt_wrong_b_msb_sign_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        b_msb: Some(205),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(256),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, false, prank_vals, true);
}

#[test]
fn rv32_slt_wrong_c_msb_negative_test() {
    let b = [145, 36, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        c_msb: Some(204),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, true, prank_vals, false);
}

#[test]
fn rv32_slt_wrong_c_msb_sign_negative_test() {
    let b = [145, 36, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        c_msb: Some(205),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(256),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLT, b, c, true, prank_vals, true);
}

#[test]
fn rv32_sltu_wrong_b_msb_negative_test() {
    let b = [145, 36, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        b_msb: Some(204),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, true, prank_vals, false);
}

#[test]
fn rv32_sltu_wrong_b_msb_sign_negative_test() {
    let b = [145, 36, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        b_msb: Some(-51),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(256),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, true, prank_vals, true);
}

#[test]
fn rv32_sltu_wrong_c_msb_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        c_msb: Some(204),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(1),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv32_sltu_wrong_c_msb_sign_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        c_msb: Some(-51),
        diff_marker: Some([0, 0, 0, 1]),
        diff_val: Some(256),
        ..Default::default()
    };
    run_rv32_lt_negative_test(LessThanOpcode::SLTU, b, c, false, prank_vals, true);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sltu_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(LessThanOpcode::SLTU, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn run_slt_same_sign_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(LessThanOpcode::SLT, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn run_slt_diff_sign_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [45, 35, 25, 55];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [173, 34, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(LessThanOpcode::SLT, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}

#[test]
fn run_less_than_equal_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [45, 35, 25, 55];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(LessThanOpcode::SLT, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_idx, RV32_REGISTER_NUM_LIMBS);
    assert!(!x_sign); // positive
    assert!(!y_sign); // negative
}
