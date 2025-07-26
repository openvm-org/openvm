use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::testing::{TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    utils::i32_to_f,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_rv32im_transpiler::LessThanOpcode::{self, *};
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

use super::{core::run_less_than, LessThanCoreAir, Rv32LessThanChip};
use crate::{
    adapters::{
        Rv32BaseAluAdapterAir, Rv32BaseAluAdapterFiller, Rv32BaseAluAdapterStep, RV32_CELL_BITS,
        RV32_REGISTER_NUM_LIMBS,
    },
    less_than::LessThanCoreCols,
    test_utils::{
        generate_rv32_is_type_immediate, get_verification_error, rv32_rand_write_register_or_imm,
    },
    LessThanFiller, Rv32LessThanAir, Rv32LessThanStep,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;
type Harness = TestChipHarness<F, Rv32LessThanStep, Rv32LessThanAir, Rv32LessThanChip<F>>;

fn create_test_chip(
    tester: &VmChipTestBuilder<F>,
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
    let air = Rv32LessThanAir::new(
        Rv32BaseAluAdapterAir::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            bitwise_bus,
        ),
        LessThanCoreAir::new(bitwise_bus, LessThanOpcode::CLASS_OFFSET),
    );
    let executor = Rv32LessThanStep::new(Rv32BaseAluAdapterStep, LessThanOpcode::CLASS_OFFSET);
    let chip = Rv32LessThanChip::<F>::new(
        LessThanFiller::new(
            Rv32BaseAluAdapterFiller::new(bitwise_chip.clone()),
            bitwise_chip.clone(),
            LessThanOpcode::CLASS_OFFSET,
        ),
        tester.memory_helper(),
    );

    let harness = Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY);
    (harness, (bitwise_chip.air, bitwise_chip))
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    harness: &mut Harness,
    rng: &mut StdRng,
    opcode: LessThanOpcode,
    b: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    is_imm: Option<bool>,
    c: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));
    let (c_imm, c) = if is_imm.unwrap_or(rng.gen_bool(0.5)) {
        let (imm, c) = if let Some(c) = c {
            ((u32::from_le_bytes(c) & 0xFFFFFF) as usize, c)
        } else {
            generate_rv32_is_type_immediate(rng)
        };
        (Some(imm), c)
    } else {
        (
            None,
            c.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX))),
        )
    };

    let (instruction, rd) = rv32_rand_write_register_or_imm(
        tester,
        b,
        c,
        c_imm,
        opcode.global_opcode().as_usize(),
        rng,
    );
    tester.execute(harness, &instruction);

    let (cmp, _, _, _) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode == SLT, &b, &c);
    let mut a = [F::ZERO; RV32_REGISTER_NUM_LIMBS];
    a[0] = F::from_bool(cmp);
    assert_eq!(a, tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd));
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(SLT, 100)]
#[test_case(SLTU, 100)]
fn run_rv32_lt_rand_test(opcode: LessThanOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_chip(&tester);

    for _ in 0..num_ops {
        set_and_execute(
            &mut tester,
            &mut harness,
            &mut rng,
            opcode,
            None,
            None,
            None,
        );
    }

    // Test special case where b = c
    let b = [101, 128, 202, 255];
    set_and_execute(
        &mut tester,
        &mut harness,
        &mut rng,
        opcode,
        Some(b),
        Some(false),
        Some(b),
    );

    let b = [36, 0, 0, 0];
    set_and_execute(
        &mut tester,
        &mut harness,
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
    b: [u8; RV32_REGISTER_NUM_LIMBS],
    c: [u8; RV32_REGISTER_NUM_LIMBS],
    prank_cmp_result: bool,
    prank_vals: LessThanPrankValues<RV32_REGISTER_NUM_LIMBS>,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut harness, bitwise) = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut harness,
        &mut rng,
        opcode,
        Some(b),
        Some(false),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
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
fn rv32_lt_wrong_false_cmp_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = Default::default();
    run_negative_less_than_test(SLT, b, c, false, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, false);
}

#[test]
fn rv32_lt_wrong_true_cmp_negative_test() {
    let b = [73, 35, 25, 205];
    let c = [145, 34, 25, 205];
    let prank_vals = Default::default();
    run_negative_less_than_test(SLT, b, c, true, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, true, prank_vals, false);
}

#[test]
fn rv32_lt_wrong_eq_negative_test() {
    let b = [73, 35, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = Default::default();
    run_negative_less_than_test(SLT, b, c, true, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, true, prank_vals, false);
}

#[test]
fn rv32_lt_fake_diff_val_negative_test() {
    let b = [145, 34, 25, 205];
    let c = [73, 35, 25, 205];
    let prank_vals = LessThanPrankValues {
        diff_val: Some(F::NEG_ONE.as_canonical_u32()),
        ..Default::default()
    };
    run_negative_less_than_test(SLT, b, c, false, prank_vals, true);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, true);
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
    run_negative_less_than_test(SLT, b, c, false, prank_vals, true);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, true);
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
    run_negative_less_than_test(SLT, b, c, false, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, false);
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
    run_negative_less_than_test(SLT, b, c, false, prank_vals, false);
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, false);
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
    run_negative_less_than_test(SLT, b, c, false, prank_vals, false);
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
    run_negative_less_than_test(SLT, b, c, false, prank_vals, true);
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
    run_negative_less_than_test(SLT, b, c, true, prank_vals, false);
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
    run_negative_less_than_test(SLT, b, c, true, prank_vals, true);
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
    run_negative_less_than_test(SLTU, b, c, true, prank_vals, false);
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
    run_negative_less_than_test(SLTU, b, c, true, prank_vals, true);
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
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, false);
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
    run_negative_less_than_test(SLTU, b, c, false, prank_vals, true);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sltu_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(false, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(!x_sign); // unsigned
    assert!(!y_sign); // unsigned
}

#[test]
fn run_slt_same_sign_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [145, 34, 25, 205];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [73, 35, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    assert!(cmp_result);
    assert_eq!(diff_idx, 1);
    assert!(x_sign); // negative
    assert!(y_sign); // negative
}

#[test]
fn run_slt_diff_sign_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [45, 35, 25, 55];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [173, 34, 25, 205];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    assert!(!cmp_result);
    assert_eq!(diff_idx, 3);
    assert!(!x_sign); // positive
    assert!(y_sign); // negative
}

#[test]
fn run_less_than_equal_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [45, 35, 25, 55];
    let (cmp_result, diff_idx, x_sign, y_sign) =
        run_less_than::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &x);
    assert!(!cmp_result);
    assert_eq!(diff_idx, RV32_REGISTER_NUM_LIMBS);
    assert!(!x_sign); // positive
    assert!(!y_sign); // negative
}
