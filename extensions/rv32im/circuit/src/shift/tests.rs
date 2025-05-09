use std::{array, borrow::BorrowMut};

use openvm_circuit::arch::{
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    InstructionExecutor, VmAirWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_rv32im_transpiler::ShiftOpcode::{self, *};
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
use test_case::test_case;

use super::{core::run_shift, Rv32ShiftChip, ShiftCoreAir, ShiftCoreCols, ShiftStep};
use crate::{
    adapters::{
        Rv32BaseAluAdapterAir, Rv32BaseAluAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    test_utils::{
        generate_rv32_is_type_immediate, get_verification_error, rv32_rand_write_register_or_imm,
    },
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 128;

fn create_test_chip(
    tester: &VmChipTestBuilder<F>,
) -> (
    Rv32ShiftChip<F>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let chip = Rv32ShiftChip::<F>::new(
        VmAirWrapper::new(
            Rv32BaseAluAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
            ),
            ShiftCoreAir::new(
                bitwise_bus,
                tester.range_checker().bus(),
                ShiftOpcode::CLASS_OFFSET,
            ),
        ),
        ShiftStep::new(
            Rv32BaseAluAdapterStep::new(bitwise_chip.clone()),
            bitwise_chip.clone(),
            tester.range_checker().clone(),
            ShiftOpcode::CLASS_OFFSET,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    (chip, bitwise_chip)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    rng: &mut StdRng,
    opcode: ShiftOpcode,
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
    tester.execute(chip, &instruction);

    let (a, _, _) = run_shift::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode, &b, &c);
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
#[test_case(SLL, 100)]
#[test_case(SRL, 100)]
#[test_case(SRA, 100)]
fn run_rv32_shift_rand_test(opcode: ShiftOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chip(&tester);

    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut chip, &mut rng, opcode, None, None, None);
    }

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct ShiftPrankValues<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub bit_shift: Option<u32>,
    pub bit_multiplier_left: Option<u32>,
    pub bit_multiplier_right: Option<u32>,
    pub b_sign: Option<u32>,
    pub bit_shift_marker: Option<[u32; LIMB_BITS]>,
    pub limb_shift_marker: Option<[u32; NUM_LIMBS]>,
    pub bit_shift_carry: Option<[u32; NUM_LIMBS]>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_shift_test(
    opcode: ShiftOpcode,
    prank_a: [u32; RV32_REGISTER_NUM_LIMBS],
    b: [u8; RV32_REGISTER_NUM_LIMBS],
    c: [u8; RV32_REGISTER_NUM_LIMBS],
    prank_vals: ShiftPrankValues<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        opcode,
        Some(b),
        Some(false),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&chip.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut ShiftCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        cols.a = prank_a.map(F::from_canonical_u32);
        if let Some(bit_multiplier_left) = prank_vals.bit_multiplier_left {
            cols.bit_multiplier_left = F::from_canonical_u32(bit_multiplier_left);
        }
        if let Some(bit_multiplier_right) = prank_vals.bit_multiplier_right {
            cols.bit_multiplier_right = F::from_canonical_u32(bit_multiplier_right);
        }
        if let Some(b_sign) = prank_vals.b_sign {
            cols.b_sign = F::from_canonical_u32(b_sign);
        }
        if let Some(bit_shift_marker) = prank_vals.bit_shift_marker {
            cols.bit_shift_marker = bit_shift_marker.map(F::from_canonical_u32);
        }
        if let Some(limb_shift_marker) = prank_vals.limb_shift_marker {
            cols.limb_shift_marker = limb_shift_marker.map(F::from_canonical_u32);
        }
        if let Some(bit_shift_carry) = prank_vals.bit_shift_carry {
            cols.bit_shift_carry = bit_shift_carry.map(F::from_canonical_u32);
        }

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn rv32_shift_wrong_negative_test() {
    let a = [1, 0, 0, 0];
    let b = [1, 0, 0, 0];
    let c = [1, 0, 0, 0];
    let prank_vals = Default::default();
    run_negative_shift_test(SLL, a, b, c, prank_vals, false);
    run_negative_shift_test(SRL, a, b, c, prank_vals, false);
    run_negative_shift_test(SRA, a, b, c, prank_vals, false);
}

#[test]
fn rv32_sll_wrong_bit_shift_negative_test() {
    let a = [0, 4, 4, 4];
    let b = [1, 1, 1, 1];
    let c = [9, 10, 100, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift: Some(2),
        bit_multiplier_left: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SLL, a, b, c, prank_vals, true);
}

#[test]
fn rv32_sll_wrong_limb_shift_negative_test() {
    let a = [0, 0, 2, 2];
    let b = [1, 1, 1, 1];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        limb_shift_marker: Some([0, 0, 1, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SLL, a, b, c, prank_vals, true);
}

#[test]
fn rv32_sll_wrong_bit_carry_negative_test() {
    let a = [0, 510, 510, 510];
    let b = [255, 255, 255, 255];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift_carry: Some([0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SLL, a, b, c, prank_vals, true);
}

#[test]
fn rv32_sll_wrong_bit_mult_side_negative_test() {
    let a = [128, 128, 128, 0];
    let b = [1, 1, 1, 1];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_multiplier_left: Some(0),
        bit_multiplier_right: Some(1),
        ..Default::default()
    };
    run_negative_shift_test(SLL, a, b, c, prank_vals, false);
}

#[test]
fn rv32_srl_wrong_bit_shift_negative_test() {
    let a = [0, 0, 32, 0];
    let b = [0, 0, 0, 128];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift: Some(2),
        bit_multiplier_left: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRL, a, b, c, prank_vals, false);
}

#[test]
fn rv32_srl_wrong_limb_shift_negative_test() {
    let a = [0, 64, 0, 0];
    let b = [0, 0, 0, 128];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        limb_shift_marker: Some([0, 1, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRL, a, b, c, prank_vals, false);
}

#[test]
fn rv32_srx_wrong_bit_mult_side_negative_test() {
    let a = [0, 0, 0, 0];
    let b = [0, 0, 0, 128];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_multiplier_left: Some(1),
        bit_multiplier_right: Some(0),
        ..Default::default()
    };
    run_negative_shift_test(SRL, a, b, c, prank_vals, false);
    run_negative_shift_test(SRA, a, b, c, prank_vals, false);
}

#[test]
fn rv32_sra_wrong_bit_shift_negative_test() {
    let a = [0, 0, 224, 255];
    let b = [0, 0, 0, 128];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        bit_shift: Some(2),
        bit_multiplier_left: Some(4),
        bit_shift_marker: Some([0, 0, 1, 0, 0, 0, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRA, a, b, c, prank_vals, false);
}

#[test]
fn rv32_sra_wrong_limb_shift_negative_test() {
    let a = [0, 192, 255, 255];
    let b = [0, 0, 0, 128];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        limb_shift_marker: Some([0, 1, 0, 0]),
        ..Default::default()
    };
    run_negative_shift_test(SRA, a, b, c, prank_vals, false);
}

#[test]
fn rv32_sra_wrong_sign_negative_test() {
    let a = [0, 0, 64, 0];
    let b = [0, 0, 0, 128];
    let c = [9, 0, 0, 0];
    let prank_vals = ShiftPrankValues {
        b_sign: Some(0),
        ..Default::default()
    };
    run_negative_shift_test(SRA, a, b, c, prank_vals, true);
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_sll_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [45, 7, 61, 186];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [91, 0, 100, 0];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 104];
    let (result, limb_shift, bit_shift) =
        run_shift::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(SLL, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS);
    assert_eq!(shift / RV32_CELL_BITS, limb_shift);
    assert_eq!(shift % RV32_CELL_BITS, bit_shift);
}

#[test]
fn run_srl_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [31, 190, 221, 200];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [49, 190, 190, 190];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [110, 100, 0, 0];
    let (result, limb_shift, bit_shift) =
        run_shift::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(SRL, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS);
    assert_eq!(shift / RV32_CELL_BITS, limb_shift);
    assert_eq!(shift % RV32_CELL_BITS, bit_shift);
}

#[test]
fn run_sra_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [31, 190, 221, 200];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [113, 20, 50, 80];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [110, 228, 255, 255];
    let (result, limb_shift, bit_shift) =
        run_shift::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(SRA, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
    let shift = (y[0] as usize) % (RV32_REGISTER_NUM_LIMBS * RV32_CELL_BITS);
    assert_eq!(shift / RV32_CELL_BITS, limb_shift);
    assert_eq!(shift % RV32_CELL_BITS, bit_shift);
}
