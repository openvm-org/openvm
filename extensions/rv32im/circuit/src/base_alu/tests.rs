use std::{array, borrow::BorrowMut};

use openvm_circuit::arch::{
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    VmAirWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::LocalOpcode;
use openvm_rv32im_transpiler::BaseAluOpcode::{self, *};
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

use super::{core::run_alu, BaseAluCoreAir, Rv32BaseAluChip, Rv32BaseAluStep};
use crate::{
    adapters::{
        Rv32BaseAluAdapterAir, Rv32BaseAluAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    base_alu::BaseAluCoreCols,
    test_utils::{
        generate_rv32_is_type_immediate, get_verification_error, rv32_rand_write_register_or_imm,
    },
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

fn create_test_chip(
    tester: &VmChipTestBuilder<F>,
) -> (
    Rv32BaseAluChip<F>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let chip = Rv32BaseAluChip::new(
        VmAirWrapper::new(
            Rv32BaseAluAdapterAir::new(
                tester.execution_bridge(),
                tester.memory_bridge(),
                bitwise_bus,
            ),
            BaseAluCoreAir::new(bitwise_bus, BaseAluOpcode::CLASS_OFFSET),
        ),
        Rv32BaseAluStep::new(
            Rv32BaseAluAdapterStep::new(bitwise_chip.clone()),
            bitwise_chip.clone(),
            BaseAluOpcode::CLASS_OFFSET,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    (chip, bitwise_chip)
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32BaseAluChip<F>,
    rng: &mut StdRng,
    opcode: BaseAluOpcode,
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

    let a = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode, &b, &c)
        .map(F::from_canonical_u8);
    assert_eq!(a, tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd))
}

//////////////////////////////////////////////////////////////////////////////////////
// POSITIVE TESTS
//
// Randomly generate computations and execute, ensuring that the generated trace
// passes all constraints.
//////////////////////////////////////////////////////////////////////////////////////

#[test_case(ADD, 100)]
#[test_case(SUB, 100)]
#[test_case(XOR, 100)]
#[test_case(OR, 100)]
#[test_case(AND, 100)]
fn rand_rv32_alu_test(opcode: BaseAluOpcode, num_ops: usize) {
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

#[allow(clippy::too_many_arguments)]
fn run_negative_alu_test(
    opcode: BaseAluOpcode,
    prank_a: [u32; RV32_REGISTER_NUM_LIMBS],
    b: [u8; RV32_REGISTER_NUM_LIMBS],
    c: [u8; RV32_REGISTER_NUM_LIMBS],
    prank_c: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    prank_opcode_flags: Option<[bool; 5]>,
    is_imm: Option<bool>,
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
        is_imm,
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&chip.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut BaseAluCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_canonical_u32);
        if let Some(prank_c) = prank_c {
            cols.c = prank_c.map(F::from_canonical_u32);
        }
        if let Some(prank_opcode_flags) = prank_opcode_flags {
            cols.opcode_add_flag = F::from_bool(prank_opcode_flags[0]);
            cols.opcode_and_flag = F::from_bool(prank_opcode_flags[1]);
            cols.opcode_or_flag = F::from_bool(prank_opcode_flags[2]);
            cols.opcode_sub_flag = F::from_bool(prank_opcode_flags[3]);
            cols.opcode_xor_flag = F::from_bool(prank_opcode_flags[4]);
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
fn rv32_alu_add_wrong_negative_test() {
    run_negative_alu_test(
        ADD,
        [246, 0, 0, 0],
        [250, 0, 0, 0],
        [250, 0, 0, 0],
        None,
        None,
        None,
        false,
    );
}

#[test]
fn rv32_alu_add_out_of_range_negative_test() {
    run_negative_alu_test(
        ADD,
        [500, 0, 0, 0],
        [250, 0, 0, 0],
        [250, 0, 0, 0],
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv32_alu_sub_wrong_negative_test() {
    run_negative_alu_test(
        SUB,
        [255, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        None,
        None,
        None,
        false,
    );
}

#[test]
fn rv32_alu_sub_out_of_range_negative_test() {
    run_negative_alu_test(
        SUB,
        [F::NEG_ONE.as_canonical_u32(), 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv32_alu_xor_wrong_negative_test() {
    run_negative_alu_test(
        XOR,
        [255, 255, 255, 255],
        [0, 0, 1, 0],
        [255, 255, 255, 255],
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv32_alu_or_wrong_negative_test() {
    run_negative_alu_test(
        OR,
        [255, 255, 255, 255],
        [255, 255, 255, 254],
        [0, 0, 0, 0],
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv32_alu_and_wrong_negative_test() {
    run_negative_alu_test(
        AND,
        [255, 255, 255, 255],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        None,
        None,
        None,
        true,
    );
}

#[test]
fn rv32_alu_adapter_unconstrained_imm_limb_test() {
    run_negative_alu_test(
        ADD,
        [255, 7, 0, 0],
        [0, 0, 0, 0],
        [255, 7, 0, 0],
        Some([511, 6, 0, 0]),
        None,
        Some(true),
        true,
    );
}

#[test]
fn rv32_alu_adapter_unconstrained_rs2_read_test() {
    run_negative_alu_test(
        ADD,
        [2, 2, 2, 2],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        None,
        Some([false, false, false, false, false]),
        Some(false),
        false,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_add_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [23, 205, 73, 49];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(ADD, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_sub_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [179, 118, 240, 172];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(SUB, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_xor_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [215, 138, 49, 173];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(XOR, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_or_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [247, 171, 61, 239];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(OR, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn run_and_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [32, 33, 12, 66];
    let result = run_alu::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(AND, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}
