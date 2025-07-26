use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::testing::{
        memory::gen_pointer, TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
        RANGE_TUPLE_CHECKER_BUS,
    },
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
use openvm_rv32im_transpiler::MulHOpcode::{self, *};
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
use rand::rngs::StdRng;
use test_case::test_case;

use super::core::run_mulh;
use crate::{
    adapters::{
        Rv32MultAdapterAir, Rv32MultAdapterFiller, Rv32MultAdapterStep, RV32_CELL_BITS,
        RV32_REGISTER_NUM_LIMBS,
    },
    mulh::{MulHCoreCols, Rv32MulHChip},
    test_utils::get_verification_error,
    MulHCoreAir, MulHFiller, Rv32MulHAir, Rv32MulHStep,
};

const MAX_INS_CAPACITY: usize = 128;
// the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
const MAX_NUM_LIMBS: u32 = 32;
type F = BabyBear;
type Harness = TestChipHarness<F, Rv32MulHStep, Rv32MulHAir, Rv32MulHChip<F>>;

fn create_test_chip(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Harness,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
    (RangeTupleCheckerAir<2>, SharedRangeTupleCheckerChip<2>),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let range_tuple_bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        [1 << RV32_CELL_BITS, MAX_NUM_LIMBS * (1 << RV32_CELL_BITS)],
    );

    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));
    let range_tuple_chip =
        SharedRangeTupleCheckerChip::new(RangeTupleCheckerChip::<2>::new(range_tuple_bus));

    let air = Rv32MulHAir::new(
        Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        MulHCoreAir::new(bitwise_bus, range_tuple_bus),
    );
    let executor = Rv32MulHStep::new(Rv32MultAdapterStep, MulHOpcode::CLASS_OFFSET);
    let chip = Rv32MulHChip::<F>::new(
        MulHFiller::new(
            Rv32MultAdapterFiller,
            bitwise_chip.clone(),
            range_tuple_chip.clone(),
        ),
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
fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    harness: &mut Harness,
    rng: &mut StdRng,
    opcode: MulHOpcode,
    b: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    c: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(generate_long_number::<
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >(rng));
    let c = c.unwrap_or(generate_long_number::<
        RV32_REGISTER_NUM_LIMBS,
        RV32_CELL_BITS,
    >(rng));

    let rs1 = gen_pointer(rng, 4);
    let rs2 = gen_pointer(rng, 4);
    let rd = gen_pointer(rng, 4);

    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));

    tester.execute(
        harness,
        &Instruction::from_usize(opcode.global_opcode(), [rd, rs1, rs2, 1, 0]),
    );

    let (a, _, _, _, _) = run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(opcode, &b, &c);
    assert_eq!(
        a.map(F::from_canonical_u32),
        tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd)
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
fn run_rv32_mulh_rand_test(opcode: MulHOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_test_chip(&mut tester);

    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut harness, &mut rng, opcode, None, None);
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
    prank_a: [u32; RV32_REGISTER_NUM_LIMBS],
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    c: [u32; RV32_REGISTER_NUM_LIMBS],
    prank_a_mul: [u32; RV32_REGISTER_NUM_LIMBS],
    prank_b_ext: u32,
    prank_c_ext: u32,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut harness, bitwise, range_tuple) = create_test_chip(&mut tester);

    set_and_execute(
        &mut tester,
        &mut harness,
        &mut rng,
        opcode,
        Some(b),
        Some(c),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut MulHCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_canonical_u32);
        cols.a_mul = prank_a_mul.map(F::from_canonical_u32);
        cols.b_ext = F::from_canonical_u32(prank_b_ext);
        cols.c_ext = F::from_canonical_u32(prank_c_ext);
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
        .load_periphery(range_tuple)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn rv32_mulh_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULH,
        [130, 9, 135, 241],
        [197, 85, 150, 32],
        [51, 109, 78, 142],
        [63, 247, 125, 234],
        0,
        255,
        true,
    );
}

#[test]
fn rv32_mulh_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULH,
        [130, 9, 135, 242],
        [197, 85, 150, 32],
        [51, 109, 78, 142],
        [63, 247, 125, 232],
        0,
        255,
        true,
    );
}

#[test]
fn rv32_mulh_wrong_ext_negative_test() {
    run_negative_mulh_test(
        MULH,
        [1, 0, 0, 0],
        [0, 0, 0, 128],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        0,
        0,
        true,
    );
}

#[test]
fn rv32_mulh_invalid_ext_negative_test() {
    run_negative_mulh_test(
        MULH,
        [3, 2, 2, 2],
        [0, 0, 0, 128],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        1,
        0,
        false,
    );
}

#[test]
fn rv32_mulhsu_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [174, 40, 246, 202],
        [197, 85, 150, 160],
        [51, 109, 78, 142],
        [63, 247, 125, 105],
        255,
        0,
        true,
    );
}

#[test]
fn rv32_mulhsu_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [174, 40, 246, 201],
        [197, 85, 150, 160],
        [51, 109, 78, 142],
        [63, 247, 125, 104],
        255,
        0,
        true,
    );
}

#[test]
fn rv32_mulhsu_wrong_b_ext_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [1, 0, 0, 0],
        [0, 0, 0, 128],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
        0,
        0,
        true,
    );
}

#[test]
fn rv32_mulhsu_wrong_c_ext_negative_test() {
    run_negative_mulh_test(
        MULHSU,
        [0, 0, 0, 64],
        [0, 0, 0, 128],
        [0, 0, 0, 128],
        [0, 0, 0, 0],
        255,
        255,
        false,
    );
}

#[test]
fn rv32_mulhu_wrong_a_mul_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [130, 9, 135, 241],
        [197, 85, 150, 32],
        [51, 109, 78, 142],
        [63, 247, 125, 234],
        0,
        0,
        true,
    );
}

#[test]
fn rv32_mulhu_wrong_a_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [130, 9, 135, 240],
        [197, 85, 150, 32],
        [51, 109, 78, 142],
        [63, 247, 125, 232],
        0,
        0,
        true,
    );
}

#[test]
fn rv32_mulhu_wrong_ext_negative_test() {
    run_negative_mulh_test(
        MULHU,
        [255, 255, 255, 255],
        [0, 0, 0, 128],
        [2, 0, 0, 0],
        [0, 0, 0, 0],
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
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [130, 9, 135, 241];
    let z_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [303, 375, 449, 463];
    let c_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 205];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(MULH, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV32_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 255);
}

#[test]
fn run_mulhu_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [71, 95, 29, 18];
    let z_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [107, 93, 18, 0];
    let c_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 205];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(MULHU, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV32_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 0);
}

#[test]
fn run_mulhsu_pos_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [71, 95, 29, 18];
    let z_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [107, 93, 18, 0];
    let c_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 205];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(MULHSU, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV32_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 0);
    assert_eq!(y_ext, 0);
}

#[test]
fn run_mulhsu_neg_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 160];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u32; RV32_REGISTER_NUM_LIMBS] = [174, 40, 246, 202];
    let z_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 104];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [212, 292, 326, 379];
    let c_mul: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 231];
    let (res, res_mul, carry, x_ext, y_ext) =
        run_mulh::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(MULHSU, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], res[i]);
        assert_eq!(z_mul[i], res_mul[i]);
        assert_eq!(c[i], carry[i + RV32_REGISTER_NUM_LIMBS]);
        assert_eq!(c_mul[i], carry[i]);
    }
    assert_eq!(x_ext, 255);
    assert_eq!(y_ext, 0);
}
