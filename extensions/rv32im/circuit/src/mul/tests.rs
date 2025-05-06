use std::{array, borrow::BorrowMut};

use openvm_circuit::arch::{
    testing::{VmChipTestBuilder, RANGE_TUPLE_CHECKER_BUS},
    InstructionExecutor, VmAirWrapper,
};
use openvm_circuit_primitives::range_tuple::{RangeTupleCheckerBus, SharedRangeTupleCheckerChip};
use openvm_instructions::LocalOpcode;
use openvm_rv32im_transpiler::MulOpcode::{self, MUL};
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

use super::core::run_mul;
use crate::{
    adapters::{Rv32MultAdapterAir, Rv32MultAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
    mul::{MultiplicationCoreCols, MultiplicationStep, Rv32MultiplicationChip},
    test_utils::{get_verification_error, rv32_rand_write_register_or_imm},
    MultiplicationCoreAir,
};

const MAX_INS_CAPACITY: usize = 128;
// the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
const MAX_NUM_LIMBS: u32 = 32;
type F = BabyBear;

fn create_test_chip(
    tester: &mut VmChipTestBuilder<F>,
) -> (Rv32MultiplicationChip<F>, SharedRangeTupleCheckerChip<2>) {
    let range_tuple_bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        [1 << RV32_CELL_BITS, MAX_NUM_LIMBS * (1 << RV32_CELL_BITS)],
    );
    let range_tuple_checker = SharedRangeTupleCheckerChip::new(range_tuple_bus);

    let chip = Rv32MultiplicationChip::<F>::new(
        VmAirWrapper::new(
            Rv32MultAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
            MultiplicationCoreAir::new(range_tuple_bus, MulOpcode::CLASS_OFFSET),
        ),
        MultiplicationStep::new(
            Rv32MultAdapterStep::new(),
            range_tuple_checker.clone(),
            MulOpcode::CLASS_OFFSET,
        ),
        MAX_INS_CAPACITY,
        tester.memory_helper(),
    );

    (chip, range_tuple_checker)
}

#[allow(clippy::too_many_arguments)]
fn set_and_execute<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    rng: &mut StdRng,
    opcode: MulOpcode,
    b: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
    c: Option<[u8; RV32_REGISTER_NUM_LIMBS]>,
) {
    let b = b.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));
    let c = c.unwrap_or(array::from_fn(|_| rng.gen_range(0..=u8::MAX)));

    let (mut instruction, rd) =
        rv32_rand_write_register_or_imm(tester, b, c, None, opcode.global_opcode().as_usize(), rng);

    instruction.e = F::ZERO;
    tester.execute(chip, &instruction);

    let (a, _) = run_mul::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&b, &c);
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

#[test]
fn run_rv32_mul_rand_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut chip, range_tuple_checker) = create_test_chip(&mut tester);
    let num_ops = 100;
    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut chip, &mut rng, MUL, None, None);
    }

    let tester = tester
        .build()
        .load(chip)
        .load(range_tuple_checker)
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
fn run_negative_mul_test(
    opcode: MulOpcode,
    prank_a: [u32; RV32_REGISTER_NUM_LIMBS],
    b: [u8; RV32_REGISTER_NUM_LIMBS],
    c: [u8; RV32_REGISTER_NUM_LIMBS],
    prank_is_valid: bool,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, range_tuple_chip) = create_test_chip(&mut tester);

    set_and_execute(&mut tester, &mut chip, &mut rng, opcode, Some(b), Some(c));

    let adapter_width = BaseAir::<F>::width(&chip.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut values = trace.row_slice(0).to_vec();
        let cols: &mut MultiplicationCoreCols<F, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS> =
            values.split_at_mut(adapter_width).1.borrow_mut();
        cols.a = prank_a.map(F::from_canonical_u32);
        cols.is_valid = F::from_bool(prank_is_valid);
        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(range_tuple_chip)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn rv32_mul_wrong_negative_test() {
    run_negative_mul_test(
        MUL,
        [63, 247, 125, 234],
        [51, 109, 78, 142],
        [197, 85, 150, 32],
        true,
        true,
    );
}

#[test]
fn rv32_mul_is_valid_false_negative_test() {
    run_negative_mul_test(
        MUL,
        [63, 247, 125, 234],
        [51, 109, 78, 142],
        [197, 85, 150, 32],
        false,
        true,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_mul_sanity_test() {
    let x: [u8; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let y: [u8; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let z: [u8; RV32_REGISTER_NUM_LIMBS] = [63, 247, 125, 232];
    let c: [u32; RV32_REGISTER_NUM_LIMBS] = [39, 100, 126, 205];
    let (result, carry) = run_mul::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(z[i], result[i]);
        assert_eq!(c[i], carry[i]);
    }
}
