use std::{array, borrow::BorrowMut, iter, sync::Arc};

use afs_primitives::xor::lookup::XorLookupChip;
use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{
    columns::ArithmeticLogicCols, solve_subtract, ArithmeticLogicChip, ALU_CMP_INSTRUCTIONS,
};
use crate::{
    alu::solve_alu,
    arch::{
        instructions::U256Opcode,
        testing::{memory::gen_pointer, MachineChipTestBuilder},
        MachineChip,
    },
    core::BYTE_XOR_BUS,
    program::Instruction,
};

type F = BabyBear;

const NUM_LIMBS: usize = 32;
const LIMB_BITS: usize = 8;

fn generate_long_number<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rng: &mut StdRng,
) -> Vec<u32> {
    (0..NUM_LIMBS)
        .map(|_| rng.gen_range(0..1 << LIMB_BITS))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_alu_rand_write_execute<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    tester: &mut MachineChipTestBuilder<F>,
    chip: &mut ArithmeticLogicChip<F, NUM_LIMBS, LIMB_BITS>,
    opcode: U256Opcode,
    x: Vec<u32>,
    y: Vec<u32>,
    rng: &mut StdRng,
) {
    let address_space_range = || 1usize..=2;

    let d = rng.gen_range(address_space_range());
    let e = rng.gen_range(address_space_range());

    let x_address = gen_pointer(rng, 32);
    let y_address = gen_pointer(rng, 32);
    let res_address = gen_pointer(rng, 32);
    let x_ptr_to_address = gen_pointer(rng, 1);
    let y_ptr_to_address = gen_pointer(rng, 1);
    let res_ptr_to_address = gen_pointer(rng, 1);

    let x_f = x
        .clone()
        .into_iter()
        .map(F::from_canonical_u32)
        .collect::<Vec<_>>();
    let y_f = y
        .clone()
        .into_iter()
        .map(F::from_canonical_u32)
        .collect::<Vec<_>>();

    tester.write_cell(d, x_ptr_to_address, F::from_canonical_usize(x_address));
    tester.write_cell(d, y_ptr_to_address, F::from_canonical_usize(y_address));
    tester.write_cell(d, res_ptr_to_address, F::from_canonical_usize(res_address));
    tester.write::<NUM_LIMBS>(e, x_address, x_f.as_slice().try_into().unwrap());
    tester.write::<NUM_LIMBS>(e, y_address, y_f.as_slice().try_into().unwrap());

    let (z, cmp) = solve_alu::<F, NUM_LIMBS, LIMB_BITS>(opcode, &x, &y);
    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize,
            [res_ptr_to_address, x_ptr_to_address, y_ptr_to_address, d, e],
        ),
    );

    if ALU_CMP_INSTRUCTIONS.contains(&opcode) {
        assert_eq!([F::from_bool(cmp)], tester.read::<1>(e, res_address))
    } else {
        assert_eq!(
            z.into_iter().map(F::from_canonical_u32).collect::<Vec<_>>(),
            tester.read::<NUM_LIMBS>(e, res_address)
        )
    }
}

/// Given a fake trace of a single operation, setup a chip and run the test.
/// We replace the "output" part of the trace, and we _may_ replace the interactions
/// based on the desired output. We check that it produces the error we expect.
#[allow(clippy::too_many_arguments)]
fn run_alu_negative_test(
    opcode: U256Opcode,
    x: Vec<u32>,
    y: Vec<u32>,
    z: Vec<u32>,
    cmp_result: bool,
    x_sign: u32,
    y_sign: u32,
    expected_error: VerificationError,
) {
    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester: MachineChipTestBuilder<BabyBear> = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    let mut rng = create_seeded_rng();
    run_alu_rand_write_execute(
        &mut tester,
        &mut chip,
        opcode,
        x.clone(),
        y.clone(),
        &mut rng,
    );

    let alu_trace = chip.clone().generate_trace();
    let mut alu_trace_row = alu_trace.row_slice(0).to_vec();
    let alu_trace_cols: &mut ArithmeticLogicCols<F, 32, 8> = (*alu_trace_row).borrow_mut();

    alu_trace_cols.io.z.data = array::from_fn(|i| F::from_canonical_u32(z[i]));
    alu_trace_cols.io.cmp_result = F::from_bool(cmp_result);
    alu_trace_cols.aux.x_sign = F::from_canonical_u32(x_sign);
    alu_trace_cols.aux.y_sign = F::from_canonical_u32(y_sign);
    let alu_trace: p3_matrix::dense::DenseMatrix<_> = RowMajorMatrix::new(
        alu_trace_row,
        ArithmeticLogicCols::<F, NUM_LIMBS, LIMB_BITS>::width(),
    );

    disable_debug_builder();
    let tester = tester
        .build()
        .load_with_custom_trace(chip, alu_trace)
        .load(xor_lookup_chip)
        .finalize();
    let msg = format!(
        "Expected verification to fail with {:?}, but it didn't",
        &expected_error
    );
    let result = tester.simple_test();
    assert_eq!(result.err(), Some(expected_error), "{}", msg);
}

#[test]
fn alu_add_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_alu_rand_write_execute(&mut tester, &mut chip, U256Opcode::ADD, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_add_out_of_range_negative_test() {
    run_alu_negative_test(
        U256Opcode::ADD,
        iter::once(250)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        iter::once(250)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        iter::once(500)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        0,
        0,
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn alu_add_wrong_negative_test() {
    run_alu_negative_test(
        U256Opcode::ADD,
        iter::once(250)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        iter::once(250)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        iter::once(500 - (1 << 8))
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        0,
        0,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_sub_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_alu_rand_write_execute(&mut tester, &mut chip, U256Opcode::SUB, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_sub_out_of_range_negative_test() {
    run_alu_negative_test(
        U256Opcode::SUB,
        iter::once(1)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        iter::once(F::neg_one().as_canonical_u32())
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        0,
        0,
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn alu_sub_wrong_negative_test() {
    run_alu_negative_test(
        U256Opcode::SUB,
        iter::once(1)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        iter::once((1 << 8) - 1)
            .chain(iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        0,
        0,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_sltu_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_alu_rand_write_execute(&mut tester, &mut chip, U256Opcode::LT, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_sltu_wrong_subtraction_test() {
    run_alu_negative_test(
        U256Opcode::LT,
        iter::once(65_000).chain(iter::repeat(0).take(31)).collect(),
        iter::once(65_000).chain(iter::repeat(0).take(31)).collect(),
        std::iter::once(1).chain(iter::repeat(0).take(31)).collect(),
        false,
        0,
        0,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_sltu_wrong_negative_test() {
    run_alu_negative_test(
        U256Opcode::LT,
        iter::once(1).chain(iter::repeat(0).take(31)).collect(),
        iter::once(1).chain(iter::repeat(0).take(31)).collect(),
        iter::once(0).chain(iter::repeat(0).take(31)).collect(),
        true,
        0,
        0,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_sltu_non_zero_sign_negative_test() {
    run_alu_negative_test(
        U256Opcode::LT,
        vec![(1 << LIMB_BITS) - 1; NUM_LIMBS],
        vec![(1 << LIMB_BITS) - 1; NUM_LIMBS],
        vec![0; NUM_LIMBS],
        false,
        1,
        1,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_eq_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_alu_rand_write_execute(&mut tester, &mut chip, U256Opcode::EQ, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_eq_wrong_negative_test() {
    run_alu_negative_test(
        U256Opcode::EQ,
        vec![0; 31].into_iter().chain(iter::once(123)).collect(),
        vec![0; 31].into_iter().chain(iter::once(456)).collect(),
        vec![0; 31].into_iter().chain(iter::once(0)).collect(),
        true,
        0,
        0,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_xor_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_alu_rand_write_execute(&mut tester, &mut chip, U256Opcode::XOR, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_xor_wrong_negative_test() {
    run_alu_negative_test(
        U256Opcode::XOR,
        vec![0; 31].into_iter().chain(iter::once(1)).collect(),
        vec![(1 << LIMB_BITS) - 1; NUM_LIMBS],
        vec![(1 << LIMB_BITS) - 1; NUM_LIMBS],
        true,
        0,
        0,
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn alu_and_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_alu_rand_write_execute(&mut tester, &mut chip, U256Opcode::AND, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_and_wrong_negative_test() {
    run_alu_negative_test(
        U256Opcode::AND,
        vec![0; 31].into_iter().chain(iter::once(1)).collect(),
        vec![(1 << LIMB_BITS) - 1; NUM_LIMBS],
        vec![0; NUM_LIMBS],
        true,
        0,
        0,
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn alu_or_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_alu_rand_write_execute(&mut tester, &mut chip, U256Opcode::OR, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_or_wrong_negative_test() {
    run_alu_negative_test(
        U256Opcode::OR,
        vec![0; NUM_LIMBS],
        vec![(1 << LIMB_BITS) - 1; NUM_LIMBS - 1]
            .into_iter()
            .chain(iter::once((1 << LIMB_BITS) - 2))
            .collect(),
        vec![(1 << LIMB_BITS) - 1; NUM_LIMBS],
        true,
        0,
        0,
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn alu_slt_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = MachineChipTestBuilder::default();
    let mut chip = ArithmeticLogicChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_chip(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_alu_rand_write_execute(&mut tester, &mut chip, U256Opcode::SLT, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn alu_slt_pos_neg_sign_negative_test() {
    let x = [0; NUM_LIMBS];
    let y = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    run_alu_negative_test(
        U256Opcode::SLT,
        x.to_vec(),
        y.to_vec(),
        solve_subtract::<NUM_LIMBS, LIMB_BITS>(&x, &y).0,
        true,
        0,
        1,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_slt_neg_pos_sign_negative_test() {
    let x = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    let y = [0; NUM_LIMBS];
    run_alu_negative_test(
        U256Opcode::SLT,
        x.to_vec(),
        y.to_vec(),
        solve_subtract::<NUM_LIMBS, LIMB_BITS>(&x, &y).0,
        false,
        1,
        0,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_slt_both_pos_sign_negative_test() {
    let x = [0; NUM_LIMBS];
    let mut y = [0; NUM_LIMBS];
    y[0] = 1;
    run_alu_negative_test(
        U256Opcode::SLT,
        x.to_vec(),
        y.to_vec(),
        solve_subtract::<NUM_LIMBS, LIMB_BITS>(&x, &y).0,
        false,
        0,
        0,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_slt_both_neg_sign_negative_test() {
    let x = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    let mut y = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    y[0] = 1;
    run_alu_negative_test(
        U256Opcode::SLT,
        x.to_vec(),
        y.to_vec(),
        solve_subtract::<NUM_LIMBS, LIMB_BITS>(&x, &y).0,
        true,
        1,
        1,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_slt_wrong_sign_negative_test() {
    let x = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    let mut y = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    y[0] = 1;
    run_alu_negative_test(
        U256Opcode::SLT,
        x.to_vec(),
        y.to_vec(),
        solve_subtract::<NUM_LIMBS, LIMB_BITS>(&x, &y).0,
        true,
        0,
        1,
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn alu_slt_non_boolean_sign_negative_test() {
    let x = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    let mut y = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    y[0] = 1;
    run_alu_negative_test(
        U256Opcode::SLT,
        x.to_vec(),
        y.to_vec(),
        solve_subtract::<NUM_LIMBS, LIMB_BITS>(&x, &y).0,
        false,
        2,
        1,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn alu_slt_wrong_xor_test() {
    let x = [(1 << (LIMB_BITS - 1)) + 1; NUM_LIMBS];
    let y = [(1 << LIMB_BITS) - 1; NUM_LIMBS];
    run_alu_negative_test(
        U256Opcode::SLT,
        x.to_vec(),
        y.to_vec(),
        solve_subtract::<NUM_LIMBS, LIMB_BITS>(&x, &y).0,
        false,
        0,
        1,
        VerificationError::NonZeroCumulativeSum,
    );
}
