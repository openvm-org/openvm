use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{
    columns::UintArithmeticCols, CalculationResult, UintArithmetic, UintArithmeticChip, NUM_LIMBS,
};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode, testing::MachineChipTestBuilder},
    cpu::trace::Instruction,
};

type F = BabyBear;

const OPCODES_ARITH: [Opcode; 2] = [Opcode::ADD256, Opcode::SUB256];

fn generate_uint_number<const ARG_SIZE: usize, const LIMB_SIZE: usize>(
    rng: &mut StdRng,
) -> Vec<u32> {
    assert_eq!(ARG_SIZE % LIMB_SIZE, 0);

    (0..ARG_SIZE / LIMB_SIZE)
        .map(|_| rng.gen_range(0..1 << LIMB_SIZE))
        .collect()
}

#[test]
fn uint_arithmetic_rand_air_test() {
    const ARG_SIZE: usize = 256;
    const LIMB_SIZE: usize = 8;
    let num_ops: usize = 15;
    let address_space_range = || 1usize..=2;
    let address_range = || 0usize..1 << 29;

    let mut tester = MachineChipTestBuilder::default();
    let mut chip = UintArithmeticChip::<ARG_SIZE, LIMB_SIZE, F>::new(
        tester.execution_bus(),
        tester.memory_chip(),
    );

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let opcode = OPCODES_ARITH[rng.gen_range(0..OPCODES_ARITH.len())];
        let operand1 = generate_uint_number::<ARG_SIZE, LIMB_SIZE>(&mut rng);
        let operand2 = generate_uint_number::<ARG_SIZE, LIMB_SIZE>(&mut rng);

        let ptr_as = rng.gen_range(address_space_range()); // d
        let result_as = rng.gen_range(address_space_range()); // e
        let as1 = rng.gen_range(address_space_range()); // f
        let as2 = rng.gen_range(address_space_range()); // g
        let address1 = rng.gen_range(address_range());
        let address2 = rng.gen_range(address_range());
        let address1_ptr = rng.gen_range(address_range());
        let address2_ptr = rng.gen_range(address_range());
        let result_ptr = rng.gen_range(address_range());
        let result_address = rng.gen_range(address_range());

        let operand1_f = operand1
            .clone()
            .into_iter()
            .map(F::from_canonical_u32)
            .collect::<Vec<_>>();
        let operand2_f = operand2
            .clone()
            .into_iter()
            .map(F::from_canonical_u32)
            .collect::<Vec<_>>();

        tester.write::<NUM_LIMBS>(as1, address1, operand1_f.as_slice().try_into().unwrap());
        tester.write_cell(ptr_as, address1_ptr, F::from_canonical_usize(address1));
        tester.write::<NUM_LIMBS>(as2, address2, operand2_f.as_slice().try_into().unwrap());
        tester.write_cell(ptr_as, address2_ptr, F::from_canonical_usize(address2));
        tester.write_cell(ptr_as, result_ptr, F::from_canonical_usize(result_address));

        let result =
            UintArithmetic::<ARG_SIZE, LIMB_SIZE, F>::solve(opcode, (&operand1, &operand2));

        tester.execute(
            &mut chip,
            Instruction::from_usize(
                opcode,
                [
                    result_ptr,
                    address1_ptr,
                    address2_ptr,
                    ptr_as,
                    result_as,
                    as1,
                    as2,
                ],
            ),
        );
        match result.0 {
            CalculationResult::Uint(result) => {
                assert_eq!(
                    result
                        .into_iter()
                        .map(F::from_canonical_u32)
                        .collect::<Vec<_>>(),
                    tester.read::<NUM_LIMBS>(result_as, result_address)
                )
            }
            CalculationResult::Short(_) => unreachable!(),
        }
    }

    let tester = tester.build().load(chip).finalize();

    tester.simple_test().expect("Verification failed");
}

/// Given a fake trace of a single operation, setup a chip and run the test.
/// We replace the "output" part of the trace, and we _may_ replace the interactions
/// based on the desired output. We check that it produces the error we expect.
#[allow(clippy::too_many_arguments)]
fn run_bad_uint_arithmetic_test(
    op: Opcode,
    x: Vec<u32>,
    y: Vec<u32>,
    z: Vec<u32>,
    buffer: Vec<u32>,
    cmp_result: bool,
    replace_interactions: bool,
    expected_error: VerificationError,
) {
    let mut tester = MachineChipTestBuilder::default();
    let mut chip =
        UintArithmeticChip::<256, 8, F>::new(tester.execution_bus(), tester.memory_chip());

    let x_f = x
        .iter()
        .map(|v| F::from_canonical_u32(*v))
        .collect::<Vec<_>>();
    let y_f = y
        .iter()
        .map(|v| F::from_canonical_u32(*v))
        .collect::<Vec<_>>();
    let ptr_as = 1;
    let mem_as = 2;
    tester.write::<NUM_LIMBS>(mem_as, 0, x_f.as_slice().try_into().unwrap());
    tester.write_cell(ptr_as, 2 * NUM_LIMBS, F::from_canonical_usize(0));
    tester.write::<NUM_LIMBS>(mem_as, NUM_LIMBS, y_f.as_slice().try_into().unwrap());
    tester.write_cell(
        ptr_as,
        2 * NUM_LIMBS + 1,
        F::from_canonical_usize(NUM_LIMBS),
    );
    tester.write_cell(ptr_as, 0, F::from_canonical_usize(0));

    tester.execute(
        &mut chip,
        Instruction::from_usize(
            op,
            [
                0,                 // result address ptr
                2 * NUM_LIMBS,     // x address ptr
                2 * NUM_LIMBS + 1, // y address ptr
                ptr_as,
                3,      // result as
                mem_as, // x as
                mem_as, // y as
            ],
        ),
    );

    if let CalculationResult::Uint(_) = UintArithmetic::<256, 8, F>::solve(op, (&x, &y)).0 {
        if replace_interactions {
            chip.range_checker_chip.clear();
            for limb in z.iter() {
                chip.range_checker_chip.add_count(*limb, 8);
            }
        }
    }

    let air = chip.air;
    let range_checker = chip.range_checker_chip.clone();
    let range_air = range_checker.air;
    let trace = chip.generate_trace();
    let row = trace.row_slice(0).to_vec();
    let mut cols = UintArithmeticCols::from_iterator(&mut row.into_iter(), &air);
    cols.io.z.data = z.into_iter().map(F::from_canonical_u32).collect();
    cols.aux.buffer = buffer.into_iter().map(F::from_canonical_u32).collect();
    cols.io.cmp_result = F::from_bool(cmp_result);
    let trace = RowMajorMatrix::new(cols.flatten(), UintArithmeticCols::<256, 8, F>::width(&air));

    let range_trace = range_checker.generate_trace();

    disable_debug_builder();
    let msg = format!(
        "Expected verification to fail with {:?}, but it didn't",
        &expected_error
    );
    assert_eq!(
        run_simple_test_no_pis(vec![&air, &range_air], vec![trace, range_trace],),
        Err(expected_error),
        "{}",
        msg
    );
}

#[test]
fn uint_add_wrong_carry_air_test() {
    run_bad_uint_arithmetic_test(
        Opcode::ADD256,
        std::iter::once(1)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(1)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(3)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(1)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        false,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn uint_add_out_of_range_air_test() {
    run_bad_uint_arithmetic_test(
        Opcode::ADD256,
        std::iter::once(250)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(250)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(500)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(0)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        false,
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn uint_add_wrong_addition_air_test() {
    run_bad_uint_arithmetic_test(
        Opcode::ADD256,
        std::iter::once(250)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(250)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(500 - (1 << 8))
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(0)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        false,
        VerificationError::OodEvaluationMismatch,
    );
}

// We NEED to check that the carry is 0 or 1
#[test]
fn uint_add_invalid_carry_air_test() {
    let bad_carry = F::from_canonical_u32(1 << 8).inverse().as_canonical_u32();

    run_bad_uint_arithmetic_test(
        Opcode::ADD256,
        vec![0; NUM_LIMBS - 1]
            .into_iter()
            .chain(std::iter::once(1))
            .collect(),
        vec![0; NUM_LIMBS - 1]
            .into_iter()
            .chain(std::iter::once(1))
            .collect(),
        vec![0; NUM_LIMBS - 1]
            .into_iter()
            .chain(std::iter::once(1))
            .collect(),
        vec![0; NUM_LIMBS - 1]
            .into_iter()
            .chain(std::iter::once(bad_carry))
            .collect(),
        false,
        true,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn uint_sub_out_of_range_air_test() {
    run_bad_uint_arithmetic_test(
        Opcode::SUB256,
        std::iter::once(1)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(2)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(F::neg_one().as_canonical_u32())
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(0)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        false,
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn uint_sub_wrong_subtraction_air_test() {
    run_bad_uint_arithmetic_test(
        Opcode::SUB256,
        std::iter::once(1)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(2)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once((1 << 8) - 1)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        std::iter::once(0)
            .chain(std::iter::repeat(0).take(NUM_LIMBS - 1))
            .collect(),
        false,
        false,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn uint_sub_invalid_carry_air_test() {
    let bad_carry = F::from_canonical_u32(1 << 8).inverse().as_canonical_u32();

    run_bad_uint_arithmetic_test(
        Opcode::SUB256,
        vec![0; NUM_LIMBS - 1]
            .into_iter()
            .chain(std::iter::once(1))
            .collect(),
        vec![0; NUM_LIMBS - 1]
            .into_iter()
            .chain(std::iter::once(1))
            .collect(),
        vec![0; NUM_LIMBS - 1]
            .into_iter()
            .chain(std::iter::once(1))
            .collect(),
        vec![0; NUM_LIMBS - 1]
            .into_iter()
            .chain(std::iter::once(bad_carry))
            .collect(),
        false,
        true,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn uint_lt_rand_air_test() {
    const ARG_SIZE: usize = 256;
    const LIMB_SIZE: usize = 8;
    let num_ops: usize = 15;
    let address_space_range = || 1usize..=2;
    let address_range = || 0usize..1 << 29;

    let mut tester = MachineChipTestBuilder::default();
    let mut chip = UintArithmeticChip::<ARG_SIZE, LIMB_SIZE, F>::new(
        tester.execution_bus(),
        tester.memory_chip(),
    );

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let opcode = Opcode::LT256;
        let operand1 = generate_uint_number::<ARG_SIZE, LIMB_SIZE>(&mut rng);
        let operand2 = generate_uint_number::<ARG_SIZE, LIMB_SIZE>(&mut rng);

        let ptr_as = rng.gen_range(address_space_range()); // d
        let result_as = rng.gen_range(address_space_range()); // e
        let as1 = rng.gen_range(address_space_range()); // f
        let as2 = rng.gen_range(address_space_range()); // g
        let address1 = rng.gen_range(address_range());
        let address2 = rng.gen_range(address_range());
        let address1_ptr = rng.gen_range(address_range());
        let address2_ptr = rng.gen_range(address_range());
        let result_ptr = rng.gen_range(address_range());
        let result_address = rng.gen_range(address_range());

        let operand1_f = operand1
            .clone()
            .into_iter()
            .map(F::from_canonical_u32)
            .collect::<Vec<_>>();
        let operand2_f = operand2
            .clone()
            .into_iter()
            .map(F::from_canonical_u32)
            .collect::<Vec<_>>();

        tester.write::<NUM_LIMBS>(as1, address1, operand1_f.as_slice().try_into().unwrap());
        tester.write_cell(ptr_as, address1_ptr, F::from_canonical_usize(address1));
        tester.write::<NUM_LIMBS>(as2, address2, operand2_f.as_slice().try_into().unwrap());
        tester.write_cell(ptr_as, address2_ptr, F::from_canonical_usize(address2));
        tester.write_cell(ptr_as, result_ptr, F::from_canonical_usize(result_address));

        let result =
            UintArithmetic::<ARG_SIZE, LIMB_SIZE, F>::solve(opcode, (&operand1, &operand2));

        tester.execute(
            &mut chip,
            Instruction::from_usize(
                opcode,
                [
                    result_ptr,
                    address1_ptr,
                    address2_ptr,
                    ptr_as,
                    result_as,
                    as1,
                    as2,
                ],
            ),
        );
        match result.0 {
            CalculationResult::Uint(_) => unreachable!(),
            CalculationResult::Short(result) => {
                assert_eq!(
                    [F::from_bool(result)],
                    tester.read::<1>(result_as, result_address)
                )
            }
        }
    }

    let tester = tester.build().load(chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn uint_eq_rand_air_test() {
    const ARG_SIZE: usize = 256;
    const LIMB_SIZE: usize = 8;
    let num_ops: usize = 15;
    let address_space_range = || 1usize..=2;
    let address_range = || 0usize..1 << 29;

    let mut tester = MachineChipTestBuilder::default();
    let mut chip = UintArithmeticChip::<ARG_SIZE, LIMB_SIZE, F>::new(
        tester.execution_bus(),
        tester.memory_chip(),
    );

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let opcode = Opcode::EQ256;
        let operand1 = generate_uint_number::<ARG_SIZE, LIMB_SIZE>(&mut rng);
        let operand2 = if rng.gen_bool(0.5) {
            generate_uint_number::<ARG_SIZE, LIMB_SIZE>(&mut rng)
        } else {
            operand1.clone()
        };

        let ptr_as = rng.gen_range(address_space_range()); // d
        let result_as = rng.gen_range(address_space_range()); // e
        let as1 = rng.gen_range(address_space_range()); // f
        let as2 = rng.gen_range(address_space_range()); // g
        let address1 = rng.gen_range(address_range());
        let address2 = rng.gen_range(address_range());
        let address1_ptr = rng.gen_range(address_range());
        let address2_ptr = rng.gen_range(address_range());
        let result_ptr = rng.gen_range(address_range());
        let result_address = rng.gen_range(address_range());

        let operand1_f = operand1
            .clone()
            .into_iter()
            .map(F::from_canonical_u32)
            .collect::<Vec<_>>();
        let operand2_f = operand2
            .clone()
            .into_iter()
            .map(F::from_canonical_u32)
            .collect::<Vec<_>>();

        tester.write::<NUM_LIMBS>(as1, address1, operand1_f.as_slice().try_into().unwrap());
        tester.write_cell(ptr_as, address1_ptr, F::from_canonical_usize(address1));
        tester.write::<NUM_LIMBS>(as2, address2, operand2_f.as_slice().try_into().unwrap());
        tester.write_cell(ptr_as, address2_ptr, F::from_canonical_usize(address2));
        tester.write_cell(ptr_as, result_ptr, F::from_canonical_usize(result_address));

        let result =
            UintArithmetic::<ARG_SIZE, LIMB_SIZE, F>::solve(opcode, (&operand1, &operand2));

        tester.execute(
            &mut chip,
            Instruction::from_usize(
                opcode,
                [
                    result_ptr,
                    address1_ptr,
                    address2_ptr,
                    ptr_as,
                    result_as,
                    as1,
                    as2,
                ],
            ),
        );
        match result.0 {
            CalculationResult::Uint(_) => unreachable!(),
            CalculationResult::Short(result) => {
                assert_eq!(
                    [F::from_bool(result)],
                    tester.read::<1>(result_as, result_address)
                )
            }
        }
    }

    let tester = tester.build().load(chip).finalize();

    tester.simple_test().expect("Verification failed");
}

#[test]
fn uint_lt_wrong_subtraction_test() {
    run_bad_uint_arithmetic_test(
        Opcode::LT256,
        std::iter::once(65_000)
            .chain(std::iter::repeat(0).take(31))
            .collect(),
        std::iter::once(65_000)
            .chain(std::iter::repeat(0).take(31))
            .collect(),
        std::iter::once(1)
            .chain(std::iter::repeat(0).take(31))
            .collect(),
        std::iter::once(0)
            .chain(std::iter::repeat(0).take(31))
            .collect(),
        false,
        false,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn uint_lt_wrong_carry_test() {
    run_bad_uint_arithmetic_test(
        Opcode::LT256,
        vec![0; 31]
            .into_iter()
            .chain(std::iter::once(65_000))
            .collect(),
        vec![0; 31]
            .into_iter()
            .chain(std::iter::once(65_000))
            .collect(),
        vec![0; 31].into_iter().chain(std::iter::once(0)).collect(),
        vec![0; 31].into_iter().chain(std::iter::once(1)).collect(),
        true,
        false,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn uint_eq_wrong_positive_test() {
    run_bad_uint_arithmetic_test(
        Opcode::EQ256,
        vec![0; 31]
            .into_iter()
            .chain(std::iter::once(123))
            .collect(),
        vec![0; 31]
            .into_iter()
            .chain(std::iter::once(456))
            .collect(),
        vec![0; 31].into_iter().chain(std::iter::once(0)).collect(),
        vec![0; 31].into_iter().chain(std::iter::once(0)).collect(),
        true,
        false,
        VerificationError::OodEvaluationMismatch,
    );
}
