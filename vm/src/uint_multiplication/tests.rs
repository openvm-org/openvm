use std::{array, borrow::BorrowMut, iter, sync::Arc};

use afs_primitives::range_tuple::{bus::RangeTupleCheckerBus, RangeTupleCheckerChip};
use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::{config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{columns::UintMultiplicationCols, solve_uint_multiplication, UintMultiplicationChip};
use crate::{
    arch::{chips::MachineChip, instructions::Opcode, testing::MachineChipTestBuilder},
    cpu::{trace::Instruction, RANGE_TUPLE_CHECKER_BUS},
};

type F = BabyBear;

fn generate_uint_number<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rng: &mut StdRng,
) -> Vec<u32> {
    (0..NUM_LIMBS)
        .map(|_| rng.gen_range(0..1 << LIMB_BITS))
        .collect()
}

fn run_uint_multiplication_rand_write_execute<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    tester: &mut MachineChipTestBuilder<F>,
    chip: &mut UintMultiplicationChip<F, NUM_LIMBS, LIMB_BITS>,
    x: Vec<u32>,
    y: Vec<u32>,
    rng: &mut StdRng,
) {
    let address_space_range = || 1usize..=2;
    let address_range = || 0usize..1 << 29;

    let d = rng.gen_range(address_space_range());
    let e = rng.gen_range(address_space_range());

    let x_address = rng.gen_range(address_range());
    let y_address = rng.gen_range(address_range());
    let z_address = rng.gen_range(address_range());
    let x_ptr_to_address = rng.gen_range(address_range());
    let y_ptr_to_address = rng.gen_range(address_range());
    let z_ptr_to_address = rng.gen_range(address_range());

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
    tester.write_cell(d, z_ptr_to_address, F::from_canonical_usize(z_address));
    tester.write::<NUM_LIMBS>(e, x_address, x_f.as_slice().try_into().unwrap());
    tester.write::<NUM_LIMBS>(e, y_address, y_f.as_slice().try_into().unwrap());

    let (z, _) = solve_uint_multiplication::<NUM_LIMBS, LIMB_BITS>(&x, &y);
    tester.execute(
        chip,
        Instruction::from_usize(
            Opcode::MUL256,
            [z_ptr_to_address, x_ptr_to_address, y_ptr_to_address, d, e],
        ),
    );
    assert_eq!(
        z.into_iter().map(F::from_canonical_u32).collect::<Vec<_>>(),
        tester.read::<NUM_LIMBS>(e, z_address)
    );
}

fn run_negative_uint_multiplication_test<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: Vec<u32>,
    y: Vec<u32>,
    z: Vec<u32>,
    carry: Vec<u32>,
    expected_error: VerificationError,
) {
    let bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        vec![1 << LIMB_BITS, (NUM_LIMBS * (1 << LIMB_BITS)) as u32],
    );
    let range_tuple_chip: Arc<RangeTupleCheckerChip> = Arc::new(RangeTupleCheckerChip::new(bus));

    let mut tester = MachineChipTestBuilder::default();
    let mut chip = UintMultiplicationChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.memory_chip(),
        range_tuple_chip.clone(),
    );

    let mut rng = create_seeded_rng();
    run_uint_multiplication_rand_write_execute(&mut tester, &mut chip, x, y, &mut rng);

    let mult_air = chip.air.clone();
    let mult_trace = chip.generate_trace();

    let mut mult_trace_vec = mult_trace.row_slice(0).to_vec();
    let mult_trace_cols: &mut UintMultiplicationCols<F, NUM_LIMBS, LIMB_BITS> =
        (*mult_trace_vec).borrow_mut();
    mult_trace_cols.io.z.data = array::from_fn(|i| F::from_canonical_u32(z[i]));
    mult_trace_cols.aux.carry = array::from_fn(|i| F::from_canonical_u32(carry[i]));
    let mult_trace: p3_matrix::dense::DenseMatrix<BabyBear> = RowMajorMatrix::new(
        mult_trace_vec,
        UintMultiplicationCols::<F, NUM_LIMBS, LIMB_BITS>::width(),
    );

    let range_air = range_tuple_chip.air.clone();
    let range_trace = range_tuple_chip.generate_trace();

    disable_debug_builder();
    let msg = format!(
        "Expected verification to fail with {:?}, but it didn't",
        &expected_error
    );
    assert_eq!(
        run_simple_test_no_pis(vec![&mult_air, &range_air], vec![mult_trace, range_trace],),
        Err(expected_error),
        "{}",
        msg
    );
}

#[test]
fn uint_multiplication_rand_air_test() {
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    let num_ops: usize = 10;

    let bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        vec![1 << LIMB_BITS, (NUM_LIMBS * (1 << LIMB_BITS)) as u32],
    );
    let range_tuple_chip: Arc<RangeTupleCheckerChip> = Arc::new(RangeTupleCheckerChip::new(bus));

    let mut tester = MachineChipTestBuilder::default();
    let mut chip = UintMultiplicationChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.memory_chip(),
        range_tuple_chip.clone(),
    );

    let mut rng = create_seeded_rng();

    for _ in 0..num_ops {
        let x = generate_uint_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_uint_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_uint_multiplication_rand_write_execute(&mut tester, &mut chip, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(range_tuple_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn negative_uint_multiplication_wrong_calc_test() {
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    // x = 00...0001
    // y = 00...0001
    // z = 00...0002
    // carry = 00...0000
    run_negative_uint_multiplication_test::<NUM_LIMBS, LIMB_BITS>(
        iter::once(1).chain(iter::repeat(0).take(31)).collect(),
        iter::once(1).chain(iter::repeat(0).take(31)).collect(),
        iter::once(2).chain(iter::repeat(0).take(31)).collect(),
        iter::repeat(0).take(32).collect(),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn negative_uint_multiplication_wrong_carry_test() {
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    // x = 00...0001
    // y = 00...0001
    // z = 00...0001
    // carry = 00...0001
    run_negative_uint_multiplication_test::<NUM_LIMBS, LIMB_BITS>(
        iter::once(1).chain(iter::repeat(0).take(31)).collect(), // 10000000...
        iter::once(1).chain(iter::repeat(0).take(31)).collect(), // 10000000...
        iter::once(1).chain(iter::repeat(0).take(31)).collect(), // 10000000...
        iter::once(1).chain(iter::repeat(0).take(31)).collect(),
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn negative_uint_multiplication_out_of_range_z_test() {
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    // x = 00...000[2^8] (out of range)
    // y = 00...0001
    // z = 00...000[2^8] (out of range)
    // carry = 00...0000
    run_negative_uint_multiplication_test::<NUM_LIMBS, LIMB_BITS>(
        iter::once(1 << LIMB_BITS)
            .chain(iter::repeat(0).take(31))
            .collect(),
        iter::once(1).chain(iter::repeat(0).take(31)).collect(),
        iter::once(1 << LIMB_BITS)
            .chain(iter::repeat(0).take(31))
            .collect(),
        iter::repeat(0).take(32).collect(),
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn negative_uint_multiplication_out_of_range_carry_test() {
    const NUM_LIMBS: usize = 32;
    const LIMB_BITS: usize = 8;
    // x = 00...000[2^8] (out of range)
    // y = 00...000[2^8] (out of range)
    // z = 00...0100
    // carry = 00...001[2^8] (out of range)
    run_negative_uint_multiplication_test::<NUM_LIMBS, LIMB_BITS>(
        iter::once(1 << LIMB_BITS)
            .chain(iter::repeat(0).take(31))
            .collect(),
        iter::once(1 << LIMB_BITS)
            .chain(iter::repeat(0).take(31))
            .collect(),
        iter::repeat(0)
            .take(2)
            .chain(iter::once(1))
            .chain(iter::repeat(0).take(29))
            .collect(),
        iter::once(1 << LIMB_BITS)
            .chain(iter::once(1))
            .chain(iter::repeat(0).take(30))
            .collect(),
        VerificationError::NonZeroCumulativeSum,
    );
}
