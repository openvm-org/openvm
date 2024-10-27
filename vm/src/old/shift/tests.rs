use std::{array, borrow::BorrowMut, iter, sync::Arc};

use afs_primitives::xor::XorLookupChip;
use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError, Chip};
use ax_sdk::utils::create_seeded_rng;
use axvm_instructions::instruction::Instruction;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};
use test_log::test;

use super::{run_shift, ShiftChip};
use crate::{
    arch::{
        instructions::U256Opcode,
        testing::{memory::gen_pointer, VmChipTestBuilder},
        BYTE_XOR_BUS,
    },
    old::shift::columns::ShiftCols,
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

fn generate_shift<const NUM_LIMBS: usize, const LIMB_BITS: usize>(rng: &mut StdRng) -> Vec<u32> {
    iter::once(rng.gen_range(0..1 << LIMB_BITS))
        .chain(iter::repeat(0))
        .take(NUM_LIMBS)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_shift_rand_write_execute<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut ShiftChip<F, NUM_LIMBS, LIMB_BITS>,
    opcode: U256Opcode,
    x: Vec<u32>,
    y: Vec<u32>,
    rng: &mut StdRng,
) {
    let address_space_range = || 1usize..=2;

    let d = rng.gen_range(address_space_range());
    let e = rng.gen_range(address_space_range());

    let x_address = gen_pointer(rng, 64);
    let y_address = gen_pointer(rng, 64);
    let res_address = gen_pointer(rng, 64);
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

    let (z, _, _) = run_shift::<NUM_LIMBS, LIMB_BITS>(&x, &y, opcode);
    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize,
            [res_ptr_to_address, x_ptr_to_address, y_ptr_to_address, d, e],
        ),
    );

    assert_eq!(
        z.into_iter().map(F::from_canonical_u32).collect::<Vec<_>>(),
        tester.read::<NUM_LIMBS>(e, res_address)
    )
}

#[allow(clippy::too_many_arguments)]
fn run_shift_negative_test(
    opcode: U256Opcode,
    x: Vec<u32>,
    y: Vec<u32>,
    z: Vec<u32>,
    bit_shift: u32,
    bit_multiplier_left: u32,
    bit_multiplier_right: u32,
    x_sign: u32,
    bit_shift_carry: Vec<u32>,
    expected_error: VerificationError,
) {
    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        xor_lookup_chip.clone(),
        0,
    );

    let mut rng = create_seeded_rng();
    run_shift_rand_write_execute::<NUM_LIMBS, LIMB_BITS>(
        &mut tester,
        &mut chip,
        opcode,
        x,
        y,
        &mut rng,
    );

    if expected_error == VerificationError::NonZeroCumulativeSum {
        chip.range_checker_chip.clear();
        chip.range_checker_chip
            .add_count(bit_shift, LIMB_BITS.ilog2() as usize);
        for (z_val, carry_val) in z.iter().zip(bit_shift_carry.iter()) {
            chip.range_checker_chip.add_count(*z_val, LIMB_BITS);
            chip.range_checker_chip
                .add_count(*carry_val, bit_shift as usize);
        }
    }
    let mut air_proof_input = chip.generate_air_proof_input();
    let shift_trace = air_proof_input.raw.common_main.as_mut().unwrap();
    let mut shift_trace_vec = shift_trace.row_slice(0).to_vec();
    let shift_trace_cols: &mut ShiftCols<F, NUM_LIMBS, LIMB_BITS> = (*shift_trace_vec).borrow_mut();

    shift_trace_cols.io.z.data = array::from_fn(|i| F::from_canonical_u32(z[i]));
    shift_trace_cols.aux.bit_shift = F::from_canonical_u32(bit_shift);
    shift_trace_cols.aux.bit_multiplier_left = F::from_canonical_u32(bit_multiplier_left);
    shift_trace_cols.aux.bit_multiplier_right = F::from_canonical_u32(bit_multiplier_right);
    shift_trace_cols.aux.x_sign = F::from_canonical_u32(x_sign);
    shift_trace_cols.aux.bit_shift_carry =
        array::from_fn(|i| F::from_canonical_u32(bit_shift_carry[i]));

    *shift_trace = RowMajorMatrix::new(
        shift_trace_vec,
        ShiftCols::<F, NUM_LIMBS, LIMB_BITS>::width(),
    );

    disable_debug_builder();
    let mut tester = tester.build();
    tester.air_proof_inputs.push(air_proof_input);
    let tester = tester.load(xor_lookup_chip).finalize();
    let msg = format!(
        "Expected verification to fail with {:?}, but it didn't",
        &expected_error
    );
    let result = tester.simple_test();
    assert_eq!(result.err(), Some(expected_error), "{}", msg);
}

#[test]
fn shift_sll_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_shift::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_shift_rand_write_execute(&mut tester, &mut chip, U256Opcode::SLL, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn shift_sll_wrong_answer_negative_test() {
    run_shift_negative_test(
        U256Opcode::SLL,
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(4)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        1,
        2,
        0,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_sll_wrong_bit_shift_negative_test() {
    run_shift_negative_test(
        U256Opcode::SLL,
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        2,
        2,
        0,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_sll_wrong_bit_mult_negative_test() {
    run_shift_negative_test(
        U256Opcode::SLL,
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(4)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        1,
        4,
        0,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_sll_nonzero_bit_mult_right_negative_test() {
    run_shift_negative_test(
        U256Opcode::SLL,
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        1,
        2,
        1,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_sll_nonzero_sign_negative_test() {
    run_shift_negative_test(
        U256Opcode::SLL,
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0).take(NUM_LIMBS - 3))
            .chain(iter::repeat(1).take(2))
            .collect(),
        1,
        2,
        0,
        1,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_sll_out_of_range_carry_negative_test() {
    run_shift_negative_test(
        U256Opcode::SLL,
        iter::once(1 << LIMB_BITS)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(0)
            .chain(iter::once(2))
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        1,
        2,
        0,
        0,
        iter::once(2)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn shift_srl_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_shift::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_shift_rand_write_execute(&mut tester, &mut chip, U256Opcode::SRL, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn shift_srl_wrong_answer_negative_test() {
    run_shift_negative_test(
        U256Opcode::SRL,
        iter::once(4)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        1,
        0,
        2,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_srl_wrong_extension_negative_test() {
    run_shift_negative_test(
        U256Opcode::SRL,
        iter::once(4)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0).take(NUM_LIMBS - 2))
            .chain(iter::once(1 << (LIMB_BITS - 1)))
            .collect(),
        1,
        0,
        2,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_srl_nonzero_bit_mult_left_negative_test() {
    run_shift_negative_test(
        U256Opcode::SRL,
        iter::once(4)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        1,
        2,
        2,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_srl_nonzero_sign_negative_test() {
    run_shift_negative_test(
        U256Opcode::SRL,
        iter::once(4)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0).take(NUM_LIMBS - 3))
            .chain(iter::repeat(1).take(2))
            .collect(),
        1,
        2,
        0,
        1,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_sra_rand_test() {
    let num_ops: usize = 10;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        xor_lookup_chip.clone(),
        0,
    );

    for _ in 0..num_ops {
        let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        let y = generate_shift::<NUM_LIMBS, LIMB_BITS>(&mut rng);
        run_shift_rand_write_execute(&mut tester, &mut chip, U256Opcode::SRA, x, y, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn shift_sra_wrong_answer_negative_test() {
    run_shift_negative_test(
        U256Opcode::SRA,
        iter::once(4)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        1,
        0,
        2,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_sra_wrong_extension_negative_test() {
    run_shift_negative_test(
        U256Opcode::SRA,
        iter::once(4)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::once(2)
            .chain(iter::repeat(0).take(NUM_LIMBS - 2))
            .chain(iter::once(1 << (LIMB_BITS - 1)))
            .collect(),
        1,
        0,
        2,
        0,
        vec![0; NUM_LIMBS],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn shift_sra_wrong_sign_negative_test() {
    run_shift_negative_test(
        U256Opcode::SRA,
        vec![(1 << LIMB_BITS) - 1; NUM_LIMBS],
        iter::once(1)
            .chain(iter::repeat(0))
            .take(NUM_LIMBS)
            .collect(),
        iter::repeat((1 << LIMB_BITS) - 1)
            .take(NUM_LIMBS - 1)
            .chain(iter::once((1 << (LIMB_BITS - 1)) - 1))
            .collect(),
        1,
        0,
        2,
        0,
        vec![1; NUM_LIMBS],
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn shift_overflow_test() {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = ShiftChip::<F, NUM_LIMBS, LIMB_BITS>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        xor_lookup_chip.clone(),
        0,
    );

    let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
    let mut y = generate_long_number::<NUM_LIMBS, LIMB_BITS>(&mut rng);
    y[1] = 100;

    run_shift_rand_write_execute(
        &mut tester,
        &mut chip,
        U256Opcode::SLL,
        x.clone(),
        y.clone(),
        &mut rng,
    );
    run_shift_rand_write_execute(
        &mut tester,
        &mut chip,
        U256Opcode::SRL,
        x.clone(),
        y.clone(),
        &mut rng,
    );
    run_shift_rand_write_execute(
        &mut tester,
        &mut chip,
        U256Opcode::SRA,
        x.clone(),
        y.clone(),
        &mut rng,
    );

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn run_sll_sanity_test() {
    let x: [u32; 32] = [
        45, 7, 61, 186, 49, 53, 119, 68, 145, 55, 102, 126, 9, 195, 23, 26, 197, 216, 251, 31, 74,
        237, 141, 92, 98, 184, 176, 106, 64, 29, 58, 246,
    ];
    let y: [u32; 32] = [
        27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let z: [u32; 32] = [
        0, 0, 0, 104, 57, 232, 209, 141, 169, 185, 35, 138, 188, 49, 243, 75, 24, 190, 208, 40,
        198, 222, 255, 80, 106, 111, 228, 18, 195, 133, 85, 3,
    ];
    let sll_result = run_shift::<32, 8>(&x, &y, U256Opcode::SLL).0;
    for i in 0..32 {
        assert_eq!(z[i], sll_result[i])
    }
}

#[test]
fn run_srl_sanity_test() {
    let x: [u32; 32] = [
        253, 247, 209, 166, 217, 253, 46, 42, 197, 8, 33, 136, 144, 148, 101, 195, 173, 150, 26,
        215, 233, 90, 213, 185, 119, 255, 238, 174, 31, 190, 221, 72,
    ];
    let y: [u32; 32] = [
        17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let z: [u32; 32] = [
        104, 211, 236, 126, 23, 149, 98, 132, 16, 68, 72, 202, 178, 225, 86, 75, 141, 235, 116,
        173, 234, 220, 187, 127, 119, 215, 15, 223, 110, 36, 0, 0,
    ];
    let srl_result = run_shift::<32, 8>(&x, &y, U256Opcode::SRL).0;
    let sra_result = run_shift::<32, 8>(&x, &y, U256Opcode::SRA).0;
    for i in 0..32 {
        assert_eq!(z[i], srl_result[i]);
        assert_eq!(z[i], sra_result[i]);
    }
}

#[test]
fn run_sra_sanity_test() {
    let x: [u32; 32] = [
        253, 247, 209, 166, 217, 253, 46, 42, 197, 8, 33, 136, 144, 148, 101, 195, 173, 150, 26,
        215, 233, 90, 213, 185, 119, 255, 238, 174, 31, 190, 221, 200,
    ];
    let y: [u32; 32] = [
        17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let z: [u32; 32] = [
        104, 211, 236, 126, 23, 149, 98, 132, 16, 68, 72, 202, 178, 225, 86, 75, 141, 235, 116,
        173, 234, 220, 187, 127, 119, 215, 15, 223, 110, 228, 255, 255,
    ];
    let sra_result = run_shift::<32, 8>(&x, &y, U256Opcode::SRA).0;
    for i in 0..32 {
        assert_eq!(z[i], sra_result[i])
    }
}
