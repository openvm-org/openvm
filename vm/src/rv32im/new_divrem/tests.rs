use std::{array, sync::Arc};

use afs_primitives::{
    range_tuple::{RangeTupleCheckerBus, RangeTupleCheckerChip},
    xor::lookup::XorLookupChip,
};
use ax_sdk::utils::create_seeded_rng;
use axvm_instructions::DivRemOpcode;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::{rngs::StdRng, Rng};

use super::core::run_divrem;
use crate::{
    arch::{
        testing::{memory::gen_pointer, VmChipTestBuilder},
        InstructionExecutor,
    },
    kernels::core::{BYTE_XOR_BUS, RANGE_TUPLE_CHECKER_BUS},
    rv32im::{
        adapters::{Rv32MultAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
        new_divrem::{run_mul_carries, run_sltu_diff_idx, DivRemCoreChip, Rv32DivRemChip},
    },
    system::program::Instruction,
};

type F = BabyBear;

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

fn generate_long_number<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    rng: &mut StdRng,
) -> [u32; NUM_LIMBS] {
    array::from_fn(|_| rng.gen_range(0..(1 << LIMB_BITS)))
}

fn limb_sra<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    x: [u32; NUM_LIMBS],
    shift: usize,
) -> [u32; NUM_LIMBS] {
    assert!(shift < NUM_LIMBS);
    let ext = (x[NUM_LIMBS - 1] >> (LIMB_BITS - 1)) * ((1 << LIMB_BITS) - 1);
    array::from_fn(|i| if i + shift < NUM_LIMBS { x[i] } else { ext })
}

#[allow(clippy::too_many_arguments)]
fn run_rv32_divrem_rand_write_execute<E: InstructionExecutor<F>>(
    opcode: DivRemOpcode,
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    b: [u32; RV32_REGISTER_NUM_LIMBS],
    c: [u32; RV32_REGISTER_NUM_LIMBS],
    rng: &mut StdRng,
) {
    let rs1 = gen_pointer(rng, 32);
    let rs2 = gen_pointer(rng, 32);
    let rd = gen_pointer(rng, 32);

    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
    tester.write::<RV32_REGISTER_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));

    let is_div = opcode == DivRemOpcode::DIV || opcode == DivRemOpcode::DIVU;
    let is_signed = opcode == DivRemOpcode::DIV || opcode == DivRemOpcode::REM;

    let (q, r, _, _, _) = run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(is_signed, &b, &c);
    tester.execute(
        chip,
        Instruction::from_usize(opcode as usize, [rd, rs1, rs2, 1, 0]),
    );

    assert_eq!(
        (if is_div { q } else { r }).map(F::from_canonical_u32),
        tester.read::<RV32_REGISTER_NUM_LIMBS>(1, rd)
    );
}

fn run_rv32_divrem_rand_test(opcode: DivRemOpcode, num_ops: usize) {
    // the max number of limbs we currently support MUL for is 32 (i.e. for U256s)
    const MAX_NUM_LIMBS: u32 = 32;
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let range_tuple_bus = RangeTupleCheckerBus::new(
        RANGE_TUPLE_CHECKER_BUS,
        [1 << RV32_CELL_BITS, MAX_NUM_LIMBS * (1 << RV32_CELL_BITS)],
    );
    let range_tuple_checker = Arc::new(RangeTupleCheckerChip::new(range_tuple_bus));

    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32DivRemChip::<F>::new(
        Rv32MultAdapterChip::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        DivRemCoreChip::new(xor_lookup_chip.clone(), range_tuple_checker.clone(), 0),
        tester.memory_controller(),
    );

    for _ in 0..num_ops {
        let b = generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng);
        let leading_zeros = rng.gen_range(0..(RV32_REGISTER_NUM_LIMBS - 1));
        let c = limb_sra::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
            generate_long_number::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(&mut rng),
            leading_zeros,
        );
        run_rv32_divrem_rand_write_execute(opcode, &mut tester, &mut chip, b, c, &mut rng);
    }

    // Test special cases in addition to random cases (i.e. zero divisor with b > 0,
    // zero divisor with b < 0, and signed overflow).
    run_rv32_divrem_rand_write_execute(
        opcode,
        &mut tester,
        &mut chip,
        [98, 188, 163, 127],
        [0, 0, 0, 0],
        &mut rng,
    );
    run_rv32_divrem_rand_write_execute(
        opcode,
        &mut tester,
        &mut chip,
        [98, 188, 163, 229],
        [0, 0, 0, 0],
        &mut rng,
    );
    run_rv32_divrem_rand_write_execute(
        opcode,
        &mut tester,
        &mut chip,
        [0, 0, 0, 128],
        [255, 255, 255, 255],
        &mut rng,
    );

    let tester = tester
        .build()
        .load(chip)
        .load(xor_lookup_chip)
        .load(range_tuple_checker)
        .finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv32_div_rand_test() {
    run_rv32_divrem_rand_test(DivRemOpcode::DIV, 100);
}

#[test]
fn rv32_divu_rand_test() {
    run_rv32_divrem_rand_test(DivRemOpcode::DIVU, 100);
}

#[test]
fn rv32_rem_rand_test() {
    run_rv32_divrem_rand_test(DivRemOpcode::REM, 100);
}

#[test]
fn rv32_remu_rand_test() {
    run_rv32_divrem_rand_test(DivRemOpcode::REMU, 100);
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// Given a fake trace of a single operation, setup a chip and run the test. We replace
/// the write part of the trace and check that the core chip throws the expected error.
/// A dummy adapter is used so memory interactions don't indirectly cause false passes.
///////////////////////////////////////////////////////////////////////////////////////

// TODO: write negative tests

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn run_divrem_unsigned_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 0, 0];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [245, 168, 6, 0];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [171, 4, 0, 0];

    let (res_q, res_r, x_sign, y_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(false, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(!x_sign);
    assert!(!y_sign);
    assert_eq!(case, 0);
}

#[test]
fn run_divrem_unsigned_zero_divisor_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [255, 255, 255, 255];

    let (res_q, res_r, x_sign, y_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(false, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert!(!x_sign);
    assert!(!y_sign);
    assert_eq!(case, 1);
}

#[test]
fn run_divrem_signed_sanity_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 0, 0];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [74, 60, 255, 255];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [212, 240, 255, 255];

    let (res_q, res_r, x_sign, y_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(!y_sign);
    assert_eq!(case, 0);
}

#[test]
fn run_divrem_signed_zero_divisor_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [98, 188, 163, 229];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [255, 255, 255, 255];

    let (res_q, res_r, x_sign, y_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(x[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(!y_sign);
    assert_eq!(case, 1);
}

#[test]
fn run_divrem_signed_overflow_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 128];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [255, 255, 255, 255];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 0];

    let (res_q, res_r, x_sign, y_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(x[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(y_sign);
    assert_eq!(case, 2);
}

#[test]
fn run_divrem_signed_min_dividend_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [0, 0, 0, 128];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 255, 255];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [236, 147, 0, 0];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [156, 149, 255, 255];

    let (res_q, res_r, x_sign, y_sign, case) =
        run_divrem::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &x, &y);
    for i in 0..RV32_REGISTER_NUM_LIMBS {
        assert_eq!(q[i], res_q[i]);
        assert_eq!(r[i], res_r[i]);
    }
    assert!(x_sign);
    assert!(y_sign);
    assert_eq!(case, 0);
}

#[test]
fn run_sltu_diff_idx_test() {
    let x: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 254, 67];
    let y: [u32; RV32_REGISTER_NUM_LIMBS] = [123, 34, 255, 67];
    assert_eq!(run_sltu_diff_idx(&x, &y, true), 2);
    assert_eq!(run_sltu_diff_idx(&y, &x, false), 2);
    assert_eq!(run_sltu_diff_idx(&x, &x, false), RV32_REGISTER_NUM_LIMBS);
}

#[test]
fn run_mul_carries_signed_sanity_test() {
    let d: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [200, 8, 68, 256];
    let c = [40, 101, 126, 206, 303, 375, 449, 463];
    let carry = run_mul_carries::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(true, &d, &q, &r);
    for (expected_c, actual_c) in c.iter().zip(carry.iter()) {
        assert_eq!(*expected_c, *actual_c)
    }
}

#[test]
fn run_mul_sanity_unsigned_test() {
    let d: [u32; RV32_REGISTER_NUM_LIMBS] = [197, 85, 150, 32];
    let q: [u32; RV32_REGISTER_NUM_LIMBS] = [51, 109, 78, 142];
    let r: [u32; RV32_REGISTER_NUM_LIMBS] = [200, 8, 68, 256];
    let c = [40, 101, 126, 206, 107, 93, 18, 0];
    let carry = run_mul_carries::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(false, &d, &q, &r);
    for (expected_c, actual_c) in c.iter().zip(carry.iter()) {
        assert_eq!(*expected_c, *actual_c)
    }
}
