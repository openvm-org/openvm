use std::{array, borrow::BorrowMut, sync::Arc};

use afs_primitives::xor::lookup::XorLookupChip;
use afs_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_sdk::utils::create_seeded_rng;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{core::solve_alu, ArithmeticLogicCoreChip, Rv32ArithmeticLogicChip};
use crate::{
    arch::{
        instructions::AluOpcode,
        testing::{memory::gen_pointer, VmChipTestBuilder},
        InstructionExecutor, VmChip, VmChipWrapper,
    },
    kernels::core::BYTE_XOR_BUS,
    rv32im::{
        adapters::{
            test::Rv32TestAdapter, Rv32AluAdapter, Rv32RTypeAdapterInterface, RV_IS_TYPE_IMM_BITS,
        },
        new_alu::ArithmeticLogicCols,
    },
    system::program::Instruction,
};

type F = BabyBear;

const RV32_NUM_LIMBS: usize = 4;
const RV32_LIMB_BITS: usize = 8;

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

fn generate_rv32_immediate(rng: &mut StdRng) -> (Option<usize>, [u32; RV32_NUM_LIMBS]) {
    let mut imm: u32 = rng.gen_range(0..(1 << RV_IS_TYPE_IMM_BITS));
    if (imm & 0x800) != 0 {
        imm |= !0xFFF
    }
    (
        Some((imm & 0xFFFFFF) as usize),
        [
            imm as u8,
            (imm >> 8) as u8,
            (imm >> 16) as u8,
            (imm >> 16) as u8,
        ]
        .map(|x| x as u32),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_rv32_alu_rand_write_execute<E: InstructionExecutor<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    opcode: AluOpcode,
    b: [u32; RV32_NUM_LIMBS],
    c: [u32; RV32_NUM_LIMBS],
    c_imm: Option<usize>,
    rng: &mut StdRng,
) {
    let is_imm = c_imm.is_some();

    let rs1 = gen_pointer(rng, 32);
    let rs2 = c_imm.unwrap_or_else(|| gen_pointer(rng, 32));
    let rd = gen_pointer(rng, 32);

    tester.write::<RV32_NUM_LIMBS>(1, rs1, b.map(F::from_canonical_u32));
    if !is_imm {
        tester.write::<RV32_NUM_LIMBS>(1, rs2, c.map(F::from_canonical_u32));
    }

    let a = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(opcode, &b, &c);
    tester.execute(
        chip,
        Instruction::from_usize(
            opcode as usize,
            [rd, rs1, rs2, 1, if is_imm { 0 } else { 1 }],
        ),
    );

    assert_eq!(
        a.map(F::from_canonical_u32),
        tester.read::<RV32_NUM_LIMBS>(1, rd)
    );
}

fn run_rv32_alu_rand_test(opcode: AluOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();

    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let mut chip = Rv32ArithmeticLogicChip::<F>::new(
        Rv32AluAdapter::new(
            tester.execution_bus(),
            tester.program_bus(),
            tester.memory_controller(),
        ),
        ArithmeticLogicCoreChip::new(xor_lookup_chip.clone(), 0),
        tester.memory_controller(),
    );

    for _ in 0..num_ops {
        let b = generate_long_number::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(&mut rng);
        let (c_imm, c) = if false {
            // rng.gen_bool(0.5) {
            (
                None,
                generate_long_number::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(&mut rng),
            )
        } else {
            generate_rv32_immediate(&mut rng)
        };
        run_rv32_alu_rand_write_execute(&mut tester, &mut chip, opcode, b, c, c_imm, &mut rng);
    }

    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test]
fn rv32_alu_add_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::ADD, 12);
}

#[test]
fn rv32_alu_sub_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::SUB, 12);
}

#[test]
fn rv32_alu_xor_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::XOR, 12);
}

#[test]
fn rv32_alu_or_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::OR, 12);
}

#[test]
fn rv32_alu_and_rand_test() {
    run_rv32_alu_rand_test(AluOpcode::AND, 12);
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// Given a fake trace of a single operation, setup a chip and run the test. We replace
/// the write part of the trace and check that the core chip throws the expected error.
/// A dummy adaptor is used so memory interactions don't indirectly cause false passes.
///////////////////////////////////////////////////////////////////////////////////////

type Rv32ArithmeticLogicTestChip<F> = VmChipWrapper<
    F,
    Rv32TestAdapter<F, Rv32RTypeAdapterInterface<F>>,
    ArithmeticLogicCoreChip<RV32_NUM_LIMBS, RV32_LIMB_BITS>,
>;

#[allow(clippy::too_many_arguments)]
fn run_rv32_alu_negative_test(
    opcode: AluOpcode,
    a: [u32; RV32_NUM_LIMBS],
    b: [u32; RV32_NUM_LIMBS],
    c: [u32; RV32_NUM_LIMBS],
    expected_error: VerificationError,
) {
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_LIMB_BITS>::new(BYTE_XOR_BUS));
    let mut tester: VmChipTestBuilder<BabyBear> = VmChipTestBuilder::default();
    let mut chip = Rv32ArithmeticLogicTestChip::<F>::new(
        Rv32TestAdapter::new(
            [b.map(F::from_canonical_u32), c.map(F::from_canonical_u32)],
            None,
        ),
        ArithmeticLogicCoreChip::new(xor_lookup_chip.clone(), 0),
        tester.memory_controller(),
    );

    tester.execute(
        &mut chip,
        Instruction::from_usize(opcode as usize, [0, 0, 0, 1, 1]),
    );

    let alu_trace = chip.clone().generate_trace();
    let mut alu_trace_row = alu_trace.row_slice(0).to_vec();
    let alu_trace_cols: &mut ArithmeticLogicCols<F, RV32_NUM_LIMBS, RV32_LIMB_BITS> =
        (*alu_trace_row).borrow_mut();

    alu_trace_cols.a = a.map(F::from_canonical_u32);
    let alu_trace: p3_matrix::dense::DenseMatrix<_> = RowMajorMatrix::new(
        alu_trace_row,
        ArithmeticLogicCols::<F, RV32_NUM_LIMBS, RV32_LIMB_BITS>::width(),
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
fn rv32_alu_add_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::ADD,
        [246, 0, 0, 0],
        [250, 0, 0, 0],
        [250, 0, 0, 0],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn rv32_alu_add_out_of_range_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::ADD,
        [500, 0, 0, 0],
        [250, 0, 0, 0],
        [250, 0, 0, 0],
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn rv32_alu_sub_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::SUB,
        [255, 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn rv32_alu_sub_out_of_range_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::SUB,
        [F::neg_one().as_canonical_u32(), 0, 0, 0],
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn rv32_alu_xor_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::XOR,
        [255, 255, 255, 255],
        [0, 0, 1, 0],
        [255, 255, 255, 255],
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn rv32_alu_or_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::OR,
        [255, 255, 255, 255],
        [255, 255, 255, 254],
        [0, 0, 0, 0],
        VerificationError::NonZeroCumulativeSum,
    );
}

#[test]
fn rv32_alu_and_wrong_negative_test() {
    run_rv32_alu_negative_test(
        AluOpcode::AND,
        [255, 255, 255, 255],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        VerificationError::NonZeroCumulativeSum,
    );
}

///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn solve_add_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [23, 205, 73, 49];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::ADD, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn solve_sub_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [179, 118, 240, 172];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::SUB, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn solve_xor_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [215, 138, 49, 173];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::XOR, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn solve_or_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [247, 171, 61, 239];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::OR, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}

#[test]
fn solve_and_sanity_test() {
    let x: [u32; RV32_NUM_LIMBS] = [229, 33, 29, 111];
    let y: [u32; RV32_NUM_LIMBS] = [50, 171, 44, 194];
    let z: [u32; RV32_NUM_LIMBS] = [32, 33, 12, 66];
    let result = solve_alu::<RV32_NUM_LIMBS, RV32_LIMB_BITS>(AluOpcode::AND, &x, &y);
    for i in 0..RV32_NUM_LIMBS {
        assert_eq!(z[i], result[i])
    }
}
