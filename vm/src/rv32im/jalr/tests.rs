use std::{array, borrow::BorrowMut, sync::Arc};

use ax_circuit_primitives::xor::XorLookupChip;
use ax_sdk::utils::create_seeded_rng;
use ax_stark_backend::{
    utils::disable_debug_builder, verifier::VerificationError, Chip, ChipUsageGetter,
};
use axvm_instructions::{instruction::Instruction, program::PC_BITS};
use num_traits::WrappingSub;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{Rv32JalrChip, Rv32JalrCoreChip};
use crate::{
    arch::{
        instructions::{
            Rv32JalrOpcode::{self, *},
            UsizeOpcode,
        },
        testing::VmChipTestBuilder,
        VmAdapterChip,
    },
    rv32im::{
        adapters::{compose, Rv32JalrAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
        jalr::{run_jalr, Rv32JalrCoreCols},
    },
    system::vm::chip_set::BYTE_XOR_BUS,
};

const IMM_BITS: usize = 16;
type F = BabyBear;

fn into_limbs(num: u32) -> [u32; 4] {
    array::from_fn(|i| (num >> (8 * i)) & 255)
}
fn sign_extend(num: u32) -> u32 {
    if num & 0x8000 != 0 {
        num | 0xffff0000
    } else {
        num
    }
}
fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32JalrChip<F>,
    rng: &mut StdRng,
    opcode: Rv32JalrOpcode,
    initial_imm: Option<u32>,
    initial_pc: Option<u32>,
    rs1: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
) {
    let imm = initial_imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_ext = sign_extend(imm);
    let a = rng.gen_range(0..32) << 2;
    let b = rng.gen_range(1..32) << 2;
    let to_pc = rng.gen_range(0..(1 << PC_BITS));

    let rs1 = rs1.unwrap_or(into_limbs(to_pc.wrapping_sub(&imm_ext)));
    let rs1 = rs1.map(F::from_canonical_u32);

    tester.write(1, b, rs1);

    tester.execute_with_pc(
        chip,
        Instruction::from_usize(
            opcode as usize + Rv32JalrOpcode::default_offset(),
            [a, b, imm as usize, 1, 0, (a != 0) as usize, 0],
        ),
        initial_pc.unwrap_or(rng.gen_range(0..(1 << PC_BITS))),
    );
    let initial_pc = tester.execution.last_from_pc().as_canonical_u32();
    let final_pc = tester.execution.last_to_pc().as_canonical_u32();

    let rs1 = compose(rs1);

    let (next_pc, rd_data) = run_jalr(opcode, initial_pc, imm_ext, rs1);
    let rd_data = if a == 0 { [0; 4] } else { rd_data };

    assert_eq!(next_pc, final_pc);
    assert_eq!(rd_data.map(F::from_canonical_u32), tester.read::<4>(1, a));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn rand_jalr_test() {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();

    let adapter = Rv32JalrAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let inner = Rv32JalrCoreChip::new(
        xor_lookup_chip.clone(),
        range_checker_chip.clone(),
        Rv32JalrOpcode::default_offset(),
    );
    let mut chip = Rv32JalrChip::<F>::new(adapter, inner, tester.memory_controller());

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, JALR, None, None, None);
    }

    drop(range_checker_chip);
    let tester = tester.build().load(chip).load(xor_lookup_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

///////////////////////////////////////////////////////////////////////////////////////
/// NEGATIVE TESTS
///
/// Given a fake trace of a single operation, setup a chip and run the test. We replace
/// the write part of the trace and check that the core chip throws the expected error.
/// A dummy adaptor is used so memory interactions don't indirectly cause false passes.
///////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_jalr_test(
    opcode: Rv32JalrOpcode,
    initial_pc: Option<u32>,
    initial_rs1: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
    rd_data: Option<[u32; RV32_REGISTER_NUM_LIMBS - 1]>,
    rs1_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    to_pc_least_sig_bit: Option<u32>,
    to_pc_limbs: Option<[u32; 2]>,
    imm_sign: Option<u32>,
    expected_error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();

    let adapter = Rv32JalrAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let adapter_width = BaseAir::<F>::width(adapter.air());
    let inner = Rv32JalrCoreChip::new(
        xor_lookup_chip.clone(),
        range_checker_chip.clone(),
        Rv32JalrOpcode::default_offset(),
    );
    let mut chip = Rv32JalrChip::<F>::new(adapter, inner, tester.memory_controller());

    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        opcode,
        imm,
        initial_pc,
        initial_rs1,
    );

    let jalr_trace_width = chip.trace_width();
    let mut chip_input = chip.generate_air_proof_input();
    let jalr_trace = chip_input.raw.common_main.as_mut().unwrap();
    {
        let mut trace_row = jalr_trace.row_slice(0).to_vec();

        let (_, core_row) = trace_row.split_at_mut(adapter_width);

        let core_cols: &mut Rv32JalrCoreCols<F> = core_row.borrow_mut();

        if let Some(data) = rd_data {
            core_cols.rd_data = data.map(F::from_canonical_u32);
        }

        if let Some(data) = rs1_data {
            core_cols.rs1_data = data.map(F::from_canonical_u32);
        }

        if let Some(data) = to_pc_least_sig_bit {
            core_cols.to_pc_least_sig_bit = F::from_canonical_u32(data);
        }

        if let Some(data) = to_pc_limbs {
            core_cols.to_pc_limbs = data.map(F::from_canonical_u32);
        }

        if let Some(data) = imm_sign {
            core_cols.imm_sign = F::from_canonical_u32(data);
        }

        *jalr_trace = RowMajorMatrix::new(trace_row, jalr_trace_width);
    }

    drop(range_checker_chip);
    disable_debug_builder();
    let tester = tester
        .build()
        .load_air_proof_input(chip_input)
        .load(xor_lookup_chip)
        .finalize();
    tester.simple_test_with_expected_error(expected_error);
}

#[test]
fn invalid_cols_negative_tests() {
    run_negative_jalr_test(
        JALR,
        None,
        None,
        Some(15362),
        None,
        None,
        None,
        None,
        Some(1),
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_jalr_test(
        JALR,
        None,
        Some([23, 154, 67, 28]),
        Some(42512),
        None,
        None,
        Some(0),
        None,
        None,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn overflow_negative_tests() {
    run_negative_jalr_test(
        JALR,
        Some(251),
        None,
        None,
        Some([1, 0, 0]),
        None,
        None,
        None,
        None,
        VerificationError::NonZeroCumulativeSum,
    );

    run_negative_jalr_test(
        JALR,
        None,
        Some([0, 0, 0, 0]),
        Some((1 << 15) - 2),
        None,
        None,
        None,
        Some([
            (F::neg_one() * F::from_canonical_u32((1 << 14) + 1)).as_canonical_u32(),
            1,
        ]),
        None,
        VerificationError::NonZeroCumulativeSum,
    );
}
///////////////////////////////////////////////////////////////////////////////////////
/// SANITY TESTS
///
/// Ensure that solve functions produce the correct results.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn execute_roundtrip_sanity_test() {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));
    let mut tester = VmChipTestBuilder::default();
    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();

    let adapter = Rv32JalrAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let inner = Rv32JalrCoreChip::new(
        xor_lookup_chip,
        range_checker_chip,
        Rv32JalrOpcode::default_offset(),
    );
    let mut chip = Rv32JalrChip::<F>::new(adapter, inner, tester.memory_controller());

    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, JALR, None, None, None);
    }
}

#[test]
fn run_jalr_sanity_test() {
    let opcode = JALR;
    let initial_pc = 789456120;
    let imm = -1235_i32 as u32;
    let rs1 = 736482910;
    let (next_pc, rd_data) = run_jalr(opcode, initial_pc, imm, rs1);
    assert_eq!(next_pc, 736481674);
    assert_eq!(rd_data, [252, 36, 14, 47]);
}
