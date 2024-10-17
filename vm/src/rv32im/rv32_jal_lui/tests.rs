use std::{borrow::BorrowMut, sync::Arc};

use afs_primitives::xor::lookup::XorLookupChip;
use afs_stark_backend::{
    utils::disable_debug_builder, verifier::VerificationError, Chip, ChipUsageGetter,
};
use ax_sdk::utils::create_seeded_rng;
use axvm_instructions::UsizeOpcode;
use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rand::{rngs::StdRng, Rng};

use super::{solve_jal_lui, Rv32JalLuiChip, Rv32JalLuiCoreChip};
use crate::{
    arch::{
        instructions::Rv32JalLuiOpcode::{self, *},
        testing::VmChipTestBuilder,
        VmAdapterChip,
    },
    kernels::core::BYTE_XOR_BUS,
    rv32im::{
        adapters::{
            Rv32CondRdWriteAdapter, Rv32CondRdWriteAdapterCols, PC_BITS, RV32_CELL_BITS,
            RV32_REGISTER_NUM_LANES, RV_IS_TYPE_IMM_BITS,
        },
        rv32_jal_lui::Rv32JalLuiCols,
    },
    system::program::Instruction,
};

const IMM_BITS: usize = 20;
const LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;
type F = BabyBear;

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32JalLuiChip<F>,
    rng: &mut StdRng,
    opcode: Rv32JalLuiOpcode,
    imm: Option<i32>,
    initial_pc: Option<u32>,
) {
    let imm: i32 = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm = match opcode {
        JAL => ((imm >> 1) << 2) - (1 << IMM_BITS),
        LUI => imm,
    };

    let a = rng.gen_range(1..32) << 2;
    let need_write = a != 0 || opcode == LUI;

    tester.execute_with_pc(
        chip,
        Instruction::large_from_isize(
            opcode as usize + Rv32JalLuiOpcode::default_offset(),
            a as isize,
            0,
            imm as isize,
            1,
            0,
            need_write as isize,
            0,
        ),
        initial_pc.unwrap_or(rng.gen_range(imm.unsigned_abs()..(1 << PC_BITS))),
    );

    let initial_pc = tester
        .execution
        .records
        .last()
        .unwrap()
        .initial_state
        .pc
        .as_canonical_u32();
    let final_pc = tester
        .execution
        .records
        .last()
        .unwrap()
        .final_state
        .pc
        .as_canonical_u32();

    let (next_pc, rd_data) = solve_jal_lui(opcode, initial_pc, imm);
    let rd_data = if need_write { rd_data } else { [0; 4] };

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
fn rand_jal_lui_test() {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));

    let mut tester = VmChipTestBuilder::default();
    let adapter = Rv32CondRdWriteAdapter::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let core = Rv32JalLuiCoreChip::new(xor_lookup_chip.clone(), Rv32JalLuiOpcode::default_offset());
    let mut chip = Rv32JalLuiChip::<F>::new(adapter, core, tester.memory_controller());

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, JAL, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, LUI, None, None);
    }

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
fn run_negative_jal_lui_test(
    opcode: Rv32JalLuiOpcode,
    initial_imm: Option<i32>,
    initial_pc: Option<u32>,
    rd_data: Option<[u32; RV32_REGISTER_NUM_LANES]>,
    imm: Option<i32>,
    is_jal: Option<bool>,
    is_lui: Option<bool>,
    need_write: Option<bool>,
    expected_error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let xor_lookup_chip = Arc::new(XorLookupChip::<RV32_CELL_BITS>::new(BYTE_XOR_BUS));

    let mut tester = VmChipTestBuilder::default();
    let adapter = Rv32CondRdWriteAdapter::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let adapter_width = BaseAir::<F>::width(adapter.air());
    let core = Rv32JalLuiCoreChip::new(xor_lookup_chip.clone(), Rv32JalLuiOpcode::default_offset());
    let mut chip = Rv32JalLuiChip::<F>::new(adapter, core, tester.memory_controller());

    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        opcode,
        initial_imm,
        initial_pc,
    );

    let jal_lui_trace_width = chip.trace_width();
    let mut chip_input = chip.generate_air_proof_input();
    let jal_lui_trace = chip_input.raw.common_main.as_mut().unwrap();
    {
        let mut trace_row = jal_lui_trace.row_slice(0).to_vec();

        let (adapter_row, core_row) = trace_row.split_at_mut(adapter_width);

        let adapter_cols: &mut Rv32CondRdWriteAdapterCols<F> = adapter_row.borrow_mut();
        let core_cols: &mut Rv32JalLuiCols<F> = core_row.borrow_mut();

        if let Some(data) = rd_data {
            core_cols.rd_data = data.map(F::from_canonical_u32);
        }

        if let Some(imm) = imm {
            core_cols.imm = if imm < 0 {
                F::neg_one() * F::from_canonical_u32((-imm) as u32)
            } else {
                F::from_canonical_u32(imm as u32)
            };
        }
        if let Some(is_jal) = is_jal {
            core_cols.is_jal = F::from_bool(is_jal);
        }
        if let Some(is_lui) = is_lui {
            core_cols.is_lui = F::from_bool(is_lui);
        }

        if let Some(need_write) = need_write {
            adapter_cols.need_write = F::from_bool(need_write);
        }

        *jal_lui_trace = RowMajorMatrix::new(trace_row, jal_lui_trace_width);
    }

    disable_debug_builder();
    let tester = tester
        .build()
        .load_air_proof_input(chip_input)
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
fn opcode_flag_negative_test() {
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        None,
        None,
        Some(false),
        Some(true),
        None,
        VerificationError::OodEvaluationMismatch,
    );
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        None,
        None,
        Some(false),
        Some(false),
        Some(false),
        VerificationError::NonZeroCumulativeSum,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        None,
        None,
        Some(true),
        Some(false),
        None,
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn overflow_negative_tests() {
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        Some([LIMB_MAX, LIMB_MAX, LIMB_MAX, LIMB_MAX]),
        None,
        None,
        None,
        None,
        VerificationError::OodEvaluationMismatch,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        Some([LIMB_MAX, LIMB_MAX, LIMB_MAX, LIMB_MAX]),
        None,
        None,
        None,
        None,
        VerificationError::OodEvaluationMismatch,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        Some([0, LIMB_MAX, LIMB_MAX, LIMB_MAX + 1]),
        None,
        None,
        None,
        None,
        VerificationError::OodEvaluationMismatch,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        None,
        Some(-1),
        None,
        None,
        None,
        VerificationError::OodEvaluationMismatch,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        None,
        Some(-28),
        None,
        None,
        None,
        VerificationError::OodEvaluationMismatch,
    );
    run_negative_jal_lui_test(
        JAL,
        None,
        Some(251),
        Some([F::neg_one().as_canonical_u32(), 1, 0, 0]),
        None,
        None,
        None,
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
    let adapter = Rv32CondRdWriteAdapter::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
    );
    let core = Rv32JalLuiCoreChip::new(xor_lookup_chip, Rv32JalLuiOpcode::default_offset());
    let mut chip = Rv32JalLuiChip::<F>::new(adapter, core, tester.memory_controller());
    let num_tests: usize = 10;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, JAL, None, None);
        set_and_execute(&mut tester, &mut chip, &mut rng, LUI, None, None);
    }

    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        LUI,
        Some((1 << IMM_BITS) - 1),
        None,
    );
    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        JAL,
        Some((1 << RV_IS_TYPE_IMM_BITS) - 1),
        None,
    );
}

#[test]
fn solve_jal_sanity_test() {
    let opcode = JAL;
    let initial_pc = 28120;
    let imm = -2048;
    let (next_pc, rd_data) = solve_jal_lui(opcode, initial_pc, imm);
    assert_eq!(next_pc, 26072);
    assert_eq!(rd_data, [220, 109, 0, 0]);
}

#[test]
fn solve_lui_sanity_test() {
    let opcode = LUI;
    let initial_pc = 456789120;
    let imm = 853679;
    let (next_pc, rd_data) = solve_jal_lui(opcode, initial_pc, imm);
    assert_eq!(next_pc, 456789124);
    assert_eq!(rd_data, [0, 240, 106, 208]);
}
