use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    VmAirWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_rv32im_transpiler::Rv32JalLuiOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use super::{run_jal_lui, Rv32JalLuiChip, Rv32JalLuiCoreAir, Rv32JalLuiStep};
use crate::{
    adapters::{
        Rv32CondRdWriteAdapterAir, Rv32CondRdWriteAdapterCols, Rv32CondRdWriteAdapterStep,
        Rv32RdWriteAdapterAir, Rv32RdWriteAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
        RV_IS_TYPE_IMM_BITS,
    },
    jal_lui::Rv32JalLuiCoreCols,
    test_utils::get_verification_error,
};

const IMM_BITS: usize = 20;
const LIMB_MAX: u32 = (1 << RV32_CELL_BITS) - 1;
const MAX_INS_CAPACITY: usize = 128;

type F = BabyBear;

fn create_test_chip(
    tester: &VmChipTestBuilder<F>,
) -> (
    Rv32JalLuiChip<F>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Rv32JalLuiChip::<F>::new(
        VmAirWrapper::new(
            Rv32CondRdWriteAdapterAir::new(Rv32RdWriteAdapterAir::new(
                tester.memory_bridge(),
                tester.execution_bridge(),
            )),
            Rv32JalLuiCoreAir::new(bitwise_bus),
        ),
        Rv32JalLuiStep::new(
            Rv32CondRdWriteAdapterStep::new(Rv32RdWriteAdapterStep::new()),
            bitwise_chip.clone(),
        ),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    (chip, bitwise_chip)
}

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

    let a = rng.gen_range((opcode == LUI) as usize..32) << 2;
    let needs_write = a != 0 || opcode == LUI;

    tester.execute_with_pc(
        chip,
        &Instruction::large_from_isize(
            opcode.global_opcode(),
            a as isize,
            0,
            imm as isize,
            1,
            0,
            needs_write as isize,
            0,
        ),
        initial_pc.unwrap_or(rng.gen_range(imm.unsigned_abs()..(1 << PC_BITS))),
    );
    let initial_pc = tester.execution.last_from_pc().as_canonical_u32();
    let final_pc = tester.execution.last_to_pc().as_canonical_u32();

    let (next_pc, rd_data) = run_jal_lui(opcode, initial_pc, imm);
    let rd_data = if needs_write { rd_data } else { [0; 4] };

    assert_eq!(next_pc, final_pc);
    assert_eq!(rd_data.map(F::from_canonical_u8), tester.read::<4>(1, a));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test_case(JAL, 100)]
#[test_case(LUI, 100)]
fn rand_jal_lui_test(opcode: Rv32JalLuiOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chip(&tester);

    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut chip, &mut rng, opcode, None, None);
    }

    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}
//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default, PartialEq)]
struct JalLuiPrankValues {
    pub rd_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    pub imm: Option<i32>,
    pub is_jal: Option<bool>,
    pub is_lui: Option<bool>,
    pub needs_write: Option<bool>,
}

#[allow(clippy::too_many_arguments)]
fn run_negative_jal_lui_test(
    opcode: Rv32JalLuiOpcode,
    initial_imm: Option<i32>,
    initial_pc: Option<u32>,
    prank_vals: JalLuiPrankValues,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut chip,
        &mut rng,
        opcode,
        initial_imm,
        initial_pc,
    );

    let adapter_width = BaseAir::<F>::width(&chip.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).to_vec();
        let (adapter_row, core_row) = trace_row.split_at_mut(adapter_width);
        let adapter_cols: &mut Rv32CondRdWriteAdapterCols<F> = adapter_row.borrow_mut();
        let core_cols: &mut Rv32JalLuiCoreCols<F> = core_row.borrow_mut();

        if let Some(data) = prank_vals.rd_data {
            core_cols.rd_data = data.map(F::from_canonical_u32);
        }
        if let Some(imm) = prank_vals.imm {
            core_cols.imm = if imm < 0 {
                F::NEG_ONE * F::from_canonical_u32((-imm) as u32)
            } else {
                F::from_canonical_u32(imm as u32)
            };
        }
        if let Some(is_jal) = prank_vals.is_jal {
            core_cols.is_jal = F::from_bool(is_jal);
        }
        if let Some(is_lui) = prank_vals.is_lui {
            core_cols.is_lui = F::from_bool(is_lui);
        }
        if let Some(needs_write) = prank_vals.needs_write {
            adapter_cols.needs_write = F::from_bool(needs_write);
        }

        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test_with_expected_error(get_verification_error(interaction_error));
}

#[test]
fn opcode_flag_negative_test() {
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        JalLuiPrankValues {
            is_jal: Some(false),
            is_lui: Some(true),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        JalLuiPrankValues {
            is_jal: Some(false),
            is_lui: Some(false),
            needs_write: Some(false),
            ..Default::default()
        },
        true,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            is_jal: Some(true),
            is_lui: Some(false),
            ..Default::default()
        },
        false,
    );
}

#[test]
fn overflow_negative_tests() {
    run_negative_jal_lui_test(
        JAL,
        None,
        None,
        JalLuiPrankValues {
            rd_data: Some([LIMB_MAX, LIMB_MAX, LIMB_MAX, LIMB_MAX]),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            rd_data: Some([LIMB_MAX, LIMB_MAX, LIMB_MAX, LIMB_MAX]),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            rd_data: Some([0, LIMB_MAX, LIMB_MAX, LIMB_MAX + 1]),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            imm: Some(-1),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        LUI,
        None,
        None,
        JalLuiPrankValues {
            imm: Some(-28),
            ..Default::default()
        },
        false,
    );
    run_negative_jal_lui_test(
        JAL,
        None,
        Some(251),
        JalLuiPrankValues {
            rd_data: Some([F::NEG_ONE.as_canonical_u32(), 1, 0, 0]),
            ..Default::default()
        },
        true,
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
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, _) = create_test_chip(&tester);

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
fn run_jal_sanity_test() {
    let opcode = JAL;
    let initial_pc = 28120;
    let imm = -2048;
    let (next_pc, rd_data) = run_jal_lui(opcode, initial_pc, imm);
    assert_eq!(next_pc, 26072);
    assert_eq!(rd_data, [220, 109, 0, 0]);
}

#[test]
fn run_lui_sanity_test() {
    let opcode = LUI;
    let initial_pc = 456789120;
    let imm = 853679;
    let (next_pc, rd_data) = run_jal_lui(opcode, initial_pc, imm);
    assert_eq!(next_pc, 456789124);
    assert_eq!(rd_data, [0, 240, 106, 208]);
}
