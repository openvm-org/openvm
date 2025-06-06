use std::borrow::BorrowMut;

use openvm_circuit::arch::{
    testing::{VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    VmAirWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, program::PC_BITS, LocalOpcode};
use openvm_rv32im_transpiler::Rv32AuipcOpcode::{self, *};
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

use super::{run_auipc, Rv32AuipcChip, Rv32AuipcCoreAir, Rv32AuipcCoreCols, Rv32AuipcStep};
use crate::{
    adapters::{
        Rv32RdWriteAdapterAir, Rv32RdWriteAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    test_utils::get_verification_error,
};

const IMM_BITS: usize = 24;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;

fn create_test_chip(
    tester: &VmChipTestBuilder<F>,
) -> (
    Rv32AuipcChip<F>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Rv32AuipcChip::<F>::new(
        VmAirWrapper::new(
            Rv32RdWriteAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
            Rv32AuipcCoreAir::new(bitwise_bus),
        ),
        Rv32AuipcStep::new(Rv32RdWriteAdapterStep::new(), bitwise_chip.clone()),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    (chip, bitwise_chip)
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32AuipcChip<F>,
    rng: &mut StdRng,
    opcode: Rv32AuipcOpcode,
    imm: Option<u32>,
    initial_pc: Option<u32>,
) {
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS))) as usize;
    let a = rng.gen_range(0..32) << 2;

    tester.execute_with_pc(
        chip,
        &Instruction::from_usize(opcode.global_opcode(), [a, 0, imm, 1, 0]),
        initial_pc.unwrap_or(rng.gen_range(0..(1 << PC_BITS))),
    );
    let initial_pc = tester.execution.last_from_pc().as_canonical_u32();
    let rd_data = run_auipc(opcode, initial_pc, imm as u32);
    assert_eq!(rd_data.map(F::from_canonical_u8), tester.read::<4>(1, a));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_auipc_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chip(&tester);

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, AUIPC, None, None);
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
struct AuipcPrankValues {
    pub rd_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    pub imm_limbs: Option<[u32; RV32_REGISTER_NUM_LIMBS - 1]>,
    pub pc_limbs: Option<[u32; RV32_REGISTER_NUM_LIMBS - 2]>,
}

fn run_negative_auipc_test(
    opcode: Rv32AuipcOpcode,
    initial_imm: Option<u32>,
    initial_pc: Option<u32>,
    prank_vals: AuipcPrankValues,
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
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut trace_row = trace.row_slice(0).to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core_cols: &mut Rv32AuipcCoreCols<F> = core_row.borrow_mut();

        if let Some(data) = prank_vals.rd_data {
            core_cols.rd_data = data.map(F::from_canonical_u32);
        }
        if let Some(data) = prank_vals.imm_limbs {
            core_cols.imm_limbs = data.map(F::from_canonical_u32);
        }
        if let Some(data) = prank_vals.pc_limbs {
            core_cols.pc_limbs = data.map(F::from_canonical_u32);
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
fn invalid_limb_negative_tests() {
    run_negative_auipc_test(
        AUIPC,
        Some(9722891),
        None,
        AuipcPrankValues {
            imm_limbs: Some([107, 46, 81]),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        Some(0),
        Some(2110400),
        AuipcPrankValues {
            rd_data: Some([194, 51, 32, 240]),
            pc_limbs: Some([51, 32]),
            ..Default::default()
        },
        true,
    );
    run_negative_auipc_test(
        AUIPC,
        None,
        None,
        AuipcPrankValues {
            pc_limbs: Some([206, 166]),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        None,
        None,
        AuipcPrankValues {
            rd_data: Some([30, 92, 82, 132]),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        None,
        Some(876487877),
        AuipcPrankValues {
            rd_data: Some([197, 202, 49, 70]),
            imm_limbs: Some([166, 243, 17]),
            pc_limbs: Some([36, 62]),
        },
        true,
    );
}

#[test]
fn overflow_negative_tests() {
    run_negative_auipc_test(
        AUIPC,
        Some(256264),
        None,
        AuipcPrankValues {
            imm_limbs: Some([3592, 219, 3]),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        None,
        None,
        AuipcPrankValues {
            pc_limbs: Some([0, 0]),
            ..Default::default()
        },
        false,
    );
    run_negative_auipc_test(
        AUIPC,
        Some(255),
        None,
        AuipcPrankValues {
            imm_limbs: Some([F::NEG_ONE.as_canonical_u32(), 1, 0]),
            ..Default::default()
        },
        true,
    );
    run_negative_auipc_test(
        AUIPC,
        Some(0),
        Some(255),
        AuipcPrankValues {
            rd_data: Some([F::NEG_ONE.as_canonical_u32(), 1, 0, 0]),
            imm_limbs: Some([0, 0, 0]),
            pc_limbs: Some([1, 0]),
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
fn run_auipc_sanity_test() {
    let opcode = AUIPC;
    let initial_pc = 234567890;
    let imm = 11302451;
    let rd_data = run_auipc(opcode, initial_pc, imm);

    assert_eq!(rd_data, [210, 107, 113, 186]);
}
