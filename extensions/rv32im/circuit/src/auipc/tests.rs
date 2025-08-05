use std::{borrow::BorrowMut, sync::Arc};

use openvm_circuit::arch::{
    testing::{TestChipHarness, VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS},
    Arena, DenseRecordArena, EmptyAdapterCoreLayout, PreflightExecutor, VmAirWrapper,
    VmChipWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
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
        Rv32RdWriteAdapterAir, Rv32RdWriteAdapterFiller, Rv32RdWriteAdapterRecord,
        Rv32RdWriteAdapterStep, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS,
    },
    test_utils::get_verification_error,
    Rv32AuipcAir, Rv32AuipcCoreRecord, Rv32AuipcFiller,
};

const IMM_BITS: usize = 24;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness<RA> = TestChipHarness<F, Rv32AuipcStep, Rv32AuipcAir, Rv32AuipcChip<F>, RA>;

fn create_test_chip<RA: Arena>(
    tester: &VmChipTestBuilder<F>,
) -> (
    Harness<RA>,
    (
        BitwiseOperationLookupAir<RV32_CELL_BITS>,
        SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    ),
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let air = VmAirWrapper::new(
        Rv32RdWriteAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
        Rv32AuipcCoreAir::new(bitwise_bus),
    );
    let executor = Rv32AuipcStep::new(Rv32RdWriteAdapterStep::new());
    let chip = VmChipWrapper::<F, _>::new(
        Rv32AuipcFiller::new(Rv32RdWriteAdapterFiller::new(), bitwise_chip.clone()),
        tester.memory_helper(),
    );
    let harness = Harness::<RA>::with_capacity(executor, air, chip, MAX_INS_CAPACITY);

    (harness, (bitwise_chip.air, bitwise_chip))
}

fn set_and_execute<RA: Arena>(
    tester: &mut VmChipTestBuilder<F>,
    harness: &mut Harness<RA>,
    rng: &mut StdRng,
    opcode: Rv32AuipcOpcode,
    imm: Option<u32>,
    initial_pc: Option<u32>,
) where
    Rv32AuipcStep: PreflightExecutor<F, RA>,
{
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS))) as usize;
    let a = rng.gen_range(0..32) << 2;

    tester.execute_with_pc(
        harness,
        &Instruction::from_usize(opcode.global_opcode(), [a, 0, imm, 1, 0]),
        initial_pc.unwrap_or(rng.gen_range(0..(1 << PC_BITS))),
    );
    let initial_pc = tester.execution.last_from_pc().as_canonical_u32();
    let rd_data = run_auipc(initial_pc, imm as u32);
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
    let (mut harness, bitwise) = create_test_chip(&tester);

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut harness, &mut rng, AUIPC, None, None);
    }
    let tester = tester
        .build()
        .load(harness)
        .load_periphery(bitwise)
        .finalize();
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
    let (mut harness, bitwise) = create_test_chip(&tester);

    set_and_execute(
        &mut tester,
        &mut harness,
        &mut rng,
        opcode,
        initial_imm,
        initial_pc,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
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
        .load_and_prank_trace(harness, modify_trace)
        .load_periphery(bitwise)
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
    let initial_pc = 234567890;
    let imm = 11302451;
    let rd_data = run_auipc(initial_pc, imm);

    assert_eq!(rd_data, [210, 107, 113, 186]);
}

// ////////////////////////////////////////////////////////////////////////////////////
// DENSE TESTS

// Ensure that the chip works as expected with dense records.
// We first execute some instructions with a [DenseRecordArena] and transfer the records
// to a [MatrixRecordArena]. After transferring we generate the trace and make sure that
// all the constraints pass.
// ////////////////////////////////////////////////////////////////////////////////////

#[test]
fn dense_record_arena_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut sparse_harness, bitwise) = create_test_chip(&tester);

    {
        let mut dense_harness = create_test_chip::<DenseRecordArena>(&tester).0;

        let num_ops: usize = 100;
        for _ in 0..num_ops {
            set_and_execute(&mut tester, &mut dense_harness, &mut rng, AUIPC, None, None);
        }

        type Record<'a> = (
            &'a mut Rv32RdWriteAdapterRecord,
            &'a mut Rv32AuipcCoreRecord,
        );

        let mut record_interpreter = dense_harness.arena.get_record_seeker::<Record, _>();
        record_interpreter.transfer_to_matrix_arena(
            &mut sparse_harness.arena,
            EmptyAdapterCoreLayout::<F, Rv32RdWriteAdapterStep>::new(),
        );
    }

    let tester = tester
        .build()
        .load(sparse_harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}
