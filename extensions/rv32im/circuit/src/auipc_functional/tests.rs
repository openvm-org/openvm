use std::sync::Arc;

use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{Rng, rngs::StdRng};

use openvm_circuit::arch::{Arena, CustomBorrow, DenseRecordArena, EmptyMultiRowMetadata, InstructionExecutor, MultiRowLayout, testing::{BITWISE_OP_LOOKUP_BUS, TestChipHarness, VmChipTestBuilder}, VmChipWrapper};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupAir, BitwiseOperationLookupBus, BitwiseOperationLookupChip,
    SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode, program::PC_BITS};
use openvm_rv32im_transpiler::Rv32AuipcOpcode::{self, *};

use crate::{adapters::RV32_CELL_BITS, run_auipc};

use super::{AuipcAir, AuipcChip, AuipcFiller, AuipcRecord, AuipcStep};

const IMM_BITS: usize = 24;
const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness<RA> = TestChipHarness<F, AuipcStep, AuipcAir, AuipcChip<F>, RA>;

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

    let air = AuipcAir {
        custom_bus_bitwise: bitwise_bus.inner.index,
        custom_bus_memory: tester.memory_bridge().memory_bus().index(),
        custom_bus_range_check: tester.memory_bridge().range_bus().index(),
        custom_bus_program: tester.execution_bridge().program_bus.index(),
        custom_bus_execution: tester.execution_bridge().execution_bus.index(),
    };

    let executor = AuipcStep::new();
    let chip = VmChipWrapper::<F, _>::new(
        AuipcFiller::new(bitwise_chip.clone(), tester.memory_helper().range_checker),
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
    AuipcStep: InstructionExecutor<F, RA>,
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

        type Record<'a> = &'a mut AuipcRecord;

        let mut record_interpreter = dense_harness.arena.get_record_seeker::<Record, MultiRowLayout<EmptyMultiRowMetadata>>();
        record_interpreter.transfer_to_matrix_arena(
            &mut sparse_harness.arena,
        );
    }

    let tester = tester
        .build()
        .load(sparse_harness)
        .load_periphery(bitwise)
        .finalize();
    tester.simple_test().expect("Verification failed");
}
