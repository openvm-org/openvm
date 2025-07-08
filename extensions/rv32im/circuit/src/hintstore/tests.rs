use std::{array, borrow::BorrowMut};

use openvm_circuit::arch::{
    testing::{
        memory::{gen_reg_pointer, gen_reg_pointer_excluding},
        VmChipTestBuilder, BITWISE_OP_LOOKUP_BUS,
    },
    DenseRecordArena, ExecutionBridge, InsExecutorE1, InstructionExecutor, NewVmChipWrapper,
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, SharedBitwiseOperationLookupChip,
};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
    VmOpcode,
};
use openvm_rv32im_transpiler::Rv32HintStoreOpcode::{self, *};
use openvm_stark_backend::{
    p3_field::FieldAlgebra,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};

use super::{Rv32HintStoreAir, Rv32HintStoreChip, Rv32HintStoreCols, Rv32HintStoreStep};
use crate::{
    adapters::decompose, hintstore::Rv32HintStoreLayout, test_utils::get_verification_error,
};

type F = BabyBear;
const MAX_INS_CAPACITY: usize = 4096;

fn create_test_chip(
    tester: &mut VmChipTestBuilder<F>,
) -> (
    Rv32HintStoreChip<F>,
    SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
) {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Rv32HintStoreChip::<F>::new(
        Rv32HintStoreAir::new(
            ExecutionBridge::new(tester.execution_bus(), tester.program_bus()),
            tester.memory_bridge(),
            bitwise_chip.bus(),
            0,
            tester.address_bits(),
        ),
        Rv32HintStoreStep::new(bitwise_chip.clone(), tester.address_bits(), 0),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    (chip, bitwise_chip)
}

fn set_and_execute<E: InstructionExecutor<F> + InsExecutorE1<F>>(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut E,
    rng: &mut StdRng,
    opcode: Rv32HintStoreOpcode,
) {
    let mem_ptr = rng
        .gen_range(0..(1 << (tester.memory_controller().mem_config().pointer_max_bits - 2)))
        << 2;
    let b = gen_reg_pointer(rng);

    tester.write(1, b, decompose(mem_ptr));

    let read_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));
    for data in read_data {
        tester.streams.hint_stream.push_back(data);
    }

    tester.execute(
        chip,
        &Instruction::from_usize(VmOpcode::from_usize(opcode as usize), [0, b, 0, 1, 2]),
    );

    let write_data = read_data;
    assert_eq!(write_data, tester.read::<4>(2, mem_ptr as usize));
}

fn set_and_execute_buffer(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32HintStoreChip<F>,
    rng: &mut StdRng,
    opcode: Rv32HintStoreOpcode,
) {
    let mem_ptr = rng
        .gen_range(0..(1 << (tester.memory_controller().mem_config().pointer_max_bits - 2)))
        << 2;
    let b = gen_reg_pointer(rng);

    tester.write(1, b, decompose(mem_ptr));

    let num_words = rng.gen_range(1..28);
    let a = gen_reg_pointer_excluding(rng, &[b]);
    tester.write(1, a, decompose(num_words));

    let data: Vec<[F; RV32_REGISTER_NUM_LIMBS]> = (0..num_words)
        .map(|_| array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS)))))
        .collect();
    for i in 0..num_words {
        for datum in data[i as usize] {
            tester.streams.hint_stream.push_back(datum);
        }
    }

    tester.execute(
        chip,
        &Instruction::from_usize(VmOpcode::from_usize(opcode as usize), [a, b, 0, 1, 2]),
    );

    for i in 0..num_words {
        assert_eq!(
            data[i as usize],
            tester.read::<4>(2, mem_ptr as usize + (i as usize * RV32_REGISTER_NUM_LIMBS))
        );
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test]
fn rand_hintstore_test() {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let (mut chip, bitwise_chip) = create_test_chip(&mut tester);
    let num_ops: usize = 100;
    for _ in 0..num_ops {
        if rng.gen_bool(0.5) {
            set_and_execute(&mut tester, &mut chip, &mut rng, HINT_STOREW);
        } else {
            set_and_execute_buffer(&mut tester, &mut chip, &mut rng, HINT_BUFFER);
        }
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

#[allow(clippy::too_many_arguments)]
fn run_negative_hintstore_test(
    opcode: Rv32HintStoreOpcode,
    prank_data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    interaction_error: bool,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();
    let (mut chip, bitwise_chip) = create_test_chip(&mut tester);

    set_and_execute(&mut tester, &mut chip, &mut rng, opcode);

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).to_vec();
        let cols: &mut Rv32HintStoreCols<F> = trace_row.as_mut_slice().borrow_mut();
        if let Some(data) = prank_data {
            cols.data = data.map(F::from_canonical_u32);
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
fn negative_hintstore_tests() {
    run_negative_hintstore_test(HINT_STOREW, Some([92, 187, 45, 280]), true);
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

    let (mut chip, _) = create_test_chip(&mut tester);

    let num_ops: usize = 10;
    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut chip, &mut rng, HINT_STOREW);
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// DENSE TESTS
///
/// Ensure that the chip works as expected with dense records.
/// We first execute some instructions with a [DenseRecordArena] and transfer the records
/// to a [MatrixRecordArena]. After transferring we generate the trace and make sure that
/// all the constraints pass.
///////////////////////////////////////////////////////////////////////////////////////
type Rv32HintStoreChipDense =
    NewVmChipWrapper<F, Rv32HintStoreAir, Rv32HintStoreStep, DenseRecordArena>;

fn create_test_chip_dense(tester: &mut VmChipTestBuilder<F>) -> Rv32HintStoreChipDense {
    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = SharedBitwiseOperationLookupChip::<RV32_CELL_BITS>::new(bitwise_bus);

    let mut chip = Rv32HintStoreChipDense::new(
        Rv32HintStoreAir::new(
            ExecutionBridge::new(tester.execution_bus(), tester.program_bus()),
            tester.memory_bridge(),
            bitwise_chip.bus(),
            0,
            tester.address_bits(),
        ),
        Rv32HintStoreStep::new(bitwise_chip.clone(), tester.address_bits(), 0),
        tester.memory_helper(),
    );

    chip.set_trace_buffer_height(MAX_INS_CAPACITY);
    chip
}

// #[test]
// fn dense_record_arena_test() {
//     let mut rng = create_seeded_rng();
//     let mut tester = VmChipTestBuilder::default();
//     let (mut sparse_chip, bitwise_chip) = create_test_chip(&mut tester);

//     {
//         let mut dense_chip = create_test_chip_dense(&mut tester);

//         let num_ops: usize = 100;
//         for _ in 0..num_ops {
//             set_and_execute(&mut tester, &mut dense_chip, &mut rng, HINT_STOREW);
//         }

//         let mut record_interpreter = dense_chip
//             .arena
//             .get_record_seeker::<_, Rv32HintStoreLayout>();
//         record_interpreter.transfer_to_matrix_arena(&mut sparse_chip.arena);
//     }

//     let tester = tester
//         .build()
//         .load(sparse_chip)
//         .load(bitwise_chip)
//         .finalize();
//     tester.simple_test().expect("Verification failed");
// }
