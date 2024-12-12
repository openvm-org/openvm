use std::{array, borrow::BorrowMut, sync::Arc};

use openvm_circuit::{
    arch::{
        testing::{memory::gen_pointer, VmChipTestBuilder},
        Streams, VmAdapterChip, BITWISE_OP_LOOKUP_BUS,
    },
    utils::{u32_into_limbs, u32_sign_extend},
};
use openvm_circuit_primitives::bitwise_op_lookup::{
    BitwiseOperationLookupBus, BitwiseOperationLookupChip,
};
use openvm_instructions::{instruction::Instruction, UsizeOpcode, VmOpcode};
use openvm_rv32im_transpiler::Rv32HintStoreOpcode::{self, *};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::AbstractField,
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{config::setup_tracing, p3_baby_bear::BabyBear, utils::create_seeded_rng};
use parking_lot::Mutex;
use rand::{rngs::StdRng, Rng};

use super::{Rv32HintStoreChip, Rv32HintStoreCoreChip};
use crate::{
    adapters::{compose, Rv32HintStoreAdapterChip, RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS},
    hintstore::Rv32HintStoreCoreCols,
};

const IMM_BITS: usize = 16;

type F = BabyBear;

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut Rv32HintStoreChip<F>,
    rng: &mut StdRng,
    opcode: Rv32HintStoreOpcode,
    rs1: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    imm: Option<u32>,
) {
    let imm = imm.unwrap_or(rng.gen_range(0..(1 << IMM_BITS)));
    let imm_ext = u32_sign_extend::<IMM_BITS>(imm);
    let ptr_val = rng.gen_range(
        0..(1
            << (tester
                .memory_controller()
                .borrow()
                .mem_config()
                .pointer_max_bits
                - 2)),
    ) << 2;
    let rs1 = rs1
        .unwrap_or(u32_into_limbs::<RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>(
            (ptr_val as u32).wrapping_sub(imm_ext),
        ))
        .map(F::from_canonical_u32);
    let b = gen_pointer(rng, 4);

    let ptr_val = imm_ext.wrapping_add(compose(rs1));
    tester.write(1, b, rs1);

    let read_data: [F; RV32_REGISTER_NUM_LIMBS] =
        array::from_fn(|_| F::from_canonical_u32(rng.gen_range(0..(1 << RV32_CELL_BITS))));
    for data in read_data {
        chip.core
            .streams
            .get()
            .unwrap()
            .lock()
            .hint_stream
            .push_back(data);
    }

    tester.execute(
        chip,
        Instruction::from_usize(
            VmOpcode::with_default_offset(opcode),
            [0, b, imm as usize, 1, 2],
        ),
    );

    let write_data = read_data;
    assert_eq!(write_data, tester.read::<4>(2, ptr_val as usize));
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////
#[test]
fn rand_hintstore_test() {
    setup_tracing();
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let adapter = Rv32HintStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
    );

    let mut core =
        Rv32HintStoreCoreChip::new(bitwise_chip.clone(), Rv32HintStoreOpcode::default_offset());
    core.set_streams(Arc::new(Mutex::new(Streams::default())));
    let mut chip = Rv32HintStoreChip::<F>::new(adapter, core, tester.memory_controller());

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, HINT_STOREW, None, None);
    }

    drop(range_checker_chip);
    let tester = tester.build().load(chip).load(bitwise_chip).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// the write part of the trace and check that the core chip throws the expected error.
// A dummy adaptor is used so memory interactions don't indirectly cause false passes.
//////////////////////////////////////////////////////////////////////////////////////

#[allow(clippy::too_many_arguments)]
fn run_negative_hintstore_test(
    opcode: Rv32HintStoreOpcode,
    data: Option<[u32; RV32_REGISTER_NUM_LIMBS]>,
    expected_error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default();

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let adapter = Rv32HintStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
    );
    let mut core =
        Rv32HintStoreCoreChip::new(bitwise_chip.clone(), Rv32HintStoreOpcode::default_offset());
    core.set_streams(Arc::new(Mutex::new(Streams::default())));
    let adapter_width = BaseAir::<F>::width(adapter.air());
    let mut chip = Rv32HintStoreChip::<F>::new(adapter, core, tester.memory_controller());

    set_and_execute(&mut tester, &mut chip, &mut rng, opcode, None, None);

    let modify_trace = |trace: &mut DenseMatrix<BabyBear>| {
        let mut trace_row = trace.row_slice(0).to_vec();
        let (_, core_row) = trace_row.split_at_mut(adapter_width);
        let core_cols: &mut Rv32HintStoreCoreCols<F> = core_row.borrow_mut();
        if let Some(data) = data {
            core_cols.data = data.map(F::from_canonical_u32);
        }
        *trace = RowMajorMatrix::new(trace_row, trace.width());
    };

    drop(range_checker_chip);
    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .load(bitwise_chip)
        .finalize();
    tester.simple_test_with_expected_error(expected_error);
}

#[test]
fn negative_hintstore_tests() {
    run_negative_hintstore_test(
        HINT_STOREW,
        Some([92, 187, 45, 280]),
        VerificationError::ChallengePhaseError,
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

    let bitwise_bus = BitwiseOperationLookupBus::new(BITWISE_OP_LOOKUP_BUS);
    let bitwise_chip = Arc::new(BitwiseOperationLookupChip::<RV32_CELL_BITS>::new(
        bitwise_bus,
    ));

    let range_checker_chip = tester.memory_controller().borrow().range_checker.clone();
    let adapter = Rv32HintStoreAdapterChip::<F>::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        range_checker_chip.clone(),
    );
    let mut core =
        Rv32HintStoreCoreChip::new(bitwise_chip.clone(), Rv32HintStoreOpcode::default_offset());
    core.set_streams(Arc::new(Mutex::new(Streams::default())));
    let mut chip = Rv32HintStoreChip::<F>::new(adapter, core, tester.memory_controller());

    let num_tests: usize = 100;
    for _ in 0..num_tests {
        set_and_execute(&mut tester, &mut chip, &mut rng, HINT_STOREW, None, None);
    }
}
