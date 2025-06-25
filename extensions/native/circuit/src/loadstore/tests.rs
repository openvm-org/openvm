use std::{array, borrow::BorrowMut};

use openvm_circuit::arch::testing::{memory::gen_pointer, VmChipTestBuilder};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::{
    conversion::AS,
    NativeLoadStoreOpcode::{self, *},
};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{FieldAlgebra, PrimeField32},
    p3_matrix::{
        dense::{DenseMatrix, RowMajorMatrix},
        Matrix,
    },
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use rand::{rngs::StdRng, Rng};
use test_case::test_case;

use super::{NativeLoadStoreChip, NativeLoadStoreCoreAir};
use crate::{
    adapters::{NativeLoadStoreAdapterAir, NativeLoadStoreAdapterCols, NativeLoadStoreAdapterStep},
    test_utils::write_native_array,
    NativeLoadStoreAir, NativeLoadStoreCoreCols, NativeLoadStoreStep,
};

const MAX_INS_CAPACITY: usize = 128;
const NUM_CELLS: usize = 1;
type F = BabyBear;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> NativeLoadStoreChip<F, NUM_CELLS> {
    let mut chip = NativeLoadStoreChip::<F, NUM_CELLS>::new(
        NativeLoadStoreAir::new(
            NativeLoadStoreAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
            NativeLoadStoreCoreAir::new(NativeLoadStoreOpcode::CLASS_OFFSET),
        ),
        NativeLoadStoreStep::new(
            NativeLoadStoreAdapterStep::new(NativeLoadStoreOpcode::CLASS_OFFSET),
            NativeLoadStoreOpcode::CLASS_OFFSET,
        ),
        tester.memory_helper(),
    );
    chip.set_trace_buffer_height(MAX_INS_CAPACITY);

    chip
}

fn set_and_execute(
    tester: &mut VmChipTestBuilder<F>,
    chip: &mut NativeLoadStoreChip<F, NUM_CELLS>,
    rng: &mut StdRng,
    opcode: NativeLoadStoreOpcode,
) {
    let a = gen_pointer(rng, NUM_CELLS);
    let ([c_val], c) = write_native_array(tester, rng, None);

    let mem_ptr = gen_pointer(rng, NUM_CELLS);
    let b = F::from_canonical_usize(mem_ptr) - c_val;
    let data: [F; NUM_CELLS] = array::from_fn(|_| rng.gen());

    match opcode {
        LOADW => {
            tester.write(AS::Native as usize, mem_ptr, data);
        }
        STOREW => {
            tester.write(AS::Native as usize, a, data);
        }
        HINT_STOREW => {
            tester.streams.hint_stream.extend(data);
        }
    }

    tester.execute(
        chip,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                a,
                b.as_canonical_u32() as usize,
                c,
                AS::Native as usize,
                AS::Native as usize,
            ],
        ),
    );

    let result = match opcode {
        STOREW | HINT_STOREW => tester.read(AS::Native as usize, mem_ptr),
        LOADW => tester.read(AS::Native as usize, a),
    };
    assert_eq!(result, data);
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test_case(STOREW, 100)]
#[test_case(HINT_STOREW, 100)]
#[test_case(LOADW, 100)]
fn rand_native_loadstore_test(opcode: NativeLoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut chip = create_test_chip(&tester);

    for _ in 0..num_ops {
        set_and_execute(&mut tester, &mut chip, &mut rng, opcode);
    }
    let tester = tester.build().load(chip).finalize();
    tester.simple_test().expect("Verification failed");
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
struct NativeLoadStorePrankValues {
    // Core cols
    pub data: Option<[F; NUM_CELLS]>,
    pub opcode_flags: Option<[bool; 3]>,
    pub pointer_read: Option<F>,
    // Adapter cols
    pub data_write_pointer: Option<F>,
}

fn run_negative_native_loadstore_test(
    opcode: NativeLoadStoreOpcode,
    prank_vals: NativeLoadStorePrankValues,
    error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut chip = create_test_chip(&tester);

    set_and_execute(&mut tester, &mut chip, &mut rng, opcode);

    let adapter_width = BaseAir::<F>::width(&chip.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut values = trace.row_slice(0).to_vec();
        let (adapter_row, core_row) = values.split_at_mut(adapter_width);
        let adapter_cols: &mut NativeLoadStoreAdapterCols<F, NUM_CELLS> = adapter_row.borrow_mut();
        let core_cols: &mut NativeLoadStoreCoreCols<F, NUM_CELLS> = core_row.borrow_mut();

        if let Some(data) = prank_vals.data {
            core_cols.data = data;
        }
        if let Some(pointer_read) = prank_vals.pointer_read {
            core_cols.pointer_read = pointer_read;
        }
        if let Some(opcode_flags) = prank_vals.opcode_flags {
            [
                core_cols.is_loadw,
                core_cols.is_storew,
                core_cols.is_hint_storew,
            ] = opcode_flags.map(F::from_bool);
        }
        if let Some(data_write_pointer) = prank_vals.data_write_pointer {
            adapter_cols.data_write_pointer = data_write_pointer;
        }

        *trace = RowMajorMatrix::new(values, trace.width());
    };

    disable_debug_builder();
    let tester = tester
        .build()
        .load_and_prank_trace(chip, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(error);
}

#[test]
fn negative_native_loadstore_tests() {
    run_negative_native_loadstore_test(
        STOREW,
        NativeLoadStorePrankValues {
            data_write_pointer: Some(F::ZERO),
            ..Default::default()
        },
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_native_loadstore_test(
        LOADW,
        NativeLoadStorePrankValues {
            data_write_pointer: Some(F::ZERO),
            ..Default::default()
        },
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn invalid_flags_native_loadstore_tests() {
    run_negative_native_loadstore_test(
        HINT_STOREW,
        NativeLoadStorePrankValues {
            opcode_flags: Some([false, false, false]),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );

    run_negative_native_loadstore_test(
        LOADW,
        NativeLoadStorePrankValues {
            opcode_flags: Some([false, false, true]),
            ..Default::default()
        },
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn invalid_data_native_loadstore_tests() {
    run_negative_native_loadstore_test(
        LOADW,
        NativeLoadStorePrankValues {
            data: Some([F::ZERO; NUM_CELLS]),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
}
