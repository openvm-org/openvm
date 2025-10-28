use std::{array, borrow::BorrowMut};

use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
    Arena, PreflightExecutor,
};
#[cfg(feature = "cuda")]
use openvm_circuit::arch::{
    testing::{GpuChipTestBuilder, GpuTestChipHarness},
    EmptyAdapterCoreLayout,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode, VmOpcode};
#[cfg(feature = "cuda")]
use openvm_native_compiler::NativeLoadStore4Opcode;
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

#[cfg(feature = "cuda")]
use crate::{
    adapters::NativeLoadStoreAdapterRecord,
    loadstore::{NativeLoadStoreChipGpu, NativeLoadStoreCoreRecord},
};
use crate::{
    adapters::{
        NativeLoadStoreAdapterAir, NativeLoadStoreAdapterCols, NativeLoadStoreAdapterExecutor,
        NativeLoadStoreAdapterFiller,
    },
    loadstore::{
        NativeLoadStoreAir, NativeLoadStoreChip, NativeLoadStoreCoreAir, NativeLoadStoreCoreCols,
        NativeLoadStoreCoreFiller, NativeLoadStoreExecutor,
    },
    test_utils::write_native_array,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness<const NUM_CELLS: usize> = TestChipHarness<
    F,
    NativeLoadStoreExecutor<NUM_CELLS>,
    NativeLoadStoreAir<NUM_CELLS>,
    NativeLoadStoreChip<F, NUM_CELLS>,
>;
#[cfg(feature = "cuda")]
type GpuHarness<const NUM_CELLS: usize> = GpuTestChipHarness<
    F,
    NativeLoadStoreExecutor<NUM_CELLS>,
    NativeLoadStoreAir<NUM_CELLS>,
    NativeLoadStoreChipGpu<NUM_CELLS>,
    NativeLoadStoreChip<F, NUM_CELLS>,
>;

fn create_test_chip<const NUM_CELLS: usize>(tester: &VmChipTestBuilder<F>) -> Harness<NUM_CELLS> {
    let air = NativeLoadStoreAir::new(
        NativeLoadStoreAdapterAir::new(tester.memory_bridge(), tester.execution_bridge()),
        NativeLoadStoreCoreAir::new(NativeLoadStoreOpcode::CLASS_OFFSET),
    );
    let executor = NativeLoadStoreExecutor::new(
        NativeLoadStoreAdapterExecutor::new(NativeLoadStoreOpcode::CLASS_OFFSET),
        NativeLoadStoreOpcode::CLASS_OFFSET,
    );
    let chip = NativeLoadStoreChip::<F, NUM_CELLS>::new(
        NativeLoadStoreCoreFiller::new(NativeLoadStoreAdapterFiller),
        tester.memory_helper(),
    );

    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_test_harness<const NUM_CELLS: usize>(
    tester: &GpuChipTestBuilder,
    offset: usize,
) -> GpuHarness<NUM_CELLS> {
    let adapter_air =
        NativeLoadStoreAdapterAir::new(tester.memory_bridge(), tester.execution_bridge());
    let core_air = NativeLoadStoreCoreAir::new(offset);
    let air = NativeLoadStoreAir::new(adapter_air, core_air);

    let adapter_step = NativeLoadStoreAdapterExecutor::new(offset);
    let executor = NativeLoadStoreExecutor::new(adapter_step, offset);

    let core_filler = NativeLoadStoreCoreFiller::new(NativeLoadStoreAdapterFiller);
    let cpu_chip = NativeLoadStoreChip::new(core_filler, tester.dummy_memory_helper());

    let gpu_chip = NativeLoadStoreChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn set_and_execute<const NUM_CELLS: usize, E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: NativeLoadStoreOpcode,
    global_opcode: VmOpcode,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
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
            tester.streams_mut().hint_stream.extend(data);
        }
    }

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            global_opcode,
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
fn rand_native_loadstore_test_1(opcode: NativeLoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip::<1>(&tester);

    for _ in 0..num_ops {
        set_and_execute::<1, _, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
            opcode.global_opcode(),
        );
    }
    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[test_case(STOREW, 100)]
#[test_case(HINT_STOREW, 100)]
#[test_case(LOADW, 100)]
fn rand_native_loadstore_test_4(opcode: NativeLoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip::<4>(&tester);

    for _ in 0..num_ops {
        set_and_execute::<4, _, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            opcode,
            opcode.global_opcode(),
        );
    }
    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(NativeLoadStoreOpcode::LOADW, 100)]
#[test_case(NativeLoadStoreOpcode::STOREW, 100)]
#[test_case(NativeLoadStoreOpcode::HINT_STOREW, 100)]
fn test_cuda_native_loadstore_1_tracegen(opcode: NativeLoadStoreOpcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_test_harness::<1>(&tester, NativeLoadStoreOpcode::CLASS_OFFSET);

    for _ in 0..num_ops {
        set_and_execute::<1, _, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode,
            opcode.global_opcode(),
        );
    }

    type Record<'a, const N: usize> = (
        &'a mut NativeLoadStoreAdapterRecord<F, N>,
        &'a mut NativeLoadStoreCoreRecord<F, N>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record<'_, 1>, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, NativeLoadStoreAdapterExecutor<1>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

#[cfg(feature = "cuda")]
#[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::LOADW), 100)]
#[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::STOREW), 100)]
#[test_case(NativeLoadStore4Opcode(NativeLoadStoreOpcode::HINT_STOREW), 100)]
fn test_cuda_native_loadstore_4_tracegen(opcode: NativeLoadStore4Opcode, num_ops: usize) {
    let mut rng = create_seeded_rng();
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_test_harness::<4>(&tester, NativeLoadStore4Opcode::CLASS_OFFSET);

    for _ in 0..num_ops {
        set_and_execute::<4, _, _>(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            opcode.0,
            opcode.global_opcode(),
        );
    }

    type Record<'a, const N: usize> = (
        &'a mut NativeLoadStoreAdapterRecord<F, N>,
        &'a mut NativeLoadStoreCoreRecord<F, N>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record<'_, 4>, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, NativeLoadStoreAdapterExecutor<4>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .unwrap();
}

//////////////////////////////////////////////////////////////////////////////////////
// NEGATIVE TESTS
//
// Given a fake trace of a single operation, setup a chip and run the test. We replace
// part of the trace and check that the chip throws the expected error.
//////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Default)]
struct NativeLoadStorePrankValues<const NUM_CELLS: usize> {
    // Core cols
    pub data: Option<[F; NUM_CELLS]>,
    pub opcode_flags: Option<[bool; 3]>,
    pub pointer_read: Option<F>,
    // Adapter cols
    pub data_write_pointer: Option<F>,
}

fn run_negative_native_loadstore_test<const NUM_CELLS: usize>(
    opcode: NativeLoadStoreOpcode,
    prank_vals: NativeLoadStorePrankValues<NUM_CELLS>,
    error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip::<NUM_CELLS>(&tester);

    set_and_execute::<NUM_CELLS, _, _>(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        opcode.global_opcode(),
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
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
        .load_and_prank_trace(harness, modify_trace)
        .finalize();
    tester.simple_test_with_expected_error(error);
}

#[test]
fn negative_native_loadstore_tests() {
    run_negative_native_loadstore_test::<1>(
        STOREW,
        NativeLoadStorePrankValues {
            data_write_pointer: Some(F::ZERO),
            ..Default::default()
        },
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_native_loadstore_test::<1>(
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
    run_negative_native_loadstore_test::<1>(
        HINT_STOREW,
        NativeLoadStorePrankValues {
            opcode_flags: Some([false, false, false]),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );

    run_negative_native_loadstore_test::<1>(
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
            data: Some([F::ZERO; 4]),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
}
