use std::{
    array,
    borrow::BorrowMut,
    ops::{Add, Div, Mul, Sub},
};

use openvm_circuit::arch::{
    testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
    Arena, PreflightExecutor,
};
#[cfg(feature = "cuda")]
use openvm_circuit::arch::{
    testing::{GpuChipTestBuilder, GpuTestChipHarness},
    EmptyAdapterCoreLayout,
};
use openvm_instructions::{instruction::Instruction, LocalOpcode};
use openvm_native_compiler::{conversion::AS, FieldExtensionOpcode};
use openvm_stark_backend::{
    p3_air::BaseAir,
    p3_field::{extension::BinomialExtensionField, FieldAlgebra, FieldExtensionAlgebra},
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
    adapters::NativeVectorizedAdapterRecord,
    field_extension::{FieldExtensionChipGpu, FieldExtensionRecord},
};
use crate::{
    adapters::{
        NativeVectorizedAdapterAir, NativeVectorizedAdapterExecutor, NativeVectorizedAdapterFiller,
    },
    field_extension::{
        run_field_extension, FieldExtension, FieldExtensionAir, FieldExtensionChip,
        FieldExtensionCoreAir, FieldExtensionCoreCols, FieldExtensionCoreFiller,
        FieldExtensionExecutor, EXT_DEG,
    },
    test_utils::write_native_array,
};

const MAX_INS_CAPACITY: usize = 128;
type F = BabyBear;
type Harness = TestChipHarness<F, FieldExtensionExecutor, FieldExtensionAir, FieldExtensionChip<F>>;

fn create_test_chip(tester: &VmChipTestBuilder<F>) -> Harness {
    let air = FieldExtensionAir::new(
        NativeVectorizedAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        FieldExtensionCoreAir::new(),
    );
    let executor = FieldExtensionExecutor::new(NativeVectorizedAdapterExecutor::new());
    let chip = FieldExtensionChip::<F>::new(
        FieldExtensionCoreFiller::new(NativeVectorizedAdapterFiller),
        tester.memory_helper(),
    );

    Harness::with_capacity(executor, air, chip, MAX_INS_CAPACITY)
}

#[cfg(feature = "cuda")]
fn create_test_harness(
    tester: &GpuChipTestBuilder,
) -> GpuTestChipHarness<
    F,
    FieldExtensionExecutor,
    FieldExtensionAir,
    FieldExtensionChipGpu,
    FieldExtensionChip<F>,
> {
    let adapter_air =
        NativeVectorizedAdapterAir::new(tester.execution_bridge(), tester.memory_bridge());
    let core_air = FieldExtensionCoreAir::new();
    let air = FieldExtensionAir::new(adapter_air, core_air);

    let adapter_step = NativeVectorizedAdapterExecutor::new();
    let executor = FieldExtensionExecutor::new(adapter_step);

    let core_filler = FieldExtensionCoreFiller::new(NativeVectorizedAdapterFiller);

    let cpu_chip = FieldExtensionChip::new(core_filler, tester.dummy_memory_helper());
    let gpu_chip = FieldExtensionChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
}

fn set_and_execute<E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    opcode: FieldExtensionOpcode,
    y: Option<[F; EXT_DEG]>,
    z: Option<[F; EXT_DEG]>,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
    let (y_val, y_ptr) = write_native_array(tester, rng, y);
    let (z_val, z_ptr) = write_native_array(tester, rng, z);

    let x_ptr = gen_pointer(rng, EXT_DEG);

    tester.execute(
        executor,
        arena,
        &Instruction::from_usize(
            opcode.global_opcode(),
            [
                x_ptr,
                y_ptr,
                z_ptr,
                AS::Native as usize,
                AS::Native as usize,
            ],
        ),
    );

    let result = tester.read::<EXT_DEG>(AS::Native as usize, x_ptr);
    let expected = run_field_extension(opcode, y_val, z_val);
    assert_eq!(result, expected);
}

fn rand_set_and_execute<E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    opcode: FieldExtensionOpcode,
    num_ops: usize,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
    let mut rng = create_seeded_rng();
    for _ in 0..num_ops {
        set_and_execute(tester, executor, arena, &mut rng, opcode, None, None);
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// POSITIVE TESTS
///
/// Randomly generate computations and execute, ensuring that the generated trace
/// passes all constraints.
///////////////////////////////////////////////////////////////////////////////////////

#[test_case(FieldExtensionOpcode::FE4ADD, 100)]
#[test_case(FieldExtensionOpcode::FE4SUB, 100)]
#[test_case(FieldExtensionOpcode::BBE4MUL, 100)]
#[test_case(FieldExtensionOpcode::BBE4DIV, 100)]
fn rand_field_extension_test(opcode: FieldExtensionOpcode, num_ops: usize) {
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&tester);

    rand_set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        opcode,
        num_ops,
    );

    let tester = tester.build().load(harness).finalize();
    tester.simple_test().expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test_case(FieldExtensionOpcode::FE4ADD, 100)]
#[test_case(FieldExtensionOpcode::FE4SUB, 100)]
#[test_case(FieldExtensionOpcode::BBE4MUL, 100)]
#[test_case(FieldExtensionOpcode::BBE4DIV, 100)]
fn test_cuda_rand_field_extension_tracegen(opcode: FieldExtensionOpcode, num_ops: usize) {
    let mut tester = GpuChipTestBuilder::default();
    let mut harness = create_test_harness(&tester);

    rand_set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.dense_arena,
        opcode,
        num_ops,
    );

    type Record<'a> = (
        &'a mut NativeVectorizedAdapterRecord<F, EXT_DEG>,
        &'a mut FieldExtensionRecord<F>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record<'_>, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, NativeVectorizedAdapterExecutor<EXT_DEG>>::new(),
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
struct FieldExtensionPrankValues {
    pub x: Option<[F; EXT_DEG]>,
    pub y: Option<[F; EXT_DEG]>,
    pub z: Option<[F; EXT_DEG]>,
    pub opcode_flags: Option<[bool; 4]>,
    pub divisor_inv: Option<[F; EXT_DEG]>,
}

fn run_negative_field_extension_test(
    opcode: FieldExtensionOpcode,
    y: Option<[F; EXT_DEG]>,
    z: Option<[F; EXT_DEG]>,
    prank_vals: FieldExtensionPrankValues,
    error: VerificationError,
) {
    let mut rng = create_seeded_rng();
    let mut tester = VmChipTestBuilder::default_native();
    let mut harness = create_test_chip(&tester);
    set_and_execute(
        &mut tester,
        &mut harness.executor,
        &mut harness.arena,
        &mut rng,
        opcode,
        y,
        z,
    );

    let adapter_width = BaseAir::<F>::width(&harness.air.adapter);
    let modify_trace = |trace: &mut DenseMatrix<F>| {
        let mut values = trace.row_slice(0).to_vec();
        let core_cols: &mut FieldExtensionCoreCols<F> =
            values.split_at_mut(adapter_width).1.borrow_mut();

        if let Some(x) = prank_vals.x {
            core_cols.x = x;
        }
        if let Some(y) = prank_vals.y {
            core_cols.y = y;
        }
        if let Some(z) = prank_vals.z {
            core_cols.z = z;
        }
        if let Some(opcode_flags) = prank_vals.opcode_flags {
            [
                core_cols.is_add,
                core_cols.is_sub,
                core_cols.is_mul,
                core_cols.is_div,
            ] = opcode_flags.map(F::from_bool);
        }
        if let Some(divisor_inv) = prank_vals.divisor_inv {
            core_cols.divisor_inv = divisor_inv;
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
fn rand_negative_field_extension_test() {
    let mut rng = create_seeded_rng();
    run_negative_field_extension_test(
        FieldExtensionOpcode::FE4ADD,
        None,
        None,
        FieldExtensionPrankValues {
            x: Some(array::from_fn(|_| rng.gen::<F>())),
            y: Some(array::from_fn(|_| rng.gen::<F>())),
            z: Some(array::from_fn(|_| rng.gen::<F>())),
            opcode_flags: Some(array::from_fn(|_| rng.gen_bool(0.5))),
            divisor_inv: Some(array::from_fn(|_| rng.gen::<F>())),
        },
        VerificationError::OodEvaluationMismatch,
    );
}

#[test]
fn field_extension_negative_tests() {
    run_negative_field_extension_test(
        FieldExtensionOpcode::BBE4DIV,
        None,
        None,
        FieldExtensionPrankValues {
            z: Some([F::ZERO; EXT_DEG]),
            ..Default::default()
        },
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_field_extension_test(
        FieldExtensionOpcode::BBE4DIV,
        None,
        None,
        FieldExtensionPrankValues {
            divisor_inv: Some([F::ZERO; EXT_DEG]),
            ..Default::default()
        },
        VerificationError::OodEvaluationMismatch,
    );

    run_negative_field_extension_test(
        FieldExtensionOpcode::BBE4MUL,
        Some([F::ZERO; EXT_DEG]),
        None,
        FieldExtensionPrankValues {
            z: Some([F::ZERO; EXT_DEG]),
            ..Default::default()
        },
        VerificationError::ChallengePhaseError,
    );
}

#[test]
fn new_field_extension_consistency_test() {
    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    let len_tests = 100;
    let mut rng = create_seeded_rng();

    let operands: Vec<([F; 4], [F; 4])> = (0..len_tests)
        .map(|_| {
            (
                array::from_fn(|_| rng.gen::<F>()),
                array::from_fn(|_| rng.gen::<F>()),
            )
        })
        .collect();

    for (a, b) in operands {
        let a_ext = EF::from_base_slice(&a);
        let b_ext = EF::from_base_slice(&b);

        let plonky_add = a_ext.add(b_ext);
        let plonky_sub = a_ext.sub(b_ext);
        let plonky_mul = a_ext.mul(b_ext);
        let plonky_div = a_ext.div(b_ext);

        let my_add = FieldExtension::add(a, b);
        let my_sub = FieldExtension::subtract(a, b);
        let my_mul = FieldExtension::multiply(a, b);
        let my_div = FieldExtension::divide(a, b);

        assert_eq!(my_add, plonky_add.as_base_slice());
        assert_eq!(my_sub, plonky_sub.as_base_slice());
        assert_eq!(my_mul, plonky_mul.as_base_slice());
        assert_eq!(my_div, plonky_div.as_base_slice());
    }
}
