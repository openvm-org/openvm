use std::sync::Arc;

use openvm_instructions::{
    instruction::Instruction, riscv::RV32_IMM_AS, LocalOpcode, PublishOpcode, NATIVE_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilderWithPublicValues},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::types::AirProvingContext,
    rap::PartitionedBaseAir,
    utils::disable_debug_builder,
    verifier::VerificationError,
    AirRef,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::{create_seeded_rng, to_field_vec},
};
use rand::{rngs::StdRng, Rng};

#[cfg(feature = "cuda")]
use crate::system::cuda::public_values::PublicValuesChipGPU;
use crate::{
    arch::{
        testing::{memory::gen_pointer, TestBuilder, TestChipHarness, VmChipTestBuilder},
        Arena, MemoryConfig, PreflightExecutor, SystemConfig, VmCoreAir,
    },
    system::{
        native_adapter::{NativeAdapterAir, NativeAdapterExecutor},
        public_values::{
            columns::PublicValuesCoreColsView,
            core::{AdapterInterface, PublicValuesCoreAir},
            PublicValuesAir, PublicValuesChip, PublicValuesExecutor, PublicValuesFiller,
        },
    },
};
#[cfg(feature = "cuda")]
use crate::{
    arch::{
        testing::{GpuChipTestBuilder, GpuTestChipHarness, RANGE_CHECKER_BUS},
        EmptyAdapterCoreLayout,
    },
    system::{
        native_adapter::NativeAdapterRecord, public_values::PublicValuesRecord,
        VariableRangeCheckerBus,
    },
};

type F = BabyBear;
type Harness = TestChipHarness<F, PublicValuesExecutor<F>, PublicValuesAir, PublicValuesChip<F>>;
#[cfg(feature = "cuda")]
type GpuHarness = GpuTestChipHarness<
    F,
    PublicValuesExecutor<F>,
    PublicValuesAir,
    PublicValuesChipGPU,
    PublicValuesChip<F>,
>;

impl<F: Field> PartitionedBaseAir<F> for PublicValuesCoreAir {}

impl<AB: InteractionBuilder + AirBuilderWithPublicValues> Air<AB> for PublicValuesCoreAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_core = main.row_slice(0);
        // It's never used, so pick any value.
        let dummy_pc = local_core[0];
        VmCoreAir::<AB, AdapterInterface<AB::Expr>>::eval(self, builder, &local_core, dummy_pc);
    }
}

fn create_test_chips(tester: &VmChipTestBuilder<F>, system_config: &SystemConfig) -> Harness {
    let num_custom_pvs = system_config.num_public_values;
    let max_degree = system_config.max_constraint_degree as u32 - 1;

    let air = PublicValuesAir::new(
        NativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        PublicValuesCoreAir::new(num_custom_pvs, max_degree),
    );

    let executor = PublicValuesExecutor::new(NativeAdapterExecutor::<F, 2, 0>::default());

    let cpu_chip = PublicValuesChip::new(
        PublicValuesFiller::new(
            NativeAdapterExecutor::<F, 2, 0>::default(),
            num_custom_pvs,
            max_degree,
        ),
        tester.memory_helper(),
    );

    Harness::with_capacity(executor, air, cpu_chip, num_custom_pvs)
}

#[cfg(feature = "cuda")]
fn create_cuda_test_harness(
    tester: &GpuChipTestBuilder,
    mem_config: &MemoryConfig,
    system_config: &SystemConfig,
) -> GpuHarness {
    let num_custom_pvs = system_config.num_public_values;
    let max_degree = system_config.max_constraint_degree as u32 - 1;
    let timestamp_max_bits = mem_config.timestamp_max_bits;

    let air = PublicValuesAir::new(
        NativeAdapterAir::new(tester.execution_bridge(), tester.memory_bridge()),
        PublicValuesCoreAir::new(num_custom_pvs, max_degree),
    );

    let executor = PublicValuesExecutor::new(NativeAdapterExecutor::<F, 2, 0>::default());

    let cpu_chip = PublicValuesChip::new(
        PublicValuesFiller::new(
            NativeAdapterExecutor::<F, 2, 0>::default(),
            num_custom_pvs,
            max_degree,
        ),
        tester.dummy_memory_helper(),
    );
    let gpu_chip = PublicValuesChipGPU::new(
        tester.range_checker(),
        num_custom_pvs,
        max_degree,
        timestamp_max_bits as u32,
    );

    GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, num_custom_pvs)
}

fn set_and_execute<E, RA>(
    tester: &mut impl TestBuilder<F>,
    executor: &mut E,
    arena: &mut RA,
    rng: &mut StdRng,
    public_values: &mut Vec<F>,
    idx: u32,
) where
    E: PreflightExecutor<F, RA>,
    RA: Arena,
{
    let (b, e) = if rng.gen_bool(0.5) {
        let val = F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32));
        public_values.push(val);
        (val, F::from_canonical_u32(RV32_IMM_AS))
    } else {
        let ptr = gen_pointer(rng, 4);
        let val = F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32));
        public_values.push(val);
        tester.write(NATIVE_AS as usize, ptr, [val]);
        (
            F::from_canonical_u32(ptr as u32),
            F::from_canonical_u32(NATIVE_AS),
        )
    };

    let (c, f) = if rng.gen_bool(0.5) {
        (
            F::from_canonical_u32(idx),
            F::from_canonical_u32(RV32_IMM_AS),
        )
    } else {
        let ptr = gen_pointer(rng, 4);
        let val = F::from_canonical_u32(idx);
        tester.write(NATIVE_AS as usize, ptr, [val]);
        (
            F::from_canonical_u32(ptr as u32),
            F::from_canonical_u32(NATIVE_AS),
        )
    };

    let instruction = Instruction {
        opcode: PublishOpcode::PUBLISH.global_opcode(),
        a: F::ZERO,
        b,
        c,
        d: F::ZERO,
        e,
        f,
        g: F::ZERO,
    };

    tester.execute(executor, arena, &instruction);
}

#[test]
fn public_values_rand_test() {
    let mut rng = create_seeded_rng();
    let system_config = SystemConfig::default();
    let mem_config = MemoryConfig::default();
    let mut tester = VmChipTestBuilder::volatile(mem_config);
    tester.set_num_public_values(system_config.num_public_values);

    let mut harness = create_test_chips(&tester, &system_config);

    let mut public_values = Vec::new();
    for idx in 0..system_config.num_public_values {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.arena,
            &mut rng,
            &mut public_values,
            idx as u32,
        );
    }
    harness.chip.inner.set_public_values(public_values);

    tester
        .build()
        .load(harness)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}

#[test]
fn public_values_happy_path_1() {
    let cols = PublicValuesCoreColsView::<F, F> {
        is_valid: F::ONE,
        value: F::from_canonical_u32(12),
        index: F::from_canonical_u32(2),
        custom_pv_vars: to_field_vec(vec![1, 0]),
        _marker: Default::default(),
    };
    let air: AirRef<_> = Arc::new(PublicValuesCoreAir::new(3, 2));
    let trace = RowMajorMatrix::new_row(cols.flatten());
    let pvs = to_field_vec(vec![0, 0, 12]);

    BabyBearPoseidon2Engine::run_test_fast(
        vec![air],
        vec![AirProvingContext::simple(Arc::new(trace), pvs)],
    )
    .expect("Verification failed");
}

#[test]
fn public_values_neg_pv_not_match() {
    let cols = PublicValuesCoreColsView::<F, F> {
        is_valid: F::ONE,
        value: F::from_canonical_u32(12),
        index: F::from_canonical_u32(2),
        custom_pv_vars: to_field_vec(vec![1, 0]),
        _marker: Default::default(),
    };
    let air: AirRef<_> = Arc::new(PublicValuesCoreAir::new(3, 2));
    let trace = RowMajorMatrix::new_row(cols.flatten());
    let pvs = to_field_vec(vec![0, 0, 56456]);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_test_fast(
            vec![air],
            vec![AirProvingContext::simple(Arc::new(trace), pvs)]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch)
    );
}

#[test]
fn public_values_neg_index_out_of_bound() {
    let cols = PublicValuesCoreColsView::<F, F> {
        is_valid: F::ONE,
        value: F::from_canonical_u32(12),
        index: F::from_canonical_u32(8),
        custom_pv_vars: to_field_vec(vec![0, 0]),
        _marker: Default::default(),
    };
    let air: AirRef<_> = Arc::new(PublicValuesCoreAir::new(3, 2));
    let trace = RowMajorMatrix::new_row(cols.flatten());
    let pvs = to_field_vec(vec![0, 0, 0]);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_test_fast(
            vec![air],
            vec![AirProvingContext::simple(Arc::new(trace), pvs)]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch)
    );
}

#[test]
fn public_values_neg_double_publish() {
    // A public value is published twice with different values. Neither of them should be accepted.
    public_values_neg_double_publish_impl(12);
    public_values_neg_double_publish_impl(13);
}

fn public_values_neg_double_publish_impl(actual_pv: u32) {
    let rows = [
        PublicValuesCoreColsView::<F, F> {
            is_valid: F::ONE,
            value: F::from_canonical_u32(12),
            index: F::from_canonical_u32(0),
            custom_pv_vars: to_field_vec(vec![0, 1]),
            _marker: Default::default(),
        },
        PublicValuesCoreColsView::<F, F> {
            is_valid: F::ONE,
            value: F::from_canonical_u32(13),
            index: F::from_canonical_u32(0),
            custom_pv_vars: to_field_vec(vec![0, 1]),
            _marker: Default::default(),
        },
    ];
    let width = rows[0].width();
    let flatten_rows: Vec<_> = rows.into_iter().flat_map(|r| r.flatten()).collect();
    let trace = RowMajorMatrix::new(flatten_rows, width);
    let air: AirRef<_> = Arc::new(PublicValuesCoreAir::new(3, 2));
    let pvs = to_field_vec(vec![0, 0, actual_pv]);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_test_fast(
            vec![air],
            vec![AirProvingContext::simple(Arc::new(trace), pvs)]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_public_values_tracegen() {
    use itertools::Itertools;

    let mut rng = create_seeded_rng();
    let system_config = SystemConfig::default();
    let mem_config = MemoryConfig::default();
    let bus = VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, mem_config.decomp);
    let mut tester = GpuChipTestBuilder::volatile(mem_config.clone(), bus);
    tester.custom_pvs = vec![None; system_config.num_public_values];

    let mut harness = create_cuda_test_harness(&tester, &mem_config, &system_config);
    let mut public_values = vec![];
    for idx in 0..system_config.num_public_values {
        set_and_execute(
            &mut tester,
            &mut harness.executor,
            &mut harness.dense_arena,
            &mut rng,
            &mut public_values,
            idx as u32,
        );
    }
    let exec_pvs = tester
        .custom_pvs
        .iter()
        .map(|&x| x.unwrap_or(F::ZERO))
        .collect_vec();
    assert_eq!(exec_pvs, public_values);

    harness
        .cpu_chip
        .inner
        .set_public_values(public_values.clone());
    harness.gpu_chip.public_values = public_values;

    type Record<'a> = (
        &'a mut NativeAdapterRecord<F, 2, 0>,
        &'a mut PublicValuesRecord<F>,
    );
    harness
        .dense_arena
        .get_record_seeker::<Record, _>()
        .transfer_to_matrix_arena(
            &mut harness.matrix_arena,
            EmptyAdapterCoreLayout::<F, NativeAdapterExecutor<F, 2, 0>>::new(),
        );

    tester
        .build()
        .load_gpu_harness(harness)
        .finalize()
        .simple_test()
        .expect("Verification failed");
}
