use itertools::Itertools;
use openvm_circuit::{
    arch::{
        instructions::program::Program, PreflightExecutionOutput, PreflightExecutor, VmBuilder,
        VmCircuitConfig, VmExecutionConfig,
    },
    utils::TestStarkEngine,
};
use openvm_native_circuit::{
    execute_program_with_config, test_native_config, NativeBuilder, NativeConfig,
};
use openvm_native_compiler::{asm::AsmBuilder, ir::Felt};
use openvm_native_recursion::testing_utils::inner::run_recursive_test;
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig, Val},
    p3_commit::PolynomialSpace,
    p3_field::{extension::BinomialExtensionField, FieldAlgebra},
    prover::{
        cpu::{CpuBackend, CpuDevice},
        hal::{DeviceDataTransporter, ProverBackend, TraceCommitter},
        types::AirProvingContext,
    },
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
        FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::ProofInputForTest,
};

fn fibonacci_program(a: u32, b: u32, n: u32) -> Program<BabyBear> {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    let mut builder = AsmBuilder::<F, EF>::default();

    let prev: Felt<_> = builder.constant(F::from_canonical_u32(a));
    let next: Felt<_> = builder.constant(F::from_canonical_u32(b));

    builder.commit_public_value(prev);
    builder.commit_public_value(next);

    for _ in 2..n {
        let tmp: Felt<_> = builder.uninit();
        builder.assign(&tmp, next);
        builder.assign(&next, prev + next);
        builder.assign(&prev, tmp);
    }

    builder.commit_public_value(next);

    builder.halt();

    builder.compile_isa()
}

// We need this for both BabyBearPoseidon2Config and BabyBearPoseidon2RootConfig
pub(crate) fn fibonacci_program_test_proof_input<SC, E, CpuEngine, const LOG_BLOWUP: usize>(
    a: u32,
    b: u32,
    n: u32,
) -> ProofInputForTest<SC>
where
    SC: StarkGenericConfig,
    E: StarkFriEngine<SC = SC>,
    CpuEngine: StarkFriEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    Domain<SC>: PolynomialSpace<Val = BabyBear>,
    NativeBuilder: VmBuilder<E, VmConfig = NativeConfig>,
    <NativeConfig as VmExecutionConfig<BabyBear>>::Executor:
        PreflightExecutor<BabyBear, <NativeBuilder as VmBuilder<E>>::RecordArena>,
    E::PB: ProverBackend<Val = Val<SC>>,
{
    let fib_program = fibonacci_program(a, b, n);
    let mut config = test_native_config();
    let fri_params = FriParameters::new_for_testing(LOG_BLOWUP);
    let engine = E::new(fri_params);
    let cpu_engine = CpuEngine::new(fri_params);
    config.as_mut().num_public_values = 3;

    let (output, mut vm) = execute_program_with_config::<E, _>(
        fib_program.clone(),
        vec![],
        NativeBuilder::default(),
        config.clone(),
    )
    .unwrap();
    let PreflightExecutionOutput {
        system_records,
        record_arenas,
        ..
    } = output;
    let cached_program_trace = vm.commit_program_on_device(&fib_program);
    vm.load_program(cached_program_trace);
    let ctx = vm
        .generate_proving_ctx(system_records, record_arenas)
        .unwrap();

    let airs = config.create_airs().unwrap().into_airs().collect_vec();
    let (used_airs, per_air): (Vec<_>, Vec<_>) = ctx
        .per_air
        .into_iter()
        .map(|(air_id, air_ctx)| {
            let AirProvingContext {
                cached_mains,
                common_main,
                public_values,
            } = air_ctx;
            let cached_mains = cached_mains
                .into_iter()
                .map(|com| {
                    let trace = engine
                        .device()
                        .transport_matrix_from_device_to_host(&com.trace);
                    let (commitment, data) = cpu_engine.device().commit(&[trace.clone()]);
                    cpu_engine
                        .device()
                        .transport_committed_trace_to_device(commitment, &trace, &data.data)
                })
                .collect::<Vec<_>>();
            let out_ctx = AirProvingContext::<CpuBackend<SC>>::new(
                cached_mains,
                common_main.map(|m| engine.device().transport_matrix_from_device_to_host(&m)),
                public_values,
            );
            (airs[air_id].clone(), out_ctx)
        })
        .unzip();
    ProofInputForTest {
        airs: used_airs,
        per_air,
    }
}

#[test]
fn test_fibonacci_program_verify() {
    let fib_program_stark = fibonacci_program_test_proof_input::<
        BabyBearPoseidon2Config,
        TestStarkEngine,
        BabyBearPoseidon2Engine,
        1, // log_blowup to match hard-coded value in `run_recursive_test`
    >(0, 1, 32);
    run_recursive_test(fib_program_stark, FriParameters::new_for_testing(3));
}

#[cfg(all(feature = "static-verifier", not(feature = "cuda")))]
#[test]
#[ignore = "needs params files"]
fn test_fibonacci_program_halo2_verify() {
    use openvm_native_recursion::halo2::testing_utils::run_static_verifier_test;
    use openvm_stark_sdk::config::baby_bear_poseidon2_root::{
        BabyBearPoseidon2RootConfig, BabyBearPoseidon2RootEngine,
    };

    const LOG_BLOWUP: usize = 3;
    let fib_program_stark = fibonacci_program_test_proof_input::<
        BabyBearPoseidon2RootConfig,
        BabyBearPoseidon2RootEngine,
        BabyBearPoseidon2RootEngine,
        { LOG_BLOWUP },
    >(0, 1, 32);
    run_static_verifier_test(
        fib_program_stark,
        FriParameters::new_for_testing(LOG_BLOWUP),
    );
}
