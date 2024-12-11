#![allow(unused_variables)]
#![allow(unused_imports)]

use ax_stark_backend::p3_field::AbstractField;
use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::{
        baby_bear_poseidon2::BabyBearPoseidon2Engine,
        fri_params::standard_fri_params_with_100_bits_conjectured_security, FriParameters,
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
};
use axvm_benchmarks::utils::{bench_from_exe, build_bench_program, time, BenchmarkCli};
use axvm_circuit::arch::{
    instructions::{exe::AxVmExe, program::DEFAULT_MAX_NUM_PUBLIC_VALUES},
    VirtualMachine,
};
use axvm_native_circuit::NativeConfig;
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_native_recursion::testing_utils::inner::build_verification_program;
use axvm_rv32im_circuit::Rv32ImConfig;
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_sdk::{
    commit::{commit_app_exe, generate_leaf_committed_exe},
    config::AppConfig,
    keygen::{leaf_keygen, AppProvingKey},
    prover::{AggStarkProver, AppProver, LeafProver},
    Sdk, StdIn,
};
use axvm_transpiler::{transpiler::Transpiler, FromElf};
use clap::Parser;
use eyre::Result;
use metrics::gauge;
use tracing::info_span;

fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_fri_params = standard_fri_params_with_100_bits_conjectured_security(
        cli_args.app_log_blowup.unwrap_or(2),
    );
    let leaf_fri_params = standard_fri_params_with_100_bits_conjectured_security(
        cli_args.agg_log_blowup.unwrap_or(2),
    );
    let compiler_options = CompilerOptions {
        // For metric collection
        enable_cycle_tracker: true,
        ..Default::default()
    };

    let app_config = AppConfig {
        app_fri_params,
        app_vm_config: Rv32ImConfig::default(),
        leaf_fri_params: leaf_fri_params.into(),
        compiler_options,
    };

    let elf = build_bench_program("fibonacci")?;
    let exe = AxVmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;

    let engine = BabyBearPoseidon2Engine::new(app_fri_params);
    let n = 100_000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let vm = VirtualMachine::new(engine, Rv32ImConfig::default());
        let app_pk = time(gauge!("keygen_time_ms"), || {
            AppProvingKey::keygen(app_config.clone())
        });
        let committed_exe = time(gauge!("commit_exe_time_ms"), || {
            commit_app_exe(app_config.app_fri_params, exe)
        });
        time(gauge!("execute_and_trace_gen_time_ms"), || {
            vm.execute_and_generate_with_cached_program(committed_exe.clone(), stdin.clone())
        })?;

        let app_prover = AppProver::new(app_pk.app_vm_pk.clone(), committed_exe)
            .with_profile()
            .with_program_name("fibonacci_program".to_string());
        let app_proof = app_prover.generate_app_proof(stdin);
        Sdk.verify_app_proof(&app_pk, &app_proof)?;

        #[cfg(all(feature = "aggregation", feature = "bench-metrics"))]
        {
            let leaf_vm_pk = leaf_keygen(leaf_fri_params);
            let leaf_prover = LeafProver::new(leaf_vm_pk, app_pk.leaf_committed_exe).with_profile();
            leaf_prover.generate_proof(&app_proof);
        }

        Ok(())
    })
}
