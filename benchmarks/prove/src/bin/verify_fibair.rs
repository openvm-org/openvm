use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::arch::DEFAULT_MAX_NUM_PUBLIC_VALUES;
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_native_recursion::testing_utils::inner::build_verification_program;
use openvm_sdk::{
    config::{AppConfig, DEFAULT_APP_LOG_BLOWUP, DEFAULT_LEAF_LOG_BLOWUP},
    prover::AppProver,
    Sdk,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    collect_airs_and_inputs,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    dummy_airs::fib_air::chip::FibonacciChip,
    engine::StarkFriEngine,
    openvm_stark_backend::Chip,
};

/// Benchmark of aggregation VM performance.
/// Proofs:
/// 1. Prove Fibonacci AIR.
/// 2. Verify the proof of 1. by execution VM program in STARK VM.
fn main() -> Result<()> {
    let args = BenchmarkCli::parse();
    let app_log_blowup = args.app_log_blowup.unwrap_or(DEFAULT_APP_LOG_BLOWUP);
    let leaf_log_blowup = args.leaf_log_blowup.unwrap_or(DEFAULT_LEAF_LOG_BLOWUP);

    let n = 1 << 15; // STARK to calculate (2 ** 15)th Fibonacci number.
    let fib_chip = FibonacciChip::new(0, 1, n);
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
    );

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        // run_test tries to setup tracing, but it will be ignored since run_with_metric_collection
        // already sets it.
        let (fib_air, fib_input) = collect_airs_and_inputs!(fib_chip);
        let vdata = engine.run_test(fib_air, fib_input).unwrap();
        // Unlike other apps, this "app" does not have continuations enabled.
        let app_fri_params =
            FriParameters::standard_with_100_bits_conjectured_security(leaf_log_blowup);
        let mut app_vm_config = NativeConfig::aggregation(
            DEFAULT_MAX_NUM_PUBLIC_VALUES,
            app_fri_params.max_constraint_degree().min(7),
        );
        app_vm_config.system.profiling = args.profiling;

        let compiler_options = CompilerOptions::default();
        let app_config = AppConfig {
            app_fri_params: app_fri_params.into(),
            app_vm_config,
            leaf_fri_params: app_fri_params.into(),
            compiler_options,
        };
        let (program, input_stream) = build_verification_program(vdata, compiler_options);
        let sdk = Sdk::new();
        let app_pk = sdk.app_keygen(app_config)?;
        let app_vk = app_pk.get_app_vk();
        let committed_exe = sdk.commit_app_exe(app_fri_params, program.into())?;
        let prover = AppProver::<_, BabyBearPoseidon2Engine>::new(app_pk.app_vm_pk, committed_exe)
            .with_program_name("verify_fibair");
        let proof = prover.generate_app_proof_without_continuations(input_stream.into());
        sdk.verify_app_proof_without_continuations(&app_vk, &proof)?;
        Ok(())
    })?;
    Ok(())
}
