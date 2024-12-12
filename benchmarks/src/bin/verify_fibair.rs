use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::{bench_from_exe, BenchmarkCli};
use openvm_circuit::arch::instructions::program::DEFAULT_MAX_NUM_PUBLIC_VALUES;
use openvm_native_circuit::NativeConfig;
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_native_recursion::testing_utils::inner::build_verification_program;
use openvm_sdk::config::AppConfig;
/// Benchmark of aggregation VM performance.
/// Proofs:
/// 1. Prove Fibonacci AIR.
/// 2. Verify the proof of 1. by execution VM program in STARK VM.
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    dummy_airs::fib_air::chip::FibonacciChip,
    engine::StarkFriEngine,
    openvm_stark_backend::Chip,
};
use tracing::info_span;

fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_log_blowup = cli_args.app_log_blowup.unwrap_or(2);
    let agg_log_blowup = cli_args.agg_log_blowup.unwrap_or(2);

    let n = 16; // STARK to calculate 16th Fibonacci number.
    let fib_chip = FibonacciChip::new(0, 1, n);
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup),
    );

    run_with_metric_collection("OUTPUT_PATH", || {
        // run_test tries to setup tracing, but it will be ignored since run_with_metric_collection already sets it.
        let vdata = engine
            .run_test(vec![fib_chip.generate_air_proof_input()])
            .unwrap();
        let leaf_fri_params =
            FriParameters::standard_with_100_bits_conjectured_security(agg_log_blowup);
        // FIXME: this should be benchmarked as a single segment VM.
        let app_vm_config = NativeConfig::aggregation(
            DEFAULT_MAX_NUM_PUBLIC_VALUES,
            leaf_fri_params.max_constraint_degree().min(7),
        )
        .with_continuations();
        let compiler_options = CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        };
        let app_config = AppConfig {
            app_fri_params: leaf_fri_params,
            app_vm_config,
            leaf_fri_params: leaf_fri_params.into(),
            compiler_options,
        };
        info_span!("Verify Fibonacci AIR").in_scope(|| {
            let (program, input_stream) = build_verification_program(vdata, compiler_options);
            bench_from_exe(
                "verify_fibair",
                app_config,
                program,
                input_stream.into(),
                false,
            )
        })
    })?;
    Ok(())
}
