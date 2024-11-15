/// Benchmark of aggregation VM performance.
/// Proofs:
/// 1. Prove Fibonacci AIR.
/// 2. Verify the proof of 1. by execution VM program in STARK VM.
use ax_stark_sdk::{
    ax_stark_backend::Chip,
    bench::run_with_metric_collection,
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters},
    dummy_airs::fib_air::chip::FibonacciChip,
    engine::StarkFriEngine,
};
use axvm_benchmarks::utils::{bench_from_exe, BenchmarkCli};
use axvm_circuit::arch::VmConfig;
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_recursion::testing_utils::inner::build_verification_program;
use clap::Parser;
use eyre::Result;
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
        let max_constraint_degree = ((1 << agg_log_blowup) + 1).min(7);
        let config = VmConfig::aggregation(0, max_constraint_degree);
        let compiler_options = CompilerOptions {
            enable_cycle_tracker: true,
            ..Default::default()
        };
        info_span!("Verify Fibonacci AIR", group = "verify_fibair",).in_scope(|| {
            let (program, input_stream) =
                build_verification_program(vdata, compiler_options.clone());
            let engine = BabyBearPoseidon2Engine::new(
                FriParameters::standard_with_100_bits_conjectured_security(agg_log_blowup),
            );
            bench_from_exe(engine, config.clone(), program, input_stream)
        })
    })?;
    Ok(())
}
