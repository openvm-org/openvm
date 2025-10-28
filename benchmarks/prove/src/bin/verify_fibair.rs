use std::sync::Arc;

use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
#[cfg(feature = "cuda")]
use openvm_circuit::utils::cpu_proving_ctx_to_gpu;
use openvm_circuit::{
    arch::{
        instructions::exe::VmExe, verify_single, SingleSegmentVmProver,
        DEFAULT_MAX_NUM_PUBLIC_VALUES,
    },
    utils::TestStarkEngine as Poseidon2Engine,
};
use openvm_native_circuit::{NativeBuilder, NativeConfig, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_native_recursion::testing_utils::inner::build_verification_program;
use openvm_sdk::{
    config::{AppConfig, DEFAULT_APP_LOG_BLOWUP, DEFAULT_LEAF_LOG_BLOWUP},
    keygen::AppProvingKey,
    prover::vm::new_local_prover,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection, config::FriParameters,
    dummy_airs::fib_air::chip::FibonacciChip, engine::StarkFriEngine, openvm_stark_backend::Chip,
};
use tracing::info_span;

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
    let engine = Poseidon2Engine::new(FriParameters::standard_with_100_bits_conjectured_security(
        app_log_blowup,
    ));

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        // run_test tries to setup tracing, but it will be ignored since run_with_metric_collection
        // already sets it.
        let (fib_air, fib_ctx) = (
            vec![fib_chip.air()],
            vec![fib_chip.generate_proving_ctx(())],
        );
        #[cfg(feature = "cuda")]
        let fib_ctx = fib_ctx.into_iter().map(cpu_proving_ctx_to_gpu).collect();
        let vdata = engine.run_test(fib_air, fib_ctx).unwrap();
        // Unlike other apps, this "app" does not have continuations enabled.
        let app_fri_params =
            FriParameters::standard_with_100_bits_conjectured_security(leaf_log_blowup);
        let mut app_vm_config = NativeConfig::aggregation(
            DEFAULT_MAX_NUM_PUBLIC_VALUES,
            app_fri_params.max_constraint_degree().min(7),
        );
        app_vm_config.system.profiling = args.profiling;
        app_vm_config.system.max_constraint_degree = (1 << app_log_blowup) + 1;

        let compiler_options = CompilerOptions::default();
        let app_config = AppConfig {
            app_fri_params: app_fri_params.into(),
            app_vm_config,
            leaf_fri_params: app_fri_params.into(),
            compiler_options,
        };
        let (program, input_stream) = build_verification_program(vdata, compiler_options);
        let app_pk = AppProvingKey::keygen(app_config)?;
        let app_vk = app_pk.get_app_vk();
        let exe = Arc::new(VmExe::new(program));
        let mut prover = new_local_prover::<Poseidon2Engine, _>(
            NativeBuilder::default(),
            &app_pk.app_vm_pk,
            exe,
        )?;
        let proof = info_span!("verify_fibair", group = "verify_fibair").in_scope(|| {
            #[cfg(feature = "metrics")]
            metrics::counter!("fri.log_blowup")
                .absolute(prover.vm.engine.fri_params().log_blowup as u64);
            SingleSegmentVmProver::prove(&mut prover, input_stream, NATIVE_MAX_TRACE_HEIGHTS)
        })?;
        verify_single(&prover.vm.engine, &app_vk.vk, &proof)?;
        Ok(())
    })?;
    Ok(())
}
