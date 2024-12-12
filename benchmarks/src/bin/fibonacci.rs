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

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let n = 100_000u64;
        let mut stdin = StdIn::default();
        stdin.write(&n);
        bench_from_exe(
            "fibonacci_program",
            app_config,
            exe,
            stdin,
            #[cfg(feature = "aggregation")]
            true,
            #[cfg(not(feature = "aggregation"))]
            false,
        )
    })
}
