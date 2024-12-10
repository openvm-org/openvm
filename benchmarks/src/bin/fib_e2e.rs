use ax_stark_sdk::{
    bench::run_with_metric_collection,
    config::fri_params::standard_fri_params_with_100_bits_conjectured_security,
};
use axvm_benchmarks::utils::{build_bench_program, BenchmarkCli};
use axvm_circuit::arch::instructions::{exe::AxVmExe, program::DEFAULT_MAX_NUM_PUBLIC_VALUES};
use axvm_native_compiler::conversion::CompilerOptions;
use axvm_rv32im_circuit::Rv32ImConfig;
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_sdk::{
    commit::commit_app_exe,
    config::{AggConfig, AppConfig, FullAggConfig, Halo2Config},
    Sdk, StdIn,
};
use axvm_transpiler::{transpiler::Transpiler, FromElf};
use clap::Parser;
use eyre::Result;

const NUM_PUBLIC_VALUES: usize = DEFAULT_MAX_NUM_PUBLIC_VALUES;

#[tokio::main]
async fn main() -> Result<()> {
    let cli_args = BenchmarkCli::parse();
    let app_fri_params = standard_fri_params_with_100_bits_conjectured_security(
        cli_args.app_log_blowup.unwrap_or(2),
    );
    let leaf_fri_params = standard_fri_params_with_100_bits_conjectured_security(
        cli_args.agg_log_blowup.unwrap_or(2),
    );
    let internal_fri_params = standard_fri_params_with_100_bits_conjectured_security(
        cli_args.internal_log_blowup.unwrap_or(2),
    );
    let root_fri_params = standard_fri_params_with_100_bits_conjectured_security(
        cli_args.root_log_blowup.unwrap_or(2),
    );
    let compiler_options = CompilerOptions {
        // For metric collection
        enable_cycle_tracker: true,
        ..Default::default()
    };

    // Must be larger than RangeTupleCheckerAir.height == 524288
    let max_segment_length = cli_args.max_segment_length.unwrap_or(1_000_000);

    let app_config = AppConfig {
        app_fri_params,
        app_vm_config: Rv32ImConfig::with_public_values_and_segment_len(
            NUM_PUBLIC_VALUES,
            max_segment_length,
        ),
        leaf_fri_params: leaf_fri_params.into(),
        compiler_options,
    };
    let full_agg_config = FullAggConfig {
        agg_config: AggConfig {
            max_num_user_public_values: NUM_PUBLIC_VALUES,
            leaf_fri_params,
            internal_fri_params,
            root_fri_params,
            compiler_options,
        },
        halo2_config: Halo2Config {
            verifier_k: 24,
            wrapper_k: None,
        },
    };

    let app_pk = Sdk.app_keygen(app_config)?;
    let full_agg_pk = Sdk.agg_keygen(full_agg_config)?;
    let elf = build_bench_program("fibonacci")?;
    let exe = AxVmExe::from_elf(
        elf,
        Transpiler::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let app_committed_exe = commit_app_exe(app_fri_params, exe);

    let n = 800_000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);
    run_with_metric_collection("OUTPUT_PATH", || {
        Sdk.generate_evm_proof(app_pk, app_committed_exe, full_agg_pk, stdin)
            .unwrap();
    });

    Ok(())
}
