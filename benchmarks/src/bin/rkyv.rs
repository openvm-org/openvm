use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::{bench_from_exe, build_bench_program, BenchmarkCli};
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_native_compiler::conversion::CompilerOptions;
use openvm_rv32im_circuit::Rv32ImConfig;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::{config::AppConfig, StdIn};
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::fri_params::standard_fri_params_with_100_bits_conjectured_security,
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};

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
        app_fri_params: app_fri_params.into(),
        app_vm_config: Rv32ImConfig::default(),
        leaf_fri_params: leaf_fri_params.into(),
        compiler_options,
    };

    let elf = build_bench_program("rkyv")?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let file_data = include_bytes!("../../programs/rkyv/minecraft_savedata.bin");
        let stdin = StdIn::from_bytes(file_data);
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
