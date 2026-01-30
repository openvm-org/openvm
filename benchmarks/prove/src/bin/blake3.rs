use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_sdk::{
    config::{SdkVmBuilder, SdkVmConfig},
    StdIn,
};
use openvm_stark_sdk::bench::run_with_metric_collection;

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    let config =
        SdkVmConfig::from_toml(include_str!("../../../guest/blake3/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("blake3", &config, None)?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let stdin = StdIn::default();
        args.bench_from_exe::<SdkVmBuilder, _>("blake3_program", config, elf, stdin)
    })
}
