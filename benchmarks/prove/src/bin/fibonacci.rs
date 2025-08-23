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
        SdkVmConfig::from_toml(include_str!("../../../guest/fibonacci/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("fibonacci", &config, None)?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let n = 100_000u64;
        let mut stdin = StdIn::default();
        stdin.write(&n);
        args.bench_from_exe::<SdkVmBuilder, _>("fibonacci_program", config, elf, stdin)
    })
}
