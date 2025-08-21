use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
#[cfg(not(feature = "cuda"))]
use openvm_sdk::config::SdkVmCpuBuilder as SdkVmBuilder;
#[cfg(feature = "cuda")]
use openvm_sdk::config::SdkVmGpuBuilder as SdkVmBuilder;
use openvm_sdk::{config::SdkVmConfig, StdIn};
use openvm_stark_sdk::bench::run_with_metric_collection;

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();
    let config = SdkVmConfig::from_toml(include_str!("../../../guest/revm_transfer/openvm.toml"))?
        .app_vm_config;
    let elf = args.build_bench_program("revm_transfer", &config, None)?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        args.bench_from_exe::<SdkVmBuilder, _>("revm_100_transfers", config, elf, StdIn::default())
    })
}
