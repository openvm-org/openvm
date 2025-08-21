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

    let vm_config =
        SdkVmConfig::from_toml(include_str!("../../../guest/pairing/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("pairing", &vm_config, None)?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        args.bench_from_exe::<SdkVmBuilder, _>("pairing", vm_config, elf, StdIn::default())
    })
}
