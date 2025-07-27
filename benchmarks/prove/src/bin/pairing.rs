use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_sdk::{
    config::{SdkVmConfig, SdkVmCpuBuilder},
    Sdk, StdIn,
};
use openvm_stark_sdk::bench::run_with_metric_collection;

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    let vm_config =
        SdkVmConfig::from_toml(include_str!("../../../guest/pairing/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("pairing", &vm_config, None)?;
    let sdk = Sdk::new();
    let exe = sdk.transpile(elf, vm_config.transpiler()).unwrap();

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        args.bench_from_exe("pairing", SdkVmCpuBuilder, vm_config, exe, StdIn::default())
    })
}
