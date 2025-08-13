use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_sdk::{
    config::{SdkVmConfig, SdkVmCpuBuilder},
    StdIn,
};
use openvm_stark_sdk::bench::run_with_metric_collection;
use openvm_transpiler::FromElf;

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    let vm_config =
        SdkVmConfig::from_toml(include_str!("../../../guest/pairing/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("pairing", &vm_config, None)?;
    let exe = VmExe::from_elf(elf, vm_config.transpiler()).unwrap();

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        args.bench_from_exe::<SdkVmCpuBuilder, _>("pairing", vm_config, exe, StdIn::default())
    })
}
