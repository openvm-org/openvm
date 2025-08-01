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

    let config =
        SdkVmConfig::from_toml(include_str!("../../../guest/fibonacci/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("fibonacci", &config, None)?;
    let exe = VmExe::from_elf(elf, config.transpiler())?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let n = 100_000u64;
        let mut stdin = StdIn::default();
        stdin.write(&n);
        args.bench_from_exe("fibonacci_program", SdkVmCpuBuilder, config, exe, stdin)
    })
}
