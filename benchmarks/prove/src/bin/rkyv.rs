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
        SdkVmConfig::from_toml(include_str!("../../../guest/rkyv/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("rkyv", &config, None)?;
    let exe = VmExe::from_elf(elf, config.transpiler())?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let file_data = include_bytes!("../../../guest/rkyv/minecraft_savedata.bin");
        let stdin = StdIn::from_bytes(file_data);
        args.bench_from_exe("rkyv", SdkVmCpuBuilder, config, exe, stdin)
    })
}
