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
        SdkVmConfig::from_toml(include_str!("../../../guest/bincode/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("bincode", &config, None)?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let file_data = include_bytes!("../../../guest/bincode/minecraft_savedata.bin");
        let stdin = StdIn::from_bytes(file_data);
        args.bench_from_exe::<SdkVmBuilder, _>("bincode", config, elf, stdin)
    })
}
