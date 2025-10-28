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
        SdkVmConfig::from_toml(include_str!("../../../guest/regex/openvm.toml"))?.app_vm_config;
    let elf = args.build_bench_program("regex", &config, None)?;
    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let data = include_str!("../../../guest/regex/regex_email.txt");

        let fe_bytes = data.to_owned().into_bytes();
        args.bench_from_exe::<SdkVmBuilder, _>(
            "regex_program",
            config,
            elf,
            StdIn::from_bytes(&fe_bytes),
        )
    })
}
