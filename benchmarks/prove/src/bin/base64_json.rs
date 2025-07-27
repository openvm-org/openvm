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

    let config = SdkVmConfig::from_toml(include_str!("../../../guest/base64_json/openvm.toml"))?
        .app_vm_config;
    let elf = args.build_bench_program("base64_json", &config, None)?;
    let exe = VmExe::from_elf(elf, config.transpiler())?;

    run_with_metric_collection("OUTPUT_PATH", || -> Result<()> {
        let data = include_str!("../../../guest/base64_json/json_payload_encoded.txt");

        let fe_bytes = data.to_owned().into_bytes();
        args.bench_from_exe(
            "base64_json",
            SdkVmCpuBuilder,
            config,
            exe,
            StdIn::from_bytes(&fe_bytes),
        )
    })
}
