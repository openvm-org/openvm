use clap::Parser;
use openvm_benchmarks_prove::{default_bench_app_params, BenchmarkCli};
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_sdk::{
    config::{AggregationSystemParams, AppConfig},
    Sdk, StdIn,
};
use openvm_sdk_config::{SdkVmConfig, TranspilerConfig};
use openvm_stark_sdk::{
    bench::run_with_metric_collection,
    config::{internal_params_with_100_bits_security, leaf_params_with_100_bits_security},
};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE, FromElf};

fn main() -> eyre::Result<()> {
    let args = BenchmarkCli::parse();
    let mut vm_config =
        SdkVmConfig::from_toml(include_str!("../../../guest/fibonacci/openvm.toml"))?;
    args.apply_config(&mut vm_config);

    let elf = Elf::decode(
        include_bytes!("../../../guest/fibonacci/elf/openvm-fibonacci-program.elf"),
        MEM_SIZE as u32,
    )?;

    let n = 800_000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        let exe = VmExe::from_elf(elf, vm_config.transpiler())?;
        let app_config = AppConfig::new(vm_config, default_bench_app_params());
        let agg_params = AggregationSystemParams {
            leaf: leaf_params_with_100_bits_security(),
            internal: internal_params_with_100_bits_security(),
        };
        let sdk = Sdk::new(app_config, agg_params)?;
        let _proof = sdk.prove_evm(exe, stdin, &[])?;
        Ok(())
    })
}
