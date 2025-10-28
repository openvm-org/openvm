use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::arch::DEFAULT_MAX_NUM_PUBLIC_VALUES;
use openvm_sdk::{config::SdkVmConfig, Sdk, StdIn};
use openvm_stark_sdk::bench::run_with_metric_collection;

const NUM_PUBLIC_VALUES: usize = DEFAULT_MAX_NUM_PUBLIC_VALUES;

#[tokio::main]
async fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    // Must be larger than RangeTupleCheckerAir.height == 524288
    let max_segment_length = args.max_segment_length.unwrap_or(1_000_000);

    let mut config =
        SdkVmConfig::from_toml(include_str!("../../../guest/fibonacci/openvm.toml"))?.app_vm_config;
    config.as_mut().segmentation_limits.max_trace_height = max_segment_length;
    config.as_mut().num_public_values = NUM_PUBLIC_VALUES;

    let elf = args.build_bench_program("fibonacci", &config, None)?;
    let app_config = args.app_config(config);

    let sdk = Sdk::new(app_config)?;

    let n = 800_000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);
    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<_> {
        #[cfg(not(feature = "evm"))]
        let _proof = sdk.prover(elf)?.with_program_name("fib_e2e").prove(stdin)?;
        #[cfg(feature = "evm")]
        let _proof = sdk
            .evm_prover(elf)?
            .with_program_name("fib_e2e")
            .prove_evm(stdin)?;
        Ok(())
    })
}
