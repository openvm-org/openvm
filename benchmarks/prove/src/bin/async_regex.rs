use std::env::var;

use clap::Parser;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_benchmarks_utils::get_programs_dir;
use openvm_sdk::{
    config::{SdkVmBuilder, SdkVmConfig},
    prover::AsyncAppProver,
    DefaultStarkEngine, Sdk, StdIn, F,
};
use openvm_stark_sdk::config::setup_tracing;

#[tokio::main]
async fn main() -> eyre::Result<()> {
    setup_tracing();
    let args = BenchmarkCli::parse();
    let mut config = SdkVmConfig::from_toml(include_str!("../../../guest/regex/openvm.toml"))?;
    if let Some(max_height) = args.max_segment_length {
        config
            .app_vm_config
            .as_mut()
            .segmentation_limits
            .max_trace_height = max_height;
    }
    if let Some(max_cells) = args.segment_max_cells {
        config.app_vm_config.as_mut().segmentation_limits.max_cells = max_cells;
    }

    let sdk = Sdk::new(config)?;

    let manifest_dir = get_programs_dir().join("regex");
    let elf = sdk.build(Default::default(), manifest_dir, &None, None)?;
    let app_exe = sdk.convert_to_exe(elf)?;

    let data = include_str!("../../../guest/regex/regex_email.txt");
    let fe_bytes = data.to_owned().into_bytes();
    let input = StdIn::<F>::from_bytes(&fe_bytes);

    let (app_pk, _app_vk) = sdk.app_keygen();

    let max_concurrency: usize = var("MAX_CONCURRENCY").map(|m| m.parse()).unwrap_or(Ok(1))?;

    let prover = AsyncAppProver::<DefaultStarkEngine, _>::new(
        SdkVmBuilder,
        app_pk.app_vm_pk.clone(),
        app_exe,
        app_pk.leaf_verifier_program_commit(),
        max_concurrency,
    )?;
    let _proof = prover.prove(input).await?;

    Ok(())
}
