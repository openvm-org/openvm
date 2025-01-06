use std::sync::Arc;

use clap::Parser;
use eyre::Result;
use openvm_benchmarks::utils::BenchmarkCli;
use openvm_circuit::arch::instructions::{exe::VmExe, program::DEFAULT_MAX_NUM_PUBLIC_VALUES};
use openvm_native_recursion::halo2::utils::CacheHalo2ParamsReader;
use openvm_rv32im_circuit::Rv32ImConfig;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::{commit::commit_app_exe, prover::ContinuationProver, Sdk, StdIn};
use openvm_stark_sdk::bench::run_with_metric_collection;
use openvm_transpiler::{transpiler::Transpiler, FromElf};

const NUM_PUBLIC_VALUES: usize = DEFAULT_MAX_NUM_PUBLIC_VALUES;

#[tokio::main]
async fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    // Must be larger than RangeTupleCheckerAir.height == 524288
    let max_segment_length = args.max_segment_length.unwrap_or(1_000_000);

    let app_config = args.app_config(Rv32ImConfig::with_public_values_and_segment_len(
        NUM_PUBLIC_VALUES,
        max_segment_length,
    ));
    let agg_config = args.agg_config();

    let sdk = Sdk;
    let halo2_params_reader = CacheHalo2ParamsReader::new_with_default_params_dir();
    let app_pk = Arc::new(sdk.app_keygen(app_config)?);
    let full_agg_pk = sdk.agg_keygen(agg_config, &halo2_params_reader)?;
    let elf = args.build_bench_program("fibonacci")?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?;
    let app_committed_exe = commit_app_exe(app_pk.app_fri_params(), exe);

    let n = 800_000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);
    run_with_metric_collection("OUTPUT_PATH", || {
        let mut e2e_prover =
            ContinuationProver::new(&halo2_params_reader, app_pk, app_committed_exe, full_agg_pk);
        e2e_prover.set_program_name("fib_e2e");
        let _proof = e2e_prover.generate_proof_for_evm(stdin);
    });

    Ok(())
}
