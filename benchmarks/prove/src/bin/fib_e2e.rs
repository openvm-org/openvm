use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::arch::{instructions::exe::VmExe, DEFAULT_MAX_NUM_PUBLIC_VALUES};
use openvm_native_recursion::halo2::utils::{CacheHalo2ParamsReader, DEFAULT_PARAMS_DIR};
use openvm_rv32im_circuit::Rv32ImConfig;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_sdk::{
    commit::commit_app_exe, prover::EvmHalo2Prover, DefaultStaticVerifierPvHandler, Sdk, StdIn,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
};
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
    let elf = args.build_bench_program("fibonacci", &app_config.app_vm_config, None)?;

    let agg_config = args.agg_config();

    let sdk = Sdk::new();
    let halo2_params_reader = CacheHalo2ParamsReader::new(
        args.kzg_params_dir
            .clone()
            .unwrap_or(PathBuf::from(DEFAULT_PARAMS_DIR)),
    );
    let app_pk = Arc::new(sdk.app_keygen(app_config)?);
    let full_agg_pk = sdk.agg_keygen(
        agg_config,
        &halo2_params_reader,
        &DefaultStaticVerifierPvHandler,
    )?;
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
        let mut e2e_prover = EvmHalo2Prover::<_, BabyBearPoseidon2Engine>::new(
            &halo2_params_reader,
            app_pk,
            app_committed_exe,
            full_agg_pk,
            args.agg_tree_config,
        );
        e2e_prover.set_program_name("fib_e2e");
        let _proof = e2e_prover.generate_proof_for_evm(stdin);
    });

    Ok(())
}
