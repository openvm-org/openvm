use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::arch::{
    execution_mode::metered::segment_ctx::SegmentationLimits, instructions::exe::VmExe,
    DEFAULT_MAX_NUM_PUBLIC_VALUES,
};
use openvm_native_circuit::NativeCpuBuilder;
use openvm_native_recursion::halo2::utils::{CacheHalo2ParamsReader, DEFAULT_PARAMS_DIR};
use openvm_sdk::{
    commit::commit_app_exe,
    config::{SdkVmConfig, SdkVmCpuBuilder},
    prover::EvmHalo2Prover,
    DefaultStaticVerifierPvHandler, Sdk, StdIn,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
};
use openvm_transpiler::FromElf;

const NUM_PUBLIC_VALUES: usize = DEFAULT_MAX_NUM_PUBLIC_VALUES;

#[tokio::main]
async fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    // Must be larger than RangeTupleCheckerAir.height == 524288
    let max_segment_length = args.max_segment_length.unwrap_or(1_000_000);

    let mut config =
        SdkVmConfig::from_toml(include_str!("../../../guest/fibonacci/openvm.toml"))?.app_vm_config;
    config.as_mut().set_segmentation_limits(
        SegmentationLimits::default().with_max_trace_height(max_segment_length as u32),
    );
    config.as_mut().num_public_values = NUM_PUBLIC_VALUES;

    let elf = args.build_bench_program("fibonacci", &config, None)?;
    let exe = VmExe::from_elf(elf, config.transpiler())?;
    let app_config = args.app_config(config);
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
    let app_committed_exe = commit_app_exe(app_pk.app_fri_params(), exe);

    let n = 800_000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);
    run_with_metric_collection("OUTPUT_PATH", || {
        let mut e2e_prover = EvmHalo2Prover::<BabyBearPoseidon2Engine, _, _>::new(
            &halo2_params_reader,
            SdkVmCpuBuilder,
            NativeCpuBuilder,
            app_pk,
            app_committed_exe,
            full_agg_pk,
            args.agg_tree_config,
        )?;
        e2e_prover.set_program_name("fib_e2e");
        e2e_prover.generate_proof_for_evm(stdin)
    })?;

    Ok(())
}
