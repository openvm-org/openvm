use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::{arch::instructions::exe::VmExe, system::program::trace::VmCommittedExe};
use openvm_continuations::verifier::leaf::types::LeafVmVerifierInput;
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_recursion::halo2::utils::{CacheHalo2ParamsReader, DEFAULT_PARAMS_DIR};
use openvm_sdk::{
    commit::commit_app_exe,
    config::{SdkVmConfig, SdkVmCpuBuilder},
    keygen::AppProvingKey,
    prover::{
        vm::{new_local_prover, types::VmProvingKey},
        EvmHalo2Prover,
    },
    DefaultStaticVerifierPvHandler, Sdk, StdIn, SC,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
};
use openvm_transpiler::FromElf;

fn verify_native_max_trace_heights(
    sdk: &Sdk,
    app_pk: Arc<AppProvingKey<SdkVmConfig>>,
    app_committed_exe: Arc<VmCommittedExe<SC>>,
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    num_children_leaf: usize,
) -> Result<()> {
    let app_proof = sdk.generate_app_proof(
        SdkVmCpuBuilder,
        app_pk.clone(),
        app_committed_exe.clone(),
        StdIn::default(),
    )?;
    let leaf_inputs =
        LeafVmVerifierInput::chunk_continuation_vm_proof(&app_proof, num_children_leaf);
    let mut leaf_prover = new_local_prover::<BabyBearPoseidon2Engine, _>(
        NativeCpuBuilder,
        &leaf_vm_pk,
        &app_pk.leaf_committed_exe,
    )?;
    let executor_idx_to_air_idx = leaf_prover.vm.executor_idx_to_air_idx();

    for leaf_input in leaf_inputs {
        let exe = leaf_prover.exe().clone();
        let vm = &mut leaf_prover.vm;
        let metered_ctx = vm.build_metered_ctx();
        let (segments, _) = vm
            .executor()
            .metered_instance(&exe, &executor_idx_to_air_idx)?
            .execute_metered(leaf_input.write_to_stream(), metered_ctx)?;
        assert_eq!(segments.len(), 1);
        let estimated_trace_heights = &segments[0].trace_heights;
        println!("estimated_trace_heights: {:?}", estimated_trace_heights);

        // Tracegen without proving since leaf proofs take a while
        let state = vm.create_initial_state(&exe, leaf_input.write_to_stream());
        vm.transport_init_memory_to_device(&state.memory);
        let out = vm.execute_preflight(&exe, state, None, estimated_trace_heights)?;
        let actual_trace_heights = vm
            .generate_proving_ctx(out.system_records, out.record_arenas)?
            .per_air
            .into_iter()
            .map(|(_, air_ctx)| air_ctx.main_trace_height())
            .collect::<Vec<usize>>();
        println!("actual_trace_heights: {:?}", actual_trace_heights);

        actual_trace_heights
            .iter()
            .zip(NATIVE_MAX_TRACE_HEIGHTS)
            .for_each(|(&actual, &expected)| {
                assert!(
                    actual <= (expected as usize),
                    "Actual trace height {} exceeds expected height {}",
                    actual,
                    expected
                );
            });
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = BenchmarkCli::parse();

    let vm_config =
        SdkVmConfig::from_toml(include_str!("../../../guest/kitchen-sink/openvm.toml"))?
            .app_vm_config;
    let elf = args.build_bench_program("kitchen-sink", &vm_config, None)?;
    let exe = VmExe::from_elf(elf, vm_config.transpiler())?;

    let sdk = Sdk::new();
    let app_config = args.app_config(vm_config.clone());
    let app_pk = Arc::new(sdk.app_keygen(app_config)?);
    let app_committed_exe = commit_app_exe(app_pk.app_fri_params(), exe);

    let agg_config = args.agg_config();
    let halo2_params_reader = CacheHalo2ParamsReader::new(
        args.kzg_params_dir
            .clone()
            .unwrap_or(PathBuf::from(DEFAULT_PARAMS_DIR)),
    );
    let full_agg_pk = sdk.agg_keygen(
        agg_config,
        &halo2_params_reader,
        &DefaultStaticVerifierPvHandler,
    )?;

    // Verify that NATIVE_MAX_TRACE_HEIGHTS remains valid
    verify_native_max_trace_heights(
        &sdk,
        app_pk.clone(),
        app_committed_exe.clone(),
        full_agg_pk.agg_stark_pk.leaf_vm_pk.clone(),
        args.agg_tree_config.num_children_leaf,
    )?;

    run_with_metric_collection("OUTPUT_PATH", || {
        let mut prover = EvmHalo2Prover::<BabyBearPoseidon2Engine, _, _>::new(
            &halo2_params_reader,
            SdkVmCpuBuilder,
            NativeCpuBuilder,
            app_pk,
            app_committed_exe,
            full_agg_pk,
            args.agg_tree_config,
        )?;
        prover.set_program_name("kitchen_sink");
        let stdin = StdIn::default();
        prover.generate_proof_for_evm(stdin)
    })?;
    Ok(())
}
