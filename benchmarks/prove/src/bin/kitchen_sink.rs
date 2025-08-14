use std::sync::Arc;

use clap::Parser;
use eyre::Result;
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_continuations::verifier::leaf::types::LeafVmVerifierInput;
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_sdk::{
    config::SdkVmConfig,
    prover::vm::{new_local_prover, types::VmProvingKey},
    Sdk, StdIn, F, SC,
};
use openvm_stark_sdk::{
    bench::run_with_metric_collection, config::baby_bear_poseidon2::BabyBearPoseidon2Engine,
};

fn verify_native_max_trace_heights(
    sdk: &Sdk,
    app_exe: Arc<VmExe<F>>,
    leaf_vm_pk: Arc<VmProvingKey<SC, NativeConfig>>,
    num_children_leaf: usize,
) -> Result<()> {
    let app_proof = sdk.app_prover(app_exe)?.prove(StdIn::default())?;
    let leaf_inputs =
        LeafVmVerifierInput::chunk_continuation_vm_proof(&app_proof, num_children_leaf);
    let mut leaf_prover = new_local_prover::<BabyBearPoseidon2Engine, _>(
        NativeCpuBuilder,
        &leaf_vm_pk,
        sdk.app_pk().leaf_committed_exe.exe.clone(),
    )?;

    for leaf_input in leaf_inputs {
        let exe = leaf_prover.exe().clone();
        let vm = &mut leaf_prover.vm;
        let metered_ctx = vm.build_metered_ctx();
        let (segments, _) = vm
            .metered_interpreter(&exe)?
            .execute_metered(leaf_input.write_to_stream(), metered_ctx)?;
        assert_eq!(segments.len(), 1);
        let estimated_trace_heights = &segments[0].trace_heights;
        println!("estimated_trace_heights: {:?}", estimated_trace_heights);

        // Tracegen without proving since leaf proofs take a while
        let state = vm.create_initial_state(&exe, leaf_input.write_to_stream());
        vm.transport_init_memory_to_device(&state.memory);
        let mut interpreter = vm.preflight_interpreter(&exe)?;
        let out = vm.execute_preflight(&mut interpreter, state, None, estimated_trace_heights)?;
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
    let app_config = args.app_config(vm_config.clone());
    let elf = args.build_bench_program("kitchen-sink", &vm_config, None)?;
    let sdk = Sdk::new(app_config)?;
    let exe = sdk.convert_to_exe(elf)?;

    let agg_pk = sdk.agg_pk();
    // Verify that NATIVE_MAX_TRACE_HEIGHTS remains valid
    verify_native_max_trace_heights(
        &sdk,
        exe.clone(),
        agg_pk.leaf_vm_pk.clone(),
        args.agg_tree_config.num_children_leaf,
    )?;

    run_with_metric_collection("OUTPUT_PATH", || -> eyre::Result<()> {
        let mut prover = sdk.evm_prover(exe)?.with_program_name("kitchen_sink");
        let stdin = StdIn::default();
        let _proof = prover.prove_evm(stdin)?;
        Ok(())
    })
}
