use std::{path::PathBuf, str::FromStr, sync::Arc};

use clap::Parser;
use eyre::Result;
use num_bigint::BigUint;
use openvm_algebra_circuit::{Fp2Extension, ModularExtension};
use openvm_benchmarks_prove::util::BenchmarkCli;
use openvm_circuit::{
    arch::{instructions::exe::VmExe, SystemConfig},
    system::program::trace::VmCommittedExe,
};
use openvm_continuations::verifier::leaf::types::LeafVmVerifierInput;
use openvm_ecc_circuit::{WeierstrassExtension, P256_CONFIG, SECP256K1_CONFIG};
use openvm_native_circuit::{NativeConfig, NativeCpuBuilder, NATIVE_MAX_TRACE_HEIGHTS};
use openvm_native_recursion::halo2::utils::{CacheHalo2ParamsReader, DEFAULT_PARAMS_DIR};
use openvm_pairing_circuit::{PairingCurve, PairingExtension};
use openvm_pairing_guest::{
    bls12_381::BLS12_381_COMPLEX_STRUCT_NAME, bn254::BN254_COMPLEX_STRUCT_NAME,
};
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
        let segments = vm.executor().execute_metered(
            app_pk.leaf_committed_exe.exe.clone(),
            leaf_input.write_to_stream(),
            &executor_idx_to_air_idx,
            metered_ctx,
        )?;
        assert_eq!(segments.len(), 1);
        let estimated_trace_heights = &segments[0].trace_heights;
        println!("estimated_trace_heights: {:?}", estimated_trace_heights);

        // Tracegen without proving since leaf proofs take a while
        let state = vm
            .executor()
            .create_initial_state(&exe, leaf_input.write_to_stream());
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

    let bn_config = PairingCurve::Bn254.curve_config();
    let bls_config = PairingCurve::Bls12_381.curve_config();
    let vm_config = SdkVmConfig::builder()
        .system(SystemConfig::default().with_continuations().into())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .keccak(Default::default())
        .sha256(Default::default())
        .bigint(Default::default())
        .modular(ModularExtension::new(vec![
            BigUint::from_str("1000000000000000003").unwrap(),
            SECP256K1_CONFIG.modulus.clone(),
            SECP256K1_CONFIG.scalar.clone(),
            P256_CONFIG.modulus.clone(),
            P256_CONFIG.scalar.clone(),
            bn_config.modulus.clone(),
            bn_config.scalar.clone(),
            bls_config.modulus.clone(),
            bls_config.scalar.clone(),
            BigUint::from(2u32).pow(61) - BigUint::from(1u32),
            BigUint::from(7u32),
        ]))
        .fp2(Fp2Extension::new(vec![
            (
                BN254_COMPLEX_STRUCT_NAME.to_string(),
                bn_config.modulus.clone(),
            ),
            (
                BLS12_381_COMPLEX_STRUCT_NAME.to_string(),
                bls_config.modulus.clone(),
            ),
        ]))
        .ecc(WeierstrassExtension::new(vec![
            SECP256K1_CONFIG.clone(),
            P256_CONFIG.clone(),
            bn_config.clone(),
            bls_config.clone(),
        ]))
        .pairing(PairingExtension::new(vec![
            PairingCurve::Bn254,
            PairingCurve::Bls12_381,
        ]))
        .build();
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
