use std::{slice::from_ref, sync::Arc};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_circuit::arch::instructions::DEFERRAL_AS;
use openvm_continuations::prover::DeferralCircuitProver;
use openvm_deferral_circuit::DeferralFn;
use openvm_stark_backend::{codec::Encode, StarkEngine, SystemParams};
use openvm_stark_sdk::{
    config::{
        app_params_with_100_bits_security, internal_params_with_100_bits_security,
        root_params_with_100_bits_security,
    },
    utils::setup_tracing,
};
use openvm_transpiler::elf::Elf;
use openvm_verify_stark_circuit::extension::{
    get_deferral_state, get_raw_deferral_results, verify_stark_deferral_fn,
};
use openvm_verify_stark_host::{
    vk::{VerificationBaseline, VmStarkVerifyingKey},
    VmStarkProof,
};

use crate::{
    config::{AggregationConfig, AggregationSystemParams, AppConfig, DEFAULT_APP_L_SKIP},
    prover::{DeferralProof, DeferralProver},
    DeferralInput, Sdk, StdIn,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuCircuitProver as VerifyCircuitProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
        type RootE = openvm_cuda_backend::BabyBearBn254Poseidon2GpuEngine;
    } else {
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuCircuitProver as VerifyCircuitProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
        type RootE = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    }
}

/// Creates a fibonacci SDK with standard test parameters.
fn make_fib_sdk() -> (Sdk, SystemParams, AggregationSystemParams) {
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();
    let sdk = Sdk::riscv32(app_params.clone(), agg_params.clone());
    (sdk, app_params, agg_params)
}

/// Generates a fibonacci VM STARK proof using the given SDK.
fn generate_fib_vm_stark_proof(fib_sdk: &Sdk) -> Result<(VmStarkProof, VerificationBaseline)> {
    let fib_elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let fib_exe = fib_sdk.convert_to_exe(fib_elf)?;
    let n = 100u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);
    Ok(fib_sdk.prove(fib_exe, stdin, &[])?)
}

/// Builds a DeferralProver from a base SDK.
fn make_deferral_prover(sdk: &Sdk, agg_params: &AggregationSystemParams) -> DeferralProver {
    let agg_prover = sdk.agg_prover();
    let ir_vk = agg_prover.internal_recursive_prover.get_vk();
    let ir_pcs_data = agg_prover
        .internal_recursive_prover
        .get_self_vk_pcs_data()
        .unwrap();
    let system_config = sdk.app_config().app_vm_config.as_ref().clone();
    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let num_user_pvs = system_config.num_public_values;
    let def_circuit_params = internal_params_with_100_bits_security();
    let deferred_verify_prover = VerifyProver::new::<E>(
        ir_vk,
        ir_pcs_data.commitment.into(),
        def_circuit_params,
        memory_dimensions,
        num_user_pvs,
        None,
        0,
    );
    let verify_stark_prover = VerifyCircuitProver::new(deferred_verify_prover);
    let hook_params = root_params_with_100_bits_security();
    let agg_config = AggregationConfig {
        params: agg_params.clone(),
    };
    DeferralProver::new(verify_stark_prover, agg_config, hook_params)
}

/// Builds a deferral-enabled verify-stark SDK from a fibonacci SDK and proof.
///
/// Returns the SDK, the verify-stark stdin, and the deferral input.
fn make_deferral_sdk(
    fib_sdk: &Sdk,
    fib_proof: VmStarkProof,
    fib_baseline: VerificationBaseline,
    app_params: SystemParams,
    agg_params: AggregationSystemParams,
) -> Result<(Sdk, StdIn, DeferralInput)> {
    let deferral_prover = make_deferral_prover(fib_sdk, &agg_params);
    let deferral_ext =
        deferral_prover.make_extension(vec![Arc::new(DeferralFn::new(verify_stark_deferral_fn))]);

    let fib_vk = VmStarkVerifyingKey {
        mvk: fib_sdk.agg_vk().as_ref().clone(),
        baseline: fib_baseline,
    };

    let raw_results = get_raw_deferral_results(&fib_vk, from_ref(&fib_proof))?;
    assert_eq!(raw_results.len(), 1);
    let input_commit: [u8; 32] = raw_results[0].input.clone().try_into().unwrap();
    let output_raw = &raw_results[0].output_raw;
    let app_exe_commit: [u8; 32] = output_raw[..32].try_into().unwrap();
    let app_vm_commit: [u8; 32] = output_raw[32..64].try_into().unwrap();
    let user_public_values = output_raw[64..].to_vec();
    let deferral_state = get_deferral_state(&fib_vk, from_ref(&fib_proof), 0)?;

    let mut vs_config = openvm_sdk_config::SdkVmConfig::riscv32();
    vs_config.deferral = Some(deferral_ext);
    vs_config.system.config.memory_config.addr_spaces[DEFERRAL_AS as usize].num_cells = 1 << 25;

    let vs_app_config = AppConfig::new(vs_config, app_params);
    let vs_sdk = Sdk::builder()
        .app_config(vs_app_config)
        .agg_params(agg_params)
        .deferral_prover(deferral_prover)
        .build()?;

    let mut vs_stdin = StdIn::default();
    vs_stdin.write(&app_exe_commit);
    vs_stdin.write(&app_vm_commit);
    vs_stdin.write(&user_public_values);
    vs_stdin.write(&input_commit);
    vs_stdin.deferrals = vec![deferral_state];

    let def_input = DeferralInput::from_inputs(&[fib_proof]);

    Ok((vs_sdk, vs_stdin, def_input))
}

#[test]
fn test_sdk_fibonacci() -> Result<()> {
    setup_tracing();
    let (sdk, _app_params, _agg_params) = make_fib_sdk();

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let app_exe = sdk.convert_to_exe(elf)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    #[cfg(not(feature = "evm-verify"))]
    {
        let mut evm_prover = sdk.evm_prover(app_exe)?;
        let proof = evm_prover.prove_unwrapped(stdin, &[])?;
        let vk = evm_prover.root_prover.0.get_vk();
        let engine = RootE::new(vk.inner.params.clone());
        engine.verify(&vk, &proof)?;
    }
    #[cfg(feature = "evm-verify")]
    {
        let evm_proof = sdk.prove_evm(app_exe, stdin, &[])?;
        let openvm_verifier = sdk.generate_halo2_verifier_solidity()?;
        let _gas_cost = Sdk::verify_evm_halo2_proof(&openvm_verifier, evm_proof)?;
    }

    Ok(())
}

#[test]
fn test_verify_stark_deferral() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let (vs_sdk, vs_stdin, def_input) =
        make_deferral_sdk(&fib_sdk, fib_proof, fib_baseline, app_params, agg_params)?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    let mut evm_prover = vs_sdk.evm_prover(vs_exe)?;
    let vs_proof = evm_prover.prove_unwrapped(vs_stdin, &[def_input])?;

    let vk = evm_prover.root_prover.0.get_vk();
    let engine = RootE::new(vk.inner.params.clone());
    engine.verify(&vk, &vs_proof)?;

    Ok(())
}

#[test]
fn test_verify_stark_with_deferral_child() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let (vs_sdk, vs_stdin, def_input) =
        make_deferral_sdk(&fib_sdk, fib_proof, fib_baseline, app_params, agg_params)?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    let (vs_proof, _) = vs_sdk.prove(vs_exe, vs_stdin, &[def_input])?;
    assert!(
        vs_proof.deferral_merkle_proofs.is_some(),
        "deferral-enabled verify-stark child proof must carry deferral merkle proofs",
    );
    let expected_def_hook_commit = vs_sdk
        .def_hook_commit()
        .expect("deferral-enabled SDK should expose a deferral hook commit");

    // ---- Step 5: Feed the encoded proof through the trait adapter ----
    let vs_agg_prover = vs_sdk.agg_prover();
    let vs_ir_vk = vs_agg_prover.internal_recursive_prover.get_vk();
    let vs_ir_pcs_data = vs_agg_prover
        .internal_recursive_prover
        .get_self_vk_pcs_data()
        .unwrap();
    let vs_system_config = vs_sdk.app_config().app_vm_config.as_ref().clone();

    // This nested verifier is intentionally constructed in deferral-aware mode because the
    // verify-stark child proof above was itself produced through a deferral-enabled SDK.
    let nested_verify_prover = VerifyProver::new::<E>(
        vs_ir_vk,
        vs_ir_pcs_data.commitment.into(),
        internal_params_with_100_bits_security(),
        vs_system_config.memory_config.memory_dimensions(),
        vs_system_config.num_public_values,
        Some(expected_def_hook_commit),
        0,
    );
    let nested_verify_circuit_prover = VerifyCircuitProver::new(nested_verify_prover);

    let encoded_vs_proof = vs_proof.encode_to_vec()?;
    let nested_def_proof = nested_verify_circuit_prover.prove(&encoded_vs_proof);

    let vk = nested_verify_circuit_prover.get_vk();
    let engine = E::new(vk.inner.params.clone());
    engine.verify(&vk, &nested_def_proof)?;

    Ok(())
}

#[test]
fn test_deferrals_enabled_without_usage() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let deferral_prover = make_deferral_prover(&fib_sdk, &agg_params);

    let sdk = Sdk::builder()
        .app_config(AppConfig::riscv32(app_params))
        .agg_params(agg_params.clone())
        .deferral_prover(deferral_prover)
        .build()?;

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let app_exe = sdk.convert_to_exe(elf)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    let mut evm_prover = sdk.evm_prover(app_exe)?;
    let proof = evm_prover.prove_unwrapped(stdin, &[])?;

    // ---- Step 3: Verify the final result ----
    let vk = evm_prover.root_prover.0.get_vk();
    let engine = RootE::new(vk.inner.params.clone());
    engine.verify(&vk, &proof)?;

    Ok(())
}

#[test]
fn test_prove_mixed_vm_def_depth_mismatch() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let (vs_sdk, vs_stdin, def_input) =
        make_deferral_sdk(&fib_sdk, fib_proof, fib_baseline, app_params, agg_params)?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    // ---- Step 1: Generate base VM and deferral proofs ----
    let agg_prover = vs_sdk.agg_prover();
    let app_proof = vs_sdk.app_prover(vs_exe)?.prove(vs_stdin)?;
    let (vm_proof, mut internal_layer_metadata) = agg_prover.prove_vm(app_proof)?;

    // We assume that the verify-stark program is small enough where only a single
    // internal_recursive layer is needed to fully aggregate its proof.
    assert_eq!(internal_layer_metadata.internal_recursive_layer, 1);

    let def_prover = vs_sdk.def_path_prover.unwrap();
    let def_hook_proofs = def_prover.deferral_prover.prove(&[def_input])?;
    let (def_proof, mut def_internal_recursive_layer) =
        def_prover.agg_prover.prove_def(def_hook_proofs)?;
    assert_eq!(def_internal_recursive_layer, 1);

    // ---- Step 2: Generate mixed proof with wrapped VM proof ----
    let mut wrapped_vm_metadata = internal_layer_metadata.clone();
    let mut wrapped_vm_proof = vm_proof.clone();
    for _ in 0..2 {
        wrapped_vm_proof = agg_prover.wrap_proof(wrapped_vm_proof, &mut wrapped_vm_metadata)?;
    }
    let wrapped_vm_mixed_proof = agg_prover.prove_mixed(
        wrapped_vm_proof,
        def_proof.clone(),
        &mut wrapped_vm_metadata,
        def_internal_recursive_layer,
    )?;

    // ---- Step 3: Generate mixed proof with wrapped deferral proof ----
    let wrapped_def_proof = match def_proof {
        DeferralProof::Present(mut p) => {
            for _ in 0..2 {
                p = agg_prover.wrap_def_inner(p, def_internal_recursive_layer)?;
                def_internal_recursive_layer += 1;
            }
            DeferralProof::Present(p)
        }
        DeferralProof::Absent(_) => panic!("expected DeferralProof::Present"),
    };
    let wrapped_def_mixed_proof = agg_prover.prove_mixed(
        vm_proof,
        wrapped_def_proof,
        &mut internal_layer_metadata,
        def_internal_recursive_layer,
    )?;

    // ---- Step 4: Verify mixed proofs ----
    let vk = agg_prover.internal_recursive_prover.get_vk();
    let engine = E::new(vk.inner.params.clone());
    engine.verify(&vk, &wrapped_vm_mixed_proof.inner)?;
    engine.verify(&vk, &wrapped_def_mixed_proof.inner)?;

    Ok(())
}

/// Cell-count profiling test for the static verifier circuit using a production root proof.
///
/// Root verifier params match `pipeline_cell_count_profiling` in static-verifier crate.
/// The root proof is generated from a full SDK aggregation pipeline and cached to disk.
///
/// Run with:
/// ```sh
/// OPENVM_CACHE_DIR=cache OPENVM_PROFILE_DIR=profile \
///   cargo nextest run --cargo-profile=fast -p openvm-sdk --features cuda,cell-profiling \
///   -- sdk_static_verifier_cell_profiling
/// ```
#[cfg(feature = "cell-profiling")]
#[test]
fn sdk_static_verifier_cell_profiling() -> Result<()> {
    use std::path::Path;

    use halo2_base::gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage};
    use openvm::platform::memory::MEM_SIZE;
    use openvm_continuations::{CommitBytes, RootSC};
    use openvm_stark_backend::{
        codec::{Decode, Encode},
        proof::Proof,
    };
    use openvm_static_verifier::{
        compute_dag_onion_commit,
        field::baby_bear::{BabyBearChip, BabyBearExtChip},
        log_heights_per_air_from_proof, StaticVerifierCircuit,
    };

    use crate::{
        config::{AggregationSystemParams, DEFAULT_APP_L_SKIP},
        keygen::dummy::compute_root_proof_heights,
        prover::{EvmProver, RootProver},
        Sdk, StdIn,
    };

    // Root verifier params matching pipeline_cell_count_profiling in static-verifier
    let root_params = root_params_with_100_bits_security();
    let cache_dir = std::env::var("OPENVM_CACHE_DIR").unwrap_or_else(|_| "cache".to_string());
    std::fs::create_dir_all(&cache_dir)?;

    let proof_path = format!("{cache_dir}/sdk_root_proof.bin");
    let vk_path = format!("{cache_dir}/sdk_root_vk.bin");
    let commit_path = format!("{cache_dir}/sdk_onion_commit.bin");

    let (root_vk, root_proof, onion_commit) =
        if Path::new(&proof_path).exists() && Path::new(&vk_path).exists() {
            eprintln!("Loading cached root proof from {cache_dir}/");
            let proof_bytes = std::fs::read(&proof_path)?;
            let root_proof = Proof::<RootSC>::decode_from_bytes(&proof_bytes)?;

            let vk_bytes = std::fs::read(&vk_path)?;
            let root_vk = bitcode::deserialize(&vk_bytes)
                .map_err(|e| eyre::eyre!("failed to deserialize root VK: {e}"))?;

            let commit_bytes: [u8; 32] = std::fs::read(&commit_path)?
                .try_into()
                .map_err(|_| eyre::eyre!("invalid commit file"))?;
            let onion_commit = CommitBytes::new(commit_bytes).into();

            (root_vk, root_proof, onion_commit)
        } else {
            eprintln!("Generating root proof via SDK pipeline (this takes a while)...");
            let n_stack = 19;
            let app_params = openvm_stark_sdk::config::app_params_with_100_bits_security(
                DEFAULT_APP_L_SKIP + n_stack,
            );
            let agg_params = AggregationSystemParams::default();

            let elf = Elf::decode(
                include_bytes!("../programs/examples/fibonacci.elf"),
                MEM_SIZE as u32,
            )?;
            let sdk = Sdk::riscv32(app_params, agg_params);
            let app_exe = sdk.convert_to_exe(elf)?;

            // Compute trace heights for root prover with profiling params
            let system_config = sdk.app_config().app_vm_config.as_ref();
            let agg_prover = sdk.agg_prover();
            let (trace_heights, root_pk) = compute_root_proof_heights::<E, _>(
                sdk.app_vm_builder().clone(),
                &sdk.app_pk().app_vm_pk,
                agg_prover.clone(),
                root_params.clone(),
                None,
            )?;

            let ir_vk = agg_prover.internal_recursive_prover.get_vk();
            let ir_pcs_data = agg_prover
                .internal_recursive_prover
                .get_self_vk_pcs_data()
                .unwrap();
            let vk_commit: CommitBytes = ir_pcs_data.commitment.into();
            let onion_commit = compute_dag_onion_commit(&ir_vk);

            let memory_dimensions = system_config.memory_config.memory_dimensions();
            let num_user_pvs = system_config.num_public_values;

            let root_prover = Arc::new(RootProver::from_pk(
                ir_vk,
                vk_commit,
                root_pk,
                memory_dimensions,
                num_user_pvs,
                None,
                Some(trace_heights),
            ));

            let mut evm_prover = EvmProver::<E, _>::new(
                sdk.app_vm_builder().clone(),
                &sdk.app_pk().app_vm_pk,
                app_exe,
                agg_prover,
                None,
                root_prover.clone(),
                None,
            )?;

            let n = 100u64;
            let mut stdin = StdIn::default();
            stdin.write(&n);

            let root_proof = evm_prover.prove_unwrapped(stdin, &[])?;
            let root_vk_arc = root_prover.0.get_vk();
            let root_vk = root_vk_arc.as_ref().clone();

            // Verify the root proof
            let engine = RootE::new(root_vk.inner.params.clone());
            engine.verify(&root_vk, &root_proof)?;

            // Cache to disk
            eprintln!("Caching root proof to {cache_dir}/");
            std::fs::write(&proof_path, root_proof.encode_to_vec()?)?;
            std::fs::write(
                &vk_path,
                bitcode::serialize(&root_vk)
                    .map_err(|e| eyre::eyre!("failed to serialize root VK: {e}"))?,
            )?;
            std::fs::write(&commit_path, CommitBytes::from(onion_commit).as_slice())?;

            (root_vk, root_proof, onion_commit)
        };

    // Run static verifier cell profiling
    eprintln!("Running static verifier cell profiling...");
    let log_heights = log_heights_per_air_from_proof(&root_proof);

    let circuit = StaticVerifierCircuit::try_new(root_vk, onion_commit, &log_heights)
        .expect("Failed to construct StaticVerifierCircuit");

    let profile_dir = std::env::var("OPENVM_PROFILE_DIR").unwrap_or_else(|_| "profile".to_string());
    std::env::set_var("OPENVM_PROFILE_DIR", &profile_dir);

    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(22)
        .use_lookup_bits(21)
        .use_instance_columns(0);
    let range = builder.range_chip();
    let ext_chip = BabyBearExtChip::new(BabyBearChip::new(Arc::new(range)));
    let ctx = builder.main(0);

    let initial_cells = ctx.advice.len();
    circuit.populate_verify_stark_constraints(ctx, &ext_chip, &root_proof);
    let final_cells = ctx.advice.len();
    eprintln!(
        "Static verifier cell count: {} (delta: {})",
        final_cells,
        final_cells - initial_cells
    );
    assert!(
        final_cells > initial_cells,
        "expected advice cells to increase"
    );

    Ok(())
}
