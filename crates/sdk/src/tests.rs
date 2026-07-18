#[cfg(feature = "rvr")]
use std::fs;
use std::{slice::from_ref, sync::Arc};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
#[cfg(feature = "rvr")]
use openvm_circuit::arch::ExecutionOutcome;
use openvm_circuit::arch::{instructions::exe::VmExe, U16_CELL_SIZE};
use openvm_continuations::prover::DeferralCircuitProver;
use openvm_sdk_config::{
    deferral::{DeferralConfig, SupportedDeferral},
    SdkVmConfig,
};
use openvm_stark_backend::{codec::Encode, StarkEngine, SystemParams};
use openvm_stark_sdk::{
    config::{
        app_params_with_100_bits_security, hook_params_with_100_bits_security,
        internal_params_with_100_bits_security, leaf_params_with_100_bits_security,
        root_params_with_100_bits_security,
    },
    utils::setup_tracing,
};
use openvm_transpiler::elf::Elf;
use openvm_verify_stark_circuit::{
    default_verify_stark_circuit_params,
    extension::{get_deferral_state, get_raw_deferral_results},
};
use openvm_verify_stark_host::{
    vk::{VerificationBaseline, VmStarkVerifyingKey},
    VmStarkProof,
};

use crate::{
    builder::GenericSdkBuilder,
    config::{
        AggregationConfig, AggregationSystemParams, AggregationTreeConfig, AppConfig,
        DEFAULT_APP_L_SKIP,
    },
    prover::{DeferralAggProver, DeferralHookCommits, DeferralProof, MultiDeferralCircuitProver},
    DeferralInput, Sdk, StdIn, F,
};
#[cfg(feature = "rvr")]
use crate::{
    compiled::{load_metered_artifact_metadata, metered_artifact_metadata_path},
    ExecutableInput,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyGpuCircuitProver as VerifyCircuitProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
        #[cfg(all(feature = "root-prover", any(not(feature = "evm-verify"), feature = "cell-profiling")))]
        type RootE = openvm_cuda_backend::BabyBearBn254Poseidon2GpuEngine;
    } else {
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuProver as VerifyProver;
        use openvm_verify_stark_circuit::prover::DeferredVerifyCpuCircuitProver as VerifyCircuitProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
        #[cfg(all(feature = "root-prover", any(not(feature = "evm-verify"), feature = "cell-profiling")))]
        type RootE = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    }
}

/// Default deferral idx for the verify-stark deferral circuit.
const DEFAULT_VERIFY_STARK_DEF_IDX: usize = 0;

/// Returns app, aggregation, and root params, allowing tests to override them via env vars.
fn get_params() -> (SystemParams, AggregationSystemParams, SystemParams) {
    let n_stack = 19;
    let app_params = get_params_from_env(
        "APP_PARAMS_OVERRIDE",
        app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack),
    );
    let agg_params = AggregationSystemParams {
        leaf: get_params_from_env("LEAF_PARAMS_OVERRIDE", leaf_params_with_100_bits_security()),
        internal: get_params_from_env(
            "INTERNAL_PARAMS_OVERRIDE",
            internal_params_with_100_bits_security(),
        ),
    };
    let root_params =
        get_params_from_env("ROOT_PARAMS_OVERRIDE", root_params_with_100_bits_security());

    (app_params, agg_params, root_params)
}

/// Creates a fibonacci SDK with standard test parameters.
fn make_fib_sdk() -> (Sdk, SystemParams, AggregationSystemParams) {
    let (app_params, agg_params, _root_params) = get_params();
    let mut sdk_builder =
        GenericSdkBuilder::new().app_config(AppConfig::riscv64(app_params.clone()));
    sdk_builder = sdk_builder.agg_params(agg_params.clone());
    #[cfg(feature = "root-prover")]
    {
        sdk_builder = sdk_builder.root_params(_root_params);
    }
    (sdk_builder.build().unwrap(), app_params, agg_params)
}

/// Reads a `SystemParams` JSON override from `env_var`, or returns `default`.
fn get_params_from_env(env_var: &str, default: SystemParams) -> SystemParams {
    match std::env::var(env_var) {
        Ok(s) => {
            eprintln!("getting params from env {env_var}");
            serde_json::from_str(&s).unwrap()
        }
        Err(_) => default,
    }
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

/// Builds the standard riscv64 SDK VM config with the supplied deferral config enabled.
fn riscv64_config_with_deferral(deferral: DeferralConfig) -> SdkVmConfig {
    SdkVmConfig::builder()
        .system(Default::default())
        .rv64i(Default::default())
        .rv64m(Default::default())
        .io(Default::default())
        .deferral(deferral)
        .build()
        .optimize()
}

/// Builds one verify-stark deferral circuit prover for `sdk` and `def_idx`.
fn make_verify_stark_circuit_prover(
    sdk: &Sdk,
    def_circuit_params: SystemParams,
    def_idx: usize,
) -> VerifyCircuitProver {
    let agg_prover = sdk.agg_prover();
    let ir_vk = agg_prover.internal_recursive_prover.get_vk();
    let ir_pcs_data = agg_prover
        .internal_recursive_prover
        .get_self_vk_pcs_data()
        .unwrap();
    let system_config = sdk.app_config().app_vm_config.as_ref().clone();
    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let num_user_pvs = system_config.num_public_values;
    let deferred_verify_prover = VerifyProver::new::<E>(
        ir_vk,
        ir_pcs_data.commitment.into(),
        def_circuit_params,
        memory_dimensions,
        num_user_pvs,
        None,
        def_idx,
    );
    VerifyCircuitProver::new(deferred_verify_prover)
}

/// Builds a MultiDeferralCircuitProver from a base SDK with `num_deferral_circuits` copies of the
/// verify-stark deferral circuit.
fn make_multi_deferral_circuit_prover_with_count(
    sdk: &Sdk,
    agg_params: &AggregationSystemParams,
    def_circuit_params: SystemParams,
    num_deferral_circuits: usize,
) -> MultiDeferralCircuitProver {
    assert!(num_deferral_circuits > 0);
    let verify_stark_prover = make_verify_stark_circuit_prover(
        sdk,
        def_circuit_params.clone(),
        DEFAULT_VERIFY_STARK_DEF_IDX,
    );
    let hook_params = hook_params_with_100_bits_security();
    let agg_config = AggregationConfig {
        params: agg_params.clone(),
    };
    let mut multi_deferral_circuit_prover =
        MultiDeferralCircuitProver::new(verify_stark_prover, agg_config, hook_params);
    for def_idx in 1..num_deferral_circuits {
        multi_deferral_circuit_prover = multi_deferral_circuit_prover.with_prover(
            make_verify_stark_circuit_prover(sdk, def_circuit_params.clone(), def_idx),
        );
    }
    multi_deferral_circuit_prover
}

/// Builds a verify-stark SDK with one deferral slot.
fn make_verify_stark_sdk(
    fib_sdk: &Sdk,
    app_params: SystemParams,
    agg_params: AggregationSystemParams,
) -> Result<Sdk> {
    make_verify_stark_sdk_with_count(
        fib_sdk,
        app_params,
        agg_params,
        default_verify_stark_circuit_params(),
        1,
    )
}

/// Builds a verify-stark SDK with `num_deferral_circuits` deferral slots.
fn make_verify_stark_sdk_with_count(
    fib_sdk: &Sdk,
    app_params: SystemParams,
    agg_params: AggregationSystemParams,
    def_circuit_params: SystemParams,
    num_deferral_circuits: usize,
) -> Result<Sdk> {
    let multi_deferral_circuit_prover = make_multi_deferral_circuit_prover_with_count(
        fib_sdk,
        &agg_params,
        def_circuit_params,
        num_deferral_circuits,
    );
    let supported_deferrals = vec![SupportedDeferral::VerifyStark; num_deferral_circuits];
    let deferral_config = multi_deferral_circuit_prover.make_config(supported_deferrals);

    let vm_config = riscv64_config_with_deferral(deferral_config);

    let sdk = Sdk::builder()
        .app_config(AppConfig::new(vm_config, app_params))
        .agg_params(agg_params)
        .multi_deferral_circuit_prover(multi_deferral_circuit_prover)
        .build()?;
    Ok(sdk)
}

/// Builds a verify-stark SDK that can recursively verify proofs produced by the same SDK.
fn make_recursive_verify_stark_sdk(
    app_params: SystemParams,
    agg_params: AggregationSystemParams,
) -> Result<Sdk> {
    let vm_config = SdkVmConfig::riscv64();
    let memory_dimensions = vm_config.system.config.memory_config.memory_dimensions();
    let num_user_pvs = vm_config.system.config.num_public_values;
    let deferral_agg_prover = DeferralAggProver::verify_stark(
        &agg_params,
        hook_params_with_100_bits_security(),
        memory_dimensions,
        num_user_pvs,
    );
    let deferral_config = deferral_agg_prover
        .multi_deferral_circuit_prover
        .make_config(vec![SupportedDeferral::VerifyStark]);
    let vm_config = riscv64_config_with_deferral(deferral_config);

    let sdk = Sdk::builder()
        .app_config(AppConfig::new(vm_config, app_params))
        .agg_params(agg_params)
        .deferral_agg_prover(deferral_agg_prover)
        .build()?;
    Ok(sdk)
}

/// Builds stdin and deferral input for a single verify-stark deferral proof.
fn make_verify_stark_inputs(
    child_sdk: &Sdk,
    child_proof: &VmStarkProof,
    child_baseline: VerificationBaseline,
    vs_sdk: &Sdk,
) -> Result<(StdIn, DeferralInput)> {
    let (stdin, mut def_inputs) = make_verify_stark_inputs_for_indices(
        child_sdk,
        child_proof,
        child_baseline,
        vs_sdk,
        &[DEFAULT_VERIFY_STARK_DEF_IDX],
        1,
    )?;
    Ok((stdin, def_inputs.pop().unwrap()))
}

/// Builds stdin and deferral inputs for selected verify-stark deferral indices. Assumes
/// that the verify-stark circuit at each index is identical.
fn make_verify_stark_inputs_for_indices(
    child_sdk: &Sdk,
    child_proof: &VmStarkProof,
    child_baseline: VerificationBaseline,
    vs_sdk: &Sdk,
    present_def_indices: &[usize],
    num_deferral_circuits: usize,
) -> Result<(StdIn, Vec<DeferralInput>)> {
    let child_vk = VmStarkVerifyingKey {
        mvk: child_sdk.agg_vk().as_ref().clone(),
        baseline: child_baseline,
    };

    let mut verify_stark_cached_commits =
        vs_sdk.deferral_circuit_cached_commits(DEFAULT_VERIFY_STARK_DEF_IDX)?;
    assert_eq!(verify_stark_cached_commits.len(), 1);
    let verify_stark_cached_commit = verify_stark_cached_commits.pop().unwrap().into();

    let raw_results =
        get_raw_deferral_results(&child_vk, from_ref(child_proof), verify_stark_cached_commit)?;
    assert_eq!(raw_results.len(), 1);
    let input_commit: [u8; 32] = raw_results[0].input.clone().try_into().unwrap();
    let output_raw = &raw_results[0].output_raw;
    let app_exe_commit: [u8; 32] = output_raw[..32].try_into().unwrap();
    let app_vm_commit: [u8; 32] = output_raw[32..64].try_into().unwrap();

    let user_public_values = collapse_user_public_values(&output_raw[64..]);

    let mut stdin = StdIn::default();
    stdin.write(&app_exe_commit);
    stdin.write(&app_vm_commit);
    stdin.write(&user_public_values);
    stdin.write(&input_commit);
    stdin.deferrals = vec![Default::default(); num_deferral_circuits];

    let proof_input = DeferralInput::from_inputs(from_ref(child_proof));
    let mut def_inputs = vec![DeferralInput::default(); num_deferral_circuits];

    for &def_idx in present_def_indices {
        assert!(def_idx < num_deferral_circuits);
        stdin.deferrals[def_idx] = get_deferral_state(
            &child_vk,
            from_ref(child_proof),
            verify_stark_cached_commit,
            def_idx as u32,
        )?;
        def_inputs[def_idx] = proof_input.clone();
    }

    Ok((stdin, def_inputs))
}

/// Converts byte-expanded BabyBear public values back to raw user public value bytes.
fn collapse_user_public_values(expanded: &[u8]) -> Vec<u8> {
    const F_NUM_BYTES: usize = core::mem::size_of::<u32>();
    assert!(expanded.len().is_multiple_of(F_NUM_BYTES));
    let mut user_public_values = Vec::with_capacity(expanded.len() / F_NUM_BYTES * U16_CELL_SIZE);
    for bytes in expanded.chunks_exact(F_NUM_BYTES) {
        assert_eq!(&bytes[U16_CELL_SIZE..], &[0; F_NUM_BYTES - U16_CELL_SIZE]);
        user_public_values.extend_from_slice(&bytes[..U16_CELL_SIZE]);
    }
    user_public_values
}

#[test]
fn collapse_user_public_values_preserves_u16_cells() {
    let expanded = [0x34, 0x12, 0, 0, 0xcd, 0xab, 0, 0];
    assert_eq!(
        collapse_user_public_values(&expanded),
        [0x34, 0x12, 0xcd, 0xab]
    );
}

/// Proves `exe` with the given inputs and verifies the resulting proof. The exact prover path
/// depends on which of `root-prover` / `evm-verify` features are enabled:
///   * neither: STARK proof via `sdk.prove`, verified with the aggregation VK
///   * `root-prover` without `evm-verify`: root proof via `evm_prover_without_halo2`
///   * `evm-verify`: EVM proof via `sdk.prove_evm`, verified against the halo2 verifier
fn prove_and_verify_e2e(
    sdk: &Sdk,
    exe: Arc<VmExe<F>>,
    stdin: StdIn,
    def_inputs: &[DeferralInput],
) -> Result<()> {
    #[cfg(not(feature = "root-prover"))]
    {
        let (proof, baseline) = sdk.prove(exe, stdin, def_inputs)?;
        Sdk::verify_proof((*sdk.agg_vk()).clone(), baseline, &proof)?;
    }
    #[cfg(all(feature = "root-prover", not(feature = "evm-verify")))]
    {
        let mut evm_prover = sdk.evm_prover_without_halo2(exe)?;
        let proof = evm_prover.prove_root(stdin, def_inputs)?;
        let vk = evm_prover.root_prover.0.get_vk();
        let engine = RootE::new(vk.inner.params.clone());
        engine.verify(&vk, &proof)?;
    }
    #[cfg(feature = "evm-verify")]
    {
        let app_commit = sdk.app_commit(exe.clone())?;
        let evm_proof = sdk.prove_evm(exe, stdin, def_inputs)?;
        let openvm_verifier = sdk.generate_halo2_verifier_solidity()?;
        let _gas_cost = Sdk::verify_evm_halo2_proof(&openvm_verifier, evm_proof, Some(app_commit))?;
    }
    Ok(())
}

#[test]
fn test_sdk_fibonacci() -> Result<()> {
    setup_tracing();
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let app_exe = sdk.convert_to_exe(elf)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    prove_and_verify_e2e(&sdk, app_exe, stdin, &[])
}

#[test]
fn test_verify_stark_deferral() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let vs_sdk = make_verify_stark_sdk(&fib_sdk, app_params, agg_params)?;
    let (vs_stdin, def_input) =
        make_verify_stark_inputs(&fib_sdk, &fib_proof, fib_baseline, &vs_sdk)?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    prove_and_verify_e2e(&vs_sdk, vs_exe, vs_stdin, &[def_input])
}

#[test]
fn test_verify_many_deferrals() -> Result<()> {
    setup_tracing();
    const NUM_DEFERRAL_CIRCUITS: usize = 5;

    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    // Use non-default params for better test coverage
    let def_circuit_params = leaf_params_with_100_bits_security();
    let vs_sdk = make_verify_stark_sdk_with_count(
        &fib_sdk,
        app_params,
        agg_params,
        def_circuit_params,
        NUM_DEFERRAL_CIRCUITS,
    )?;
    let (vs_stdin, def_inputs) = make_verify_stark_inputs_for_indices(
        &fib_sdk,
        &fib_proof,
        fib_baseline,
        &vs_sdk,
        &[0, 1, 3, 4],
        NUM_DEFERRAL_CIRCUITS,
    )?;

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-many.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = vs_sdk.convert_to_exe(vs_elf)?;

    prove_and_verify_e2e(&vs_sdk, vs_exe, vs_stdin, &def_inputs)
}

#[test]
fn test_verify_stark_path_sdk_can_verify_own_proofs() -> Result<()> {
    setup_tracing();
    let (app_params, agg_params, _) = get_params();
    let sdk = make_recursive_verify_stark_sdk(app_params, agg_params)?;
    let agg_vk = sdk.agg_vk().as_ref().clone();

    let vs_elf = Elf::decode(
        include_bytes!("../programs/examples/verify-stark.elf"),
        MEM_SIZE as u32,
    )?;
    let vs_exe = sdk.convert_to_exe(vs_elf)?;

    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&sdk)?;
    assert!(fib_proof.deferral_merkle_proofs.is_some(),);
    Sdk::verify_proof(agg_vk.clone(), fib_baseline.clone(), &fib_proof)?;

    let (vs_stdin, def_input) = make_verify_stark_inputs(&sdk, &fib_proof, fib_baseline, &sdk)?;
    let (vs_proof, vs_baseline) = sdk.prove(vs_exe.clone(), vs_stdin, &[def_input])?;
    assert!(vs_proof.deferral_merkle_proofs.is_some(),);
    Sdk::verify_proof(agg_vk.clone(), vs_baseline.clone(), &vs_proof)?;

    let (vs2_stdin, vs2_def_input) = make_verify_stark_inputs(&sdk, &vs_proof, vs_baseline, &sdk)?;
    prove_and_verify_e2e(&sdk, vs_exe, vs2_stdin, &[vs2_def_input])
}

#[test]
fn test_deferrals_enabled_without_usage() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let sdk = make_verify_stark_sdk(&fib_sdk, app_params, agg_params)?;

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let app_exe = sdk.convert_to_exe(elf)?;

    let n = 1000u64;
    let mut stdin = StdIn::default();
    stdin.write(&n);

    prove_and_verify_e2e(&sdk, app_exe, stdin, &[])
}

#[cfg(feature = "rvr")]
#[test]
fn test_sdk_compiled_pure_save_load_roundtrip() -> Result<()> {
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;

    let mut stdin = StdIn::default();
    stdin.write(&100u64);

    let compiled_a = sdk.compile(exe.clone())?;
    let baseline = sdk.execute(&compiled_a, stdin.clone())?;

    let tmp = tempfile::tempdir()?;
    let lib_path = compiled_a.save(tmp.path())?;
    drop(compiled_a);

    let compiled_b = sdk.load_compiled(&lib_path, exe)?;
    let reloaded = sdk.execute(&compiled_b, stdin)?;

    assert_eq!(baseline, reloaded);
    Ok(())
}

#[cfg(feature = "rvr")]
#[test]
fn test_sdk_compiled_instret_tracking_save_load_roundtrip() -> Result<()> {
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;

    let mut stdin = StdIn::default();
    stdin.write(&100u64);

    let compiled = sdk.compile_with_instret_tracking(exe.clone())?;
    let initial_pc = exe.pc_start;
    let state = compiled.create_initial_vm_state(stdin);
    let state = match compiled.execute_from_state_for(state, 0)? {
        ExecutionOutcome::Suspended(execution) => {
            assert_eq!(execution.retired, 0);
            execution.state
        }
        ExecutionOutcome::Terminated(_) => {
            panic!("zero-budget execution unexpectedly terminated")
        }
    };
    assert_eq!(state.pc(), initial_pc);

    let tmp = tempfile::tempdir()?;
    let lib_path = compiled.save(tmp.path())?;
    drop(compiled);

    assert!(sdk.load_compiled(&lib_path, exe.clone()).is_err());
    let loaded = sdk.load_compiled_with_instret_tracking(&lib_path, exe)?;
    let execution = loaded.execute_from_state(state)?;
    assert!(execution.retired > 0);
    Ok(())
}

#[cfg(feature = "rvr")]
#[test]
fn test_sdk_compiled_metered_save_load_roundtrip() -> Result<()> {
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;

    let mut stdin = StdIn::default();
    stdin.write(&100u64);

    let compiled_a = sdk.compile_metered(exe.clone())?;
    let (baseline_pv, baseline_segments) = sdk.execute_metered(&compiled_a, stdin.clone())?;

    let tmp = tempfile::tempdir()?;
    let lib_path = compiled_a.save(tmp.path())?;
    assert_eq!(
        lib_path.file_stem().and_then(|stem| stem.to_str()),
        Some("libopenvm-metered")
    );
    let mut metadata = load_metered_artifact_metadata(&lib_path)?;
    assert!(!metadata.profile_compatible);
    let metadata_path = metered_artifact_metadata_path(&lib_path);

    let mut metadata_without_compatibility = serde_json::to_value(&metadata)?;
    metadata_without_compatibility
        .as_object_mut()
        .unwrap()
        .remove("profile_compatible");
    fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&metadata_without_compatibility)?,
    )?;
    let missing_compatibility = load_metered_artifact_metadata(&lib_path)
        .expect_err("metadata without profile_compatible must be rejected");
    assert!(format!("{missing_compatibility:?}").contains("missing field `profile_compatible`"));

    metadata.profile_compatible = true;
    fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)?;
    drop(compiled_a);

    let profile_mismatch = sdk
        .load_compiled_metered(&lib_path, exe.clone())
        .err()
        .expect("tampered profile compatibility metadata must be rejected");
    assert!(profile_mismatch
        .to_string()
        .contains("artifact profile compatibility mismatch"));
    metadata.profile_compatible = false;
    fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)?;

    let mismatch = sdk.load_compiled(&lib_path, exe.clone());
    assert!(mismatch.is_err());
    assert!(mismatch
        .err()
        .unwrap()
        .to_string()
        .contains("RVR execution kind mismatch"));

    let compiled_b = sdk.load_compiled_metered(&lib_path, exe)?;
    let (reloaded_pv, reloaded_segments) = sdk.execute_metered(&compiled_b, stdin)?;

    assert_eq!(baseline_pv, reloaded_pv);
    assert_eq!(baseline_segments.len(), reloaded_segments.len());
    for (a, b) in baseline_segments.iter().zip(reloaded_segments.iter()) {
        assert_eq!(a.instret_start, b.instret_start);
        assert_eq!(a.num_insns, b.num_insns);
        assert_eq!(a.trace_heights, b.trace_heights);
    }
    Ok(())
}

#[cfg(feature = "rvr")]
#[test]
fn test_sdk_profiled_artifact_save_load_and_unprofiled_execution() -> Result<()> {
    let (sdk, _, _) = make_fib_sdk();
    let elf_bytes = include_bytes!("../programs/examples/fibonacci.elf");
    let elf = Elf::decode(elf_bytes, MEM_SIZE as u32)?;
    let exe = sdk.convert_to_exe(elf)?;

    let tmp = tempfile::tempdir()?;
    let elf_path = tmp.path().join("fibonacci.elf");
    fs::write(&elf_path, elf_bytes)?;

    let mut stdin = StdIn::default();
    stdin.write(&100u64);

    let compiled = sdk.compile_profiled(ExecutableInput::with_elf_path(exe.clone(), &elf_path))?;
    assert!(compiled.is_profile_compatible());
    let baseline = sdk.execute(&compiled, stdin.clone())?;

    let lib_path = compiled.save(tmp.path())?;
    assert_eq!(
        lib_path.file_stem().and_then(|stem| stem.to_str()),
        Some("libopenvm-pure-profiled")
    );
    drop(compiled);

    let loaded = sdk.load_compiled(&lib_path, exe)?;
    assert!(loaded.is_profile_compatible());
    assert_eq!(baseline, sdk.execute(&loaded, stdin)?);
    Ok(())
}

#[cfg(feature = "rvr")]
#[test]
fn test_sdk_compiled_metered_cost_save_load_roundtrip() -> Result<()> {
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;

    let mut stdin = StdIn::default();
    stdin.write(&100u64);

    let compiled_a = sdk.compile_metered_cost(exe.clone())?;
    let (baseline_pv, baseline_cost) = sdk.execute_metered_cost(&compiled_a, stdin.clone())?;

    let tmp = tempfile::tempdir()?;
    let lib_path = compiled_a.save(tmp.path())?;
    drop(compiled_a);

    let mismatch = sdk.load_compiled(&lib_path, exe.clone());
    assert!(mismatch.is_err());
    assert!(mismatch
        .err()
        .unwrap()
        .to_string()
        .contains("RVR execution kind mismatch"));

    let compiled_b = sdk.load_compiled_metered_cost(&lib_path, exe)?;
    let (reloaded_pv, reloaded_cost) = sdk.execute_metered_cost(&compiled_b, stdin)?;

    assert_eq!(baseline_pv, reloaded_pv);
    assert_eq!(baseline_cost, reloaded_cost);
    Ok(())
}

#[test]
fn test_sdk_compiled_metered_execute() -> Result<()> {
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;

    let mut stdin = StdIn::default();
    stdin.write(&100u64);

    let compiled = sdk.compile_metered(exe)?;
    let (_, segments) = sdk.execute_metered(&compiled, stdin)?;
    assert!(!segments.is_empty());
    Ok(())
}

#[test]
fn test_sdk_compiled_metered_cost_execute() -> Result<()> {
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;

    let mut stdin = StdIn::default();
    stdin.write(&100u64);

    let compiled = sdk.compile_metered_cost(exe)?;
    let (_, (_, instret)) = sdk.execute_metered_cost(&compiled, stdin)?;
    assert!(instret > 0);
    Ok(())
}

#[test]
fn test_deferral_aware_sdk_with_odd_children() -> Result<()> {
    setup_tracing();
    let n_stack = 16;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();
    let hook_commits =
        DeferralHookCommits::from_system_params(&agg_params, hook_params_with_100_bits_security());
    let mut app_config = AppConfig::riscv64(app_params);
    app_config
        .app_vm_config
        .as_mut()
        .set_segmentation_max_memory(256 << 20);
    let aware_sdk = Sdk::builder()
        .app_config(app_config)
        .agg_params(agg_params)
        .agg_tree_config(AggregationTreeConfig {
            num_children_leaf: 1,
            num_children_internal: 3,
        })
        .deferral_hook_commits(hook_commits)
        .build()?;

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let app_exe = aware_sdk.convert_to_exe(elf)?;

    let mut stdin = StdIn::default();
    stdin.write(&(1u64 << 17));

    let compiled = aware_sdk.compile_metered(app_exe.clone())?;
    let (_, segments) = aware_sdk.execute_metered(&compiled, stdin.clone())?;
    assert!(segments.len() >= 3, "expected >= 3 segments");

    prove_and_verify_e2e(&aware_sdk, app_exe, stdin, &[])
}

#[test]
fn test_verify_stark_with_deferral_child() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let vs_sdk = make_verify_stark_sdk(&fib_sdk, app_params, agg_params.clone())?;
    let (vs_stdin, def_input) =
        make_verify_stark_inputs(&fib_sdk, &fib_proof, fib_baseline, &vs_sdk)?;

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
        agg_params.internal.clone(),
        vs_system_config.memory_config.memory_dimensions(),
        vs_system_config.num_public_values,
        Some(expected_def_hook_commit.into()),
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
fn test_prove_mixed_vm_def_depth_mismatch() -> Result<()> {
    setup_tracing();
    let (fib_sdk, app_params, agg_params) = make_fib_sdk();
    let (fib_proof, fib_baseline) = generate_fib_vm_stark_proof(&fib_sdk)?;
    let vs_sdk = make_verify_stark_sdk(&fib_sdk, app_params, agg_params)?;
    let (vs_stdin, def_input) =
        make_verify_stark_inputs(&fib_sdk, &fib_proof, fib_baseline, &vs_sdk)?;

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

    let def_prover = vs_sdk
        .deferral_agg_prover()
        .expect("deferral-enabled SDK should expose a deferral prover");
    let def_hook_proofs = def_prover
        .multi_deferral_circuit_prover
        .prove(&[def_input])?;
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

#[test]
fn test_deferral_aware_and_active_have_equivalent_vks() -> Result<()> {
    setup_tracing();
    let n_stack = 19;
    let app_params = app_params_with_100_bits_security(DEFAULT_APP_L_SKIP + n_stack);
    let agg_params = AggregationSystemParams::default();
    let active_sdk = make_recursive_verify_stark_sdk(app_params.clone(), agg_params.clone())?;
    let hook_commits = DeferralHookCommits {
        hook_cached_commit: active_sdk.def_hook_cached_commit().unwrap(),
        hook_commit: active_sdk.def_hook_commit().unwrap(),
    };
    let aware_sdk = Sdk::builder()
        .app_config(active_sdk.app_config().clone())
        .agg_params(agg_params)
        .deferral_hook_commits(hook_commits)
        .build()?;
    assert_eq!(
        active_sdk.agg_vk().as_ref().pre_hash,
        aware_sdk.agg_vk().as_ref().pre_hash
    );
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
#[cfg(all(feature = "cell-profiling", feature = "root-prover"))]
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
        DeferralSetup, Sdk, StdIn,
    };

    // Root verifier params matching pipeline_cell_count_profiling in static-verifier
    let (app_params, agg_params, root_params) = get_params();
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

            let elf = Elf::decode(
                include_bytes!("../programs/examples/fibonacci.elf"),
                MEM_SIZE as u32,
            )?;
            let sdk = Sdk::riscv64(app_params, agg_params);
            let app_exe = sdk.convert_to_exe(elf)?;

            // Compute trace heights for root prover with profiling params
            let system_config = sdk.app_config().app_vm_config.as_ref();
            let agg_prover = sdk.agg_prover();
            let (trace_heights, root_pk) = compute_root_proof_heights(
                system_config.clone(),
                sdk.agg_config().params.clone(),
                sdk.agg_tree_config().clone(),
                root_params.clone(),
                DeferralSetup::Disabled,
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

            let root_prover = std::sync::Arc::new(RootProver::from_pk(
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
                DeferralSetup::Disabled,
                root_prover.clone(),
                None,
            )?;

            let n = 100u64;
            let mut stdin = StdIn::default();
            stdin.write(&n);

            let root_proof = evm_prover.prove_root(stdin, &[])?;
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
    let ext_chip = BabyBearExtChip::new(BabyBearChip::new(std::sync::Arc::new(range)));
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
