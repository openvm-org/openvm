use std::{slice::from_ref, sync::Arc};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
#[cfg(feature = "rvr")]
use openvm_circuit::arch::ExecutionOutcome;
use openvm_circuit::arch::{instructions::exe::VmExe, U16_CELL_SIZE};
#[cfg(feature = "cuda")]
use openvm_circuit::arch::{verify_segments, VirtualMachineError};
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

#[cfg(feature = "cuda")]
#[test]
fn test_preflight_app_prover_reuse() -> Result<()> {
    setup_tracing();
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;
    let mut prover = sdk.app_prover(exe)?;

    let error = match prover.prove(StdIn::default()) {
        Ok(_) => panic!("missing guest input must fail"),
        Err(error) => error,
    };
    assert!(
        matches!(error, VirtualMachineError::Execution(_)),
        "unexpected preflight proof error: {error}"
    );

    let mut stdin = StdIn::default();
    stdin.write(&1000u64);
    let first = prover.prove(stdin.clone())?;
    let second = prover.prove(stdin)?;

    let (_, app_vk) = sdk.app_keygen();
    verify_segments(&prover.vm().engine, &app_vk.vk, &first.per_segment)?;
    verify_segments(&prover.vm().engine, &app_vk.vk, &second.per_segment)?;
    assert_eq!(
        first.user_public_values.public_values,
        second.user_public_values.public_values
    );
    Ok(())
}

#[cfg(all(feature = "cuda", not(feature = "root-prover")))]
#[test]
fn test_preflight_stark_prover() -> Result<()> {
    setup_tracing();
    let (sdk, _, _) = make_fib_sdk();
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;
    let mut prover = sdk.prover(exe)?;
    let mut stdin = StdIn::default();
    stdin.write(&1000u64);
    let proof = prover.prove(stdin, &[])?.0;
    Sdk::verify_proof((*sdk.agg_vk()).clone(), prover.generate_baseline(), &proof)?;
    Ok(())
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
    drop(compiled_a);

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

/// Segmentation ceiling low enough that a modest input still splits into several
/// segments, so the scheduler graph has a real execute chain without paying for a
/// large proof.
#[cfg(feature = "cuda")]
const SCHEDULED_SEGMENTATION_MAX_MEMORY: usize = 128 << 20;
/// More segments than the first batch's production target can absorb.
///
/// Four would be enough for two proves to be dispatched together, but not for
/// segment production to still have work left by the time that batch runs: the
/// first batch produces up to `proved + prove_lookahead + 2` segments, so with only
/// four the producer is already finished when the first two-prove batch starts.
/// Overlap and concurrency would then be observed in different batches, and the
/// interesting configuration -- production advancing while two proves are resident
/// -- would never occur.
#[cfg(feature = "cuda")]
const SCHEDULED_MIN_SEGMENTS: usize = 6;
/// A 32 GB card as `nvidia-smi` reports it: driver and CUDA context already
/// counted, so this is the whole device rather than a workload-only remainder.
#[cfg(feature = "cuda")]
const SCHEDULED_DEVICE_GPU_BYTES: u64 = 32_768 << 20;

/// The order-insensitive envelope one segment's proof attests to.
///
/// Every field is read from the *returned proof*, never from driver internals or
/// device memory. That is what makes this instrument safe by construction rather
/// than by a flag someone could flip: comparing `ProvingContext` would force D2H
/// copies and a stream fence, so this never looks at one. There is no code path by
/// which this can execute inside a measured arm — it is a `#[test]` that consumes
/// proofs a campaign has already finished producing.
///
/// Commitments are deliberately absent. `common_main_commit` is one aggregate over
/// all traces, including the periphery Poseidon2 trace whose row order is
/// insertion-history dependent, so comparing it across arms either flakes — naming
/// a scheduler defect that did not happen — or passes and thereby implies the
/// ordering problem does not exist.
#[cfg(feature = "cuda")]
#[derive(Debug, PartialEq, Eq)]
struct SegmentEnvelope {
    initial_pc: u32,
    final_pc: u32,
    exit_code: u32,
    is_terminate: bool,
    initial_memory_root: [u32; 8],
    final_memory_root: [u32; 8],
    /// Per-AIR log trace heights: shape and resource demand, order-insensitive.
    log_heights: Vec<Option<usize>>,
}

#[cfg(feature = "cuda")]
fn segment_envelopes(
    proofs: &[openvm_stark_backend::proof::Proof<crate::SC>],
) -> Vec<SegmentEnvelope> {
    use std::borrow::Borrow;

    use openvm_circuit::{
        arch::{CONNECTOR_AIR_ID, MERKLE_AIR_ID},
        system::{connector::VmConnectorPvs, memory::merkle::MemoryMerklePvs},
    };
    use openvm_stark_backend::p3_field::PrimeField32;

    proofs
        .iter()
        .map(|proof| {
            let connector: &VmConnectorPvs<F> =
                proof.public_values[CONNECTOR_AIR_ID].as_slice().borrow();
            let merkle: &MemoryMerklePvs<F, 8> =
                proof.public_values[MERKLE_AIR_ID].as_slice().borrow();
            let digest = |d: &[F; 8]| {
                let mut out = [0u32; 8];
                for (slot, value) in out.iter_mut().zip(d.iter()) {
                    *slot = value.as_canonical_u32();
                }
                out
            };
            SegmentEnvelope {
                initial_pc: connector.initial_pc.as_canonical_u32(),
                final_pc: connector.final_pc.as_canonical_u32(),
                exit_code: connector.exit_code.as_canonical_u32(),
                is_terminate: connector.is_terminate.as_canonical_u32() != 0,
                initial_memory_root: digest(&merkle.initial_root),
                final_memory_root: digest(&merkle.final_root),
                log_heights: proof
                    .trace_vdata
                    .iter()
                    .map(|air| air.as_ref().map(|vdata| vdata.log_height))
                    .collect(),
            }
        })
        .collect()
}

/// Builds the scheduled-GPU fixture: enough segments that two proves are admitted
/// together and segment production still has work left while they run.
#[cfg(feature = "cuda")]
fn scheduled_gpu_fixture() -> Result<(
    Arc<crate::keygen::AppProvingKey<SdkVmConfig>>,
    Arc<VmExe<F>>,
    StdIn,
)> {
    let (app_params, agg_params, _) = get_params();
    let mut app_config = AppConfig::riscv64(app_params);
    app_config
        .app_vm_config
        .as_mut()
        .set_segmentation_max_memory(SCHEDULED_SEGMENTATION_MAX_MEMORY);
    let sdk = Sdk::builder()
        .app_config(app_config)
        .agg_params(agg_params)
        .build()?;
    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;
    let mut stdin = StdIn::default();
    stdin.write(&(1u64 << 17));
    Ok((sdk.app_pk().clone().into(), exe, stdin))
}

/// I3-A — driver boundary fidelity on GPU. Correctness-only; never measured.
///
/// The scheduled driver must hand each segment the same slice of execution the
/// serial driver does. Boundaries pin which slice; the connector and merkle public
/// values pin where execution started and ended and what memory it started and
/// ended on; trace heights pin the shape. All order-insensitive, all read from
/// returned proofs.
///
/// Claim scope: boundary fidelity only. This does NOT prove byte-output identity —
/// GPU grinding keeps the first `atomicCAS` winner, so identical inputs cannot
/// imply identical bytes — and it does NOT cover post-checkpoint interference,
/// which is I3-C.
#[cfg(feature = "cuda")]
#[test]
fn scheduled_gpu_presents_the_serial_input_envelope() -> Result<()> {
    use openvm_circuit::arch::SegmentSchedulerConfig;
    use openvm_sdk_config::SdkVmGpuBuilder;

    use crate::prover::{vm::new_local_prover, AppProver};

    setup_tracing();
    let (app_pk, exe, stdin) = scheduled_gpu_fixture()?;
    let app_vm_pk = &app_pk.app_vm_pk;
    let app_vk = app_vm_pk.vm_pk.get_vk();

    // Separate instances per arm. Proving twice on one instance does not reproduce
    // that instance's own first proof, so sharing one would compare carry-over
    // state rather than the two drivers.
    let serial_instance =
        new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe.clone())?;
    assert!(
        serial_instance.segment_scheduler().is_none(),
        "the serial driver must be the default"
    );
    let mut serial_prover = AppProver::new_from_instance(serial_instance, app_vk.clone());
    let serial = serial_prover.prove(stdin.clone())?;
    let serial_boundaries = serial_prover.instance().segment_boundaries().to_vec();
    let serial_envelopes = segment_envelopes(&serial.per_segment);
    drop(serial_prover);

    let mut scheduled_instance =
        new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe)?;
    scheduled_instance.set_segment_scheduler(Some(SegmentSchedulerConfig::for_device(
        SCHEDULED_DEVICE_GPU_BYTES,
    )));
    let mut scheduled_prover = AppProver::new_from_instance(scheduled_instance, app_vk);
    let scheduled = scheduled_prover.prove(stdin)?;
    let scheduled_boundaries = scheduled_prover.instance().segment_boundaries().to_vec();
    let scheduled_envelopes = segment_envelopes(&scheduled.per_segment);
    let max_concurrent = scheduled_prover.instance().max_concurrent_proves();

    tracing::info!(
        segments = serial.per_segment.len(),
        max_concurrent_proves = max_concurrent,
        ?serial_boundaries,
        ?scheduled_boundaries,
        "I3-A envelope comparison"
    );

    // Without two resident proves this compares the scheduled driver to itself in
    // serial clothing, and would pass however broken concurrency was.
    assert!(
        serial.per_segment.len() >= SCHEDULED_MIN_SEGMENTS,
        "need several segments for this to mean anything, got {}",
        serial.per_segment.len()
    );
    assert!(
        max_concurrent >= 2,
        "two proves must have run together or this comparison is vacuous; saw {max_concurrent}"
    );

    assert_eq!(
        serial_boundaries, scheduled_boundaries,
        "segment boundaries must not depend on the driver"
    );
    assert_eq!(
        serial_envelopes.len(),
        scheduled_envelopes.len(),
        "segment count must not depend on the driver"
    );
    for (idx, (want, got)) in serial_envelopes
        .iter()
        .zip(scheduled_envelopes.iter())
        .enumerate()
    {
        assert_eq!(want, got, "segment {idx} envelope differs between drivers");
    }

    // Negative control: the comparison must be capable of failing. A perturbed
    // input has to move the envelope, or `assert_eq` above proves nothing.
    let mut other_stdin = StdIn::default();
    other_stdin.write(&(1u64 << 16));
    let other_instance = new_local_prover::<E, SdkVmGpuBuilder>(
        SdkVmGpuBuilder,
        app_vm_pk,
        scheduled_prover.instance().exe().clone(),
    )?;
    let mut other_prover = AppProver::new_from_instance(other_instance, app_vm_pk.vm_pk.get_vk());
    let other = other_prover.prove(other_stdin)?;
    let other_envelopes = segment_envelopes(&other.per_segment);
    assert_ne!(
        serial_envelopes, other_envelopes,
        "NEGATIVE CONTROL FAILED: a different input produced an identical envelope, so \
         the envelope does not actually discriminate and every assertion above is vacuous"
    );
    // Vec inequality can be satisfied by length alone, and a smaller input does
    // produce fewer segments. Compare a segment that exists in both runs, so the
    // control shows the envelope discriminates on what a segment attests to and not
    // merely on how many there are.
    assert_ne!(
        serial_envelopes[0], other_envelopes[0],
        "NEGATIVE CONTROL FAILED: segment 0's envelope is identical under a different \
         input, so the per-segment comparison discriminates only on segment count"
    );
    tracing::info!(
        serial_segments = serial_envelopes.len(),
        other_segments = other_envelopes.len(),
        "I3-A negative control: a different input moved the envelope, segment 0 included"
    );
    Ok(())
}

/// I3-B — semantic output equivalence on GPU.
///
/// Both arms verify, and each arm's user-public-values Merkle proof is checked
/// against *that arm's own* verified final memory root — not byte-compared, and
/// not checked against the other arm's root, which a scheduled run could otherwise
/// pass by borrowing.
///
/// Known limit, stated rather than papered over: `VmConnectorPvs` exposes only
/// initial/final PC, exit code and termination. `num_insns` and timestamps are
/// absent, so a different valid private execution with identical public endpoints
/// would pass this check. I3-A's boundary envelope is what closes that gap; I3-B
/// alone does not.
#[cfg(feature = "cuda")]
#[test]
fn scheduled_gpu_means_the_same_as_serial() -> Result<()> {
    use openvm_circuit::arch::{hasher::poseidon2::vm_poseidon2_hasher, SegmentSchedulerConfig};
    use openvm_sdk_config::SdkVmGpuBuilder;

    use crate::prover::{vm::new_local_prover, AppProver};

    setup_tracing();
    let (app_pk, exe, stdin) = scheduled_gpu_fixture()?;
    let app_vm_pk = &app_pk.app_vm_pk;
    let app_vk = app_vm_pk.vm_pk.get_vk();
    let memory_dimensions = app_vm_pk
        .vm_config
        .as_ref()
        .memory_config
        .memory_dimensions();
    let hasher = vm_poseidon2_hasher();

    let serial_instance =
        new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe.clone())?;
    let mut serial_prover = AppProver::new_from_instance(serial_instance, app_vk.clone());
    let serial = serial_prover.prove(stdin.clone())?;
    let serial_payload = verify_segments(
        &serial_prover.instance().vm.engine,
        &app_vk,
        &serial.per_segment,
    )
    .map_err(|error| eyre::eyre!("serial segment proofs must verify: {error}"))?;
    drop(serial_prover);

    let mut scheduled_instance =
        new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe)?;
    scheduled_instance.set_segment_scheduler(Some(SegmentSchedulerConfig::for_device(
        SCHEDULED_DEVICE_GPU_BYTES,
    )));
    let mut scheduled_prover = AppProver::new_from_instance(scheduled_instance, app_vk.clone());
    let scheduled = scheduled_prover.prove(stdin)?;
    let scheduled_payload = verify_segments(
        &scheduled_prover.instance().vm.engine,
        &app_vk,
        &scheduled.per_segment,
    )
    .map_err(|error| eyre::eyre!("scheduled segment proofs must verify: {error}"))?;
    let max_concurrent = scheduled_prover.instance().max_concurrent_proves();
    assert!(
        max_concurrent >= 2,
        "two proves must have run together or this compares serial to serial; saw {max_concurrent}"
    );

    assert_eq!(
        serial_payload.exe_commit, scheduled_payload.exe_commit,
        "executable commitment must not depend on the driver"
    );
    assert_eq!(
        serial_payload.final_memory_root, scheduled_payload.final_memory_root,
        "final memory root must not depend on the driver"
    );
    assert_eq!(
        serial.user_public_values.public_values, scheduled.user_public_values.public_values,
        "user public values must not depend on the driver"
    );

    // Each arm against its own root.
    serial
        .user_public_values
        .verify(&hasher, memory_dimensions, serial_payload.final_memory_root)
        .map_err(|error| eyre::eyre!("serial public values must verify: {error}"))?;
    scheduled
        .user_public_values
        .verify(
            &hasher,
            memory_dimensions,
            scheduled_payload.final_memory_root,
        )
        .map_err(|error| eyre::eyre!("scheduled public values must verify: {error}"))?;

    // Negative control: the public-values check must be capable of rejecting. A
    // corrupted root has to fail, or "it verified" carries no information.
    let mut wrong_root = scheduled_payload.final_memory_root;
    wrong_root[0] += <F as openvm_stark_backend::p3_field::PrimeCharacteristicRing>::ONE;
    assert!(
        scheduled
            .user_public_values
            .verify(&hasher, memory_dimensions, wrong_root)
            .is_err(),
        "NEGATIVE CONTROL FAILED: the public-values proof verified against a corrupted \
         memory root, so verifying against the correct one proves nothing"
    );
    tracing::info!("I3-B negative control: a corrupted memory root was rejected");
    Ok(())
}

/// I3-C — post-checkpoint interference on GPU.
///
/// I3-A and I3-B compare *drivers*. This compares *concurrency inside the
/// scheduled driver*, which is the only thing that observes the channels neither
/// of those touches: scheduled proves run on separate host threads that share a
/// cloned `GpuDevice` and the process-global `MEMORY_MANAGER`. Both arms here are
/// the scheduled driver, in one process, differing only in whether the budget
/// admits one prove or two — so a difference implicates concurrency and nothing
/// else.
///
/// Byte-identity is deliberately not asserted: GPU grinding keeps the first
/// `atomicCAS` winner, so identical inputs cannot imply identical output bytes on
/// this backend. What is asserted is that the concurrent run's segment envelope
/// and verified semantics are those of the width-1 run.
#[cfg(feature = "cuda")]
#[test]
fn concurrency_does_not_perturb_the_scheduled_gpu_replay() -> Result<()> {
    use openvm_circuit::arch::{Budget, SegmentSchedulerConfig, PROVE_MARGINAL_GPU_BYTES};
    use openvm_sdk_config::SdkVmGpuBuilder;

    use crate::prover::{vm::new_local_prover, AppProver};

    setup_tracing();
    let (app_pk, exe, stdin) = scheduled_gpu_fixture()?;
    let app_vm_pk = &app_pk.app_vm_pk;
    let app_vk = app_vm_pk.vm_pk.get_vk();

    // Width pinned by construction, so the two arms differ in concurrency alone and
    // not in however much memory the card happens to have.
    let width =
        |n: u64| SegmentSchedulerConfig::new(Budget::new(PROVE_MARGINAL_GPU_BYTES * n, 0, 0));

    let mut one_at_a_time =
        new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe.clone())?;
    one_at_a_time.set_segment_scheduler(Some(width(1)));
    let mut serial_prover = AppProver::new_from_instance(one_at_a_time, app_vk.clone());
    let sequential = serial_prover.prove(stdin.clone())?;
    let sequential_envelopes = segment_envelopes(&sequential.per_segment);
    let sequential_concurrent = serial_prover.instance().max_concurrent_proves();
    let sequential_boundaries = serial_prover.instance().segment_boundaries().to_vec();
    drop(serial_prover);

    let mut two_at_a_time =
        new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe)?;
    two_at_a_time.set_segment_scheduler(Some(width(2)));
    let mut concurrent_prover = AppProver::new_from_instance(two_at_a_time, app_vk.clone());
    let concurrent = concurrent_prover.prove(stdin)?;
    let concurrent_envelopes = segment_envelopes(&concurrent.per_segment);
    let concurrent_concurrent = concurrent_prover.instance().max_concurrent_proves();
    let concurrent_boundaries = concurrent_prover.instance().segment_boundaries().to_vec();

    tracing::info!(
        width_one_max_concurrent = sequential_concurrent,
        width_two_max_concurrent = concurrent_concurrent,
        "I3-C replay: same driver, concurrency is the only difference"
    );

    // The two arms must actually differ in concurrency, or this is one arm twice.
    assert_eq!(
        sequential_concurrent, 1,
        "the width-1 arm must never have held two proves at once"
    );
    assert!(
        concurrent_concurrent >= 2,
        "NEGATIVE CONTROL FAILED: the width-2 arm never held two proves at once, so this \
         compares a sequential run to a sequential run and observes no interference \
         channel at all; saw {concurrent_concurrent}"
    );

    verify_segments(
        &concurrent_prover.instance().vm.engine,
        &app_vk,
        &concurrent.per_segment,
    )
    .map_err(|error| eyre::eyre!("concurrent segment proofs must verify: {error}"))?;

    assert_eq!(
        sequential_boundaries, concurrent_boundaries,
        "segment boundaries must not depend on how many proves are resident"
    );
    assert_eq!(
        sequential_envelopes, concurrent_envelopes,
        "the concurrent replay diverged from the one-at-a-time replay, which means \
         proves interfered through the shared device or the global memory manager"
    );
    Ok(())
}

/// The `PerProve` proving-key residency arm, actually run.
///
/// It is compiled but every end-to-end run so far used `Shared`, so its N+1
/// residency was reasoned and not measured. M4-2 has to report the device-memory
/// marginal each way, and whether two residents fit under a 32 GiB budget depends
/// on which arm is used, so this measures it rather than deriving it.
#[cfg(feature = "cuda")]
#[test]
fn per_prove_pk_residency_costs_a_key_per_prove() -> Result<()> {
    use openvm_circuit::arch::{
        Budget, ProvingKeyResidency, SegmentSchedulerConfig, PROVE_MARGINAL_GPU_BYTES,
    };
    use openvm_cuda_common::memory_manager::device_memory_used;
    use openvm_sdk_config::SdkVmGpuBuilder;

    use crate::prover::{vm::new_local_prover, AppProver};

    setup_tracing();
    let (app_pk, exe, stdin) = scheduled_gpu_fixture()?;
    let app_vm_pk = &app_pk.app_vm_pk;
    let app_vk = app_vm_pk.vm_pk.get_vk();
    let config = SegmentSchedulerConfig::new(Budget::new(PROVE_MARGINAL_GPU_BYTES * 2, 0, 0));

    let run = |residency: ProvingKeyResidency<crate::SC>| -> Result<(usize, usize, usize)> {
        let mut instance =
            new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe.clone())?;
        instance.set_segment_scheduler(Some(config));
        instance.set_prove_pk_residency(residency);
        let before = device_memory_used();
        let mut prover = AppProver::new_from_instance(instance, app_vk.clone());
        let proof = prover.prove(stdin.clone())?;
        let after = device_memory_used();
        let concurrent = prover.instance().max_concurrent_proves();
        assert!(
            concurrent >= 2,
            "the pool must have held two proves for a per-prove key to cost anything; saw \
             {concurrent}"
        );
        verify_segments(&prover.instance().vm.engine, &app_vk, &proof.per_segment)
            .map_err(|error| eyre::eyre!("segment proofs must verify: {error}"))?;
        Ok((before, after, concurrent))
    };

    let (shared_before, shared_after, shared_concurrent) = run(ProvingKeyResidency::Shared)?;
    let (per_before, per_after, per_concurrent) =
        run(ProvingKeyResidency::PerProve(app_vm_pk.vm_pk.clone()))?;

    let mib = |bytes: usize| bytes as f64 / (1024.0 * 1024.0);
    let shared_resident = shared_after.saturating_sub(shared_before);
    let per_resident = per_after.saturating_sub(per_before);
    tracing::info!(
        shared_before_mib = mib(shared_before),
        shared_after_mib = mib(shared_after),
        shared_resident_mib = mib(shared_resident),
        shared_max_concurrent = shared_concurrent,
        per_prove_before_mib = mib(per_before),
        per_prove_after_mib = mib(per_after),
        per_prove_resident_mib = mib(per_resident),
        per_prove_max_concurrent = per_concurrent,
        "PerProve vs Shared device-memory marginal (nvidia-smi semantics: cudaMemGetInfo)"
    );

    // Both arms must produce a verifying proof — measured above — and the PerProve
    // arm must actually cost more, or it is not doing what its name says.
    assert!(
        per_resident > shared_resident,
        "NEGATIVE CONTROL FAILED: PerProve residency cost no more device memory than \
         Shared ({} MiB vs {} MiB), so either the per-slot keys were never transported \
         or this measurement cannot see them",
        mib(per_resident),
        mib(shared_resident)
    );
    Ok(())
}

/// Every scheduled prove must hand release ownership of its traces to its own
/// stream — and the serial driver must not.
///
/// The allocator's own tests cover what the handoff *does*: adopt the consumer's
/// stream and a sibling cannot overwrite the trace; skip it and the sibling does.
/// They cannot see whether production still calls it — they invoke the operation
/// directly, so they stay green for any state of the call site. This test watches
/// the production path instead of the primitive.
///
/// It counts handoffs across a real scheduled prove rather than reproducing the
/// race, so it is deterministic — a counter comparison, not a timing lottery. The
/// two halves are complementary and neither is sufficient alone: this one shows the
/// wiring is present, the allocator tests show the wiring does something.
///
/// The serial half matters as much as the scheduled half. It pins the handoff to
/// the scheduled path, so moving it somewhere shared — where the "previous stream
/// is already complete" precondition is not established by the tracegen fence —
/// fails here rather than silently becoming unsound.
#[cfg(feature = "cuda")]
#[test]
fn scheduled_gpu_hands_every_prove_the_release_of_its_own_traces() -> Result<()> {
    use openvm_circuit::arch::SegmentSchedulerConfig;
    use openvm_cuda_common::memory_manager::release_stream_handoffs;
    use openvm_sdk_config::SdkVmGpuBuilder;

    use crate::prover::{vm::new_local_prover, AppProver};

    setup_tracing();
    let (app_pk, exe, stdin) = scheduled_gpu_fixture()?;
    let app_vm_pk = &app_pk.app_vm_pk;
    let app_vk = app_vm_pk.vm_pk.get_vk();

    // --- serial: the handoff must not happen at all ---------------------
    let serial_instance =
        new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe.clone())?;
    let mut serial_prover = AppProver::new_from_instance(serial_instance, app_vk.clone());
    let serial_before = release_stream_handoffs();
    let serial = serial_prover.prove(stdin.clone())?;
    let serial_handoffs = release_stream_handoffs() - serial_before;
    drop(serial_prover);
    assert_eq!(
        serial_handoffs, 0,
        "the serial driver proves on the stream that generated the traces, so it must \
         perform no release-stream handoff; saw {serial_handoffs}. A handoff here means \
         the operation moved to a shared path, where the tracegen fence does not \
         establish its precondition"
    );

    // --- scheduled: every prove must hand off every common_main ---------
    let mut scheduled_instance =
        new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe)?;
    scheduled_instance.set_segment_scheduler(Some(SegmentSchedulerConfig::for_device(
        SCHEDULED_DEVICE_GPU_BYTES,
    )));
    let mut scheduled_prover = AppProver::new_from_instance(scheduled_instance, app_vk.clone());
    let scheduled_before = release_stream_handoffs();
    let scheduled = scheduled_prover.prove(stdin)?;
    let scheduled_handoffs = release_stream_handoffs() - scheduled_before;

    let proves = scheduled.per_segment.len() as u64;
    let max_concurrent = scheduled_prover.instance().max_concurrent_proves();
    assert!(
        max_concurrent >= 2,
        "two proves must have run together, or this exercises a driver with no sibling \
         stream and the handoff is not the thing under test; saw {max_concurrent}"
    );
    assert_eq!(
        serial.per_segment.len(),
        scheduled.per_segment.len(),
        "both drivers must span the same segments for the counts to be comparable"
    );

    // Each scheduled prove hands off every `common_main` in its context, and every
    // segment has at least one AIR with a trace. A lower bound rather than an
    // equality because the counter is process-global: a concurrently running GPU
    // test can only inflate it, never deflate it, so this direction stays sound
    // under `--test-threads > 1`.
    assert!(
        scheduled_handoffs >= proves,
        "the scheduled driver performed {scheduled_handoffs} release-stream handoffs \
         across {proves} proves, so at least one prove read traces whose release was \
         still ordered on the trace-generation stream. A sibling slot may then reuse \
         that allocation mid-prove, which produces a well-formed but unverifiable \
         proof. Was the handoff in `ProveSlot::prove` removed or bypassed?"
    );

    tracing::info!(
        proves,
        scheduled_handoffs,
        serial_handoffs,
        max_concurrent,
        "release-stream handoff wiring"
    );

    verify_segments(
        &scheduled_prover.vm().engine,
        &app_vk,
        &scheduled.per_segment,
    )?;
    Ok(())
}

/// Two proves admitted together must run on different CUDA streams.
///
/// Seating two proves is not the same as running them concurrently. Every prove
/// built from one engine receives a clone of that engine's device, sharing a
/// single `Arc<CudaStream>`, and their kernel work then runs in issue order rather
/// than together. That shape stays functionally correct — disjoint buffers, total
/// stream order — so the proof it produces is indistinguishable from a concurrent
/// one and every proof-level check passes straight through it, while delivering
/// the throughput of a single prover. The stream each prove was enqueued on is
/// therefore the only thing that separates real concurrency from seating, which is
/// why this asserts on stream identity and not on overlap in time.
#[cfg(feature = "cuda")]
#[test]
fn scheduled_gpu_proves_are_admitted_onto_distinct_streams() -> Result<()> {
    use std::collections::HashSet;

    use openvm_circuit::arch::SegmentSchedulerConfig;
    use openvm_sdk_config::SdkVmGpuBuilder;

    use crate::prover::{vm::new_local_prover, AppProver};

    setup_tracing();
    let (app_params, agg_params, _) = get_params();
    let mut app_config = AppConfig::riscv64(app_params);
    app_config
        .app_vm_config
        .as_mut()
        .set_segmentation_max_memory(SCHEDULED_SEGMENTATION_MAX_MEMORY);
    let sdk = Sdk::builder()
        .app_config(app_config)
        .agg_params(agg_params)
        .build()?;

    let elf = Elf::decode(
        include_bytes!("../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = sdk.convert_to_exe(elf)?;
    let app_vm_pk = &sdk.app_pk().app_vm_pk;
    let app_vk = app_vm_pk.vm_pk.get_vk();

    let mut instance = new_local_prover::<E, SdkVmGpuBuilder>(SdkVmGpuBuilder, app_vm_pk, exe)?;
    instance.set_segment_scheduler(Some(SegmentSchedulerConfig::for_device(
        SCHEDULED_DEVICE_GPU_BYTES,
    )));
    let mut prover = AppProver::new_from_instance(instance, app_vk.clone());

    let mut stdin = StdIn::default();
    stdin.write(&(1u64 << 17));
    let proof = prover.prove(stdin)?;

    let record = prover.instance().scheduled_run();
    // Emitted so the artifact carries the stream handles themselves, not just the
    // fact that an assertion about them held.
    tracing::info!(
        segments = proof.per_segment.len(),
        max_concurrent_proves = record.max_concurrent_proves,
        prove_batches = ?record.prove_batches,
        "scheduled GPU run"
    );
    assert!(
        proof.per_segment.len() >= SCHEDULED_MIN_SEGMENTS,
        "the workload must span several segments for this to mean anything, got {}",
        proof.per_segment.len()
    );
    assert!(
        record.max_concurrent_proves >= 2,
        "two proves must have been dispatched together, high-water mark was {}",
        record.max_concurrent_proves
    );

    // Half one: proves dispatched together could run concurrently at all, because
    // they hold different device queues. This says nothing about whether anything
    // else advanced meanwhile.
    let concurrent: Vec<_> = record
        .prove_batches
        .iter()
        .filter(|batch| batch.queues.len() >= 2)
        .collect();
    assert!(
        !concurrent.is_empty(),
        "no batch held two proves, so a distinct-stream assertion would be vacuous"
    );
    for batch in &concurrent {
        let streams: HashSet<u64> = batch.queues.iter().map(|(_, stream)| *stream).collect();
        assert_eq!(
            streams.len(),
            batch.queues.len(),
            "proves dispatched together shared a CUDA stream and so serialized: {batch:?}"
        );
    }

    // Half two: segment production actually advanced while proves were still
    // running. Distinct queues permit concurrency; this is what shows it happened.
    // `still_running_after_production` is sampled before the join, so a non-zero
    // value means those proves had not completed when production stopped -- which
    // rules out production merely following a finished prove.
    let overlapped: Vec<_> = record
        .prove_batches
        .iter()
        .filter(|batch| batch.produced_while_proving > 0)
        .collect();
    assert!(
        !overlapped.is_empty(),
        "the producer never advanced inside a prove window, so the rvr path streams \
         without overlapping"
    );
    assert!(
        overlapped
            .iter()
            .any(|batch| batch.still_running_after_production > 0),
        "production always finished after every prove had completed, so nothing \
         overlapped: {overlapped:?}"
    );
    // The configuration M4-2 will measure: production advancing while two proves
    // are resident, not merely while one is.
    assert!(
        record
            .prove_batches
            .iter()
            .any(|batch| batch.queues.len() >= 2
                && batch.produced_while_proving > 0
                && batch.still_running_after_production >= 2),
        "no batch had production advancing while two proves were still running: {:?}",
        record.prove_batches
    );

    verify_segments(&prover.vm().engine, &app_vk, &proof.per_segment)?;
    Ok(())
}
