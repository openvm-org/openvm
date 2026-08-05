use std::{path::PathBuf, slice::from_ref, sync::Arc};

use eyre::Result;
use openvm_build::GuestOptions;
use openvm_recursion_circuit::batch_constraint::commit_child_vk;
use openvm_sdk::{
    config::{AggregationConfig, AppConfig},
    keygen::SdkCachedProvingKey,
    openvm_circuit::arch::instructions::exe::VmExe,
    prover::{DeferralAggProver, MultiDeferralCircuitProver},
    types::VersionedVmStarkProof,
    DefaultStarkEngine as E, DeferralInput, Sdk, StdIn, F, SC,
};
use openvm_sdk_config::{deferral::SupportedDeferral, SdkSystemConfig, SdkVmConfig};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, StarkEngine};
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, baby_bear_poseidon2::Digest,
    hook_params_with_100_bits_security, MAX_APP_LOG_STACKED_HEIGHT,
};
use openvm_verify_stark_circuit::{
    default_verify_stark_circuit_params,
    extension::{get_deferral_state, get_raw_deferral_results},
};
use openvm_verify_stark_host::{
    vk::{VerificationBaseline, VmStarkVerifyingKey},
    VmStarkProof,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_verify_stark_circuit::prover::{
            DeferredVerifyGpuCircuitProver as VerifyCircuitProver,
            DeferredVerifyGpuProver as VerifyProver,
        };
    } else {
        use openvm_verify_stark_circuit::prover::{
            DeferredVerifyCpuCircuitProver as VerifyCircuitProver,
            DeferredVerifyCpuProver as VerifyProver,
        };
    }
}

pub const VERIFY_STARK_DEF_IDX: usize = 0;
const MAX_VERIFY_STARK_GUEST_DEF_CIRCUITS: usize = 8;

pub fn verify_stark_example_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("host crate should live under examples/verify-stark")
        .to_path_buf()
}

pub fn verify_stark_guest_dir() -> PathBuf {
    verify_stark_example_dir().join("guest")
}

pub fn keygen(
    child_agg_vk: Arc<MultiStarkVerifyingKey<SC>>,
    num_def_circuits: u32,
) -> Result<(
    SdkCachedProvingKey<SdkVmConfig>,
    SdkVmConfig,
    MultiStarkVerifyingKey<SC>,
)> {
    // App memory dimensions and number of public values the verify-stark prover expects.
    let default_config = SdkSystemConfig::default();
    let memory_dimensions = default_config.config.memory_config.memory_dimensions();
    let num_user_pvs = default_config.config.num_public_values;

    // System parameters for VM proof aggregation. Deferral circuit proof aggregation reuses
    // agg_config.
    let app_params = app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT);
    let agg_config = AggregationConfig::default();

    // Create the DeferralAggProver.
    let deferral_agg_prover = {
        // Default verify-stark circuit system parameters.
        let verify_prover_params = default_verify_stark_circuit_params();

        // Generate the internal-recursive cached commit that Proofs are expected to expose.
        let engine = <E as StarkEngine>::new(child_agg_vk.inner.params.clone());
        let child_internal_recursive_cached_commit = commit_child_vk(&engine, &child_agg_vk, true)
            .commitment
            .into();

        // Create the verify-stark deferral circuit prover and use it in the DeferralAggProver.
        assert!(num_def_circuits > 0);
        assert!(num_def_circuits as usize <= MAX_VERIFY_STARK_GUEST_DEF_CIRCUITS);

        let verify_prover = VerifyProver::new::<E>(
            child_agg_vk.clone(),
            child_internal_recursive_cached_commit,
            verify_prover_params.clone(),
            memory_dimensions,
            num_user_pvs,
            None,
            VERIFY_STARK_DEF_IDX,
        );
        let verify_circuit_prover = VerifyCircuitProver::new(verify_prover);
        let mut multi_deferral_circuit_prover = MultiDeferralCircuitProver::new(
            verify_circuit_prover,
            agg_config.clone(),
            hook_params_with_100_bits_security(),
        );

        for def_offset in 1..num_def_circuits as usize {
            let verify_prover = VerifyProver::new::<E>(
                child_agg_vk.clone(),
                child_internal_recursive_cached_commit,
                verify_prover_params.clone(),
                memory_dimensions,
                num_user_pvs,
                None,
                VERIFY_STARK_DEF_IDX + def_offset,
            );
            let verify_circuit_prover = VerifyCircuitProver::new(verify_prover);
            multi_deferral_circuit_prover =
                multi_deferral_circuit_prover.with_prover(verify_circuit_prover);
        }
        DeferralAggProver::new(agg_config.clone(), Arc::new(multi_deferral_circuit_prover))
    };

    // Create the SdkVmConfig from the DeferralAggProver
    let supported_deferrals = (0..num_def_circuits)
        .map(|_| SupportedDeferral::VerifyStark)
        .collect();
    let deferral_config = deferral_agg_prover
        .multi_deferral_circuit_prover
        .make_config(supported_deferrals);
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv64i(Default::default())
        .rv64m(Default::default())
        .io(Default::default())
        .deferral(deferral_config)
        .build()
        .optimize();

    // Build the SDK and return keygen artifacts.
    let sdk = Sdk::builder()
        .app_config(AppConfig::new(vm_config.clone(), app_params))
        .agg_params(agg_config.params)
        .deferral_agg_prover(deferral_agg_prover)
        .build()?;
    let _ = sdk.app_keygen();
    let _ = sdk.agg_pk();
    let agg_vk = sdk.agg_vk().as_ref().clone();
    Ok((sdk.cached_proving_key()?, vm_config, agg_vk))
}

pub fn build(
    cached_pk: SdkCachedProvingKey<SdkVmConfig>,
) -> Result<(Arc<VmExe<F>>, VerificationBaseline)> {
    let num_configured_def_circuits = get_num_configured_def_circuits(&cached_pk);
    let guest_features = if num_configured_def_circuits <= 1 {
        Vec::new()
    } else {
        vec![format!("deferral-{}", num_configured_def_circuits - 1)]
    };

    let sdk = Sdk::from_deferral_cached_proving_key(cached_pk)?;
    let elf = sdk.build(
        GuestOptions::default().with_features(guest_features),
        verify_stark_guest_dir(),
        &None,
        None,
    )?;
    let exe = sdk.convert_to_exe(elf)?;
    let prover = sdk.prover(exe.clone())?;
    let baseline = prover.generate_baseline();
    Ok((exe, baseline))
}

pub fn prove(
    cached_pk: SdkCachedProvingKey<SdkVmConfig>,
    exe: Arc<VmExe<F>>,
    child_agg_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_baseline: VerificationBaseline,
    input_proof: VersionedVmStarkProof,
    num_proves_per_circuit: Vec<u32>,
) -> Result<VersionedVmStarkProof> {
    let num_configured_def_circuits = get_num_configured_def_circuits(&cached_pk);
    assert_eq!(num_proves_per_circuit.len(), num_configured_def_circuits);

    let sdk = Sdk::from_deferral_cached_proving_key(cached_pk)?;

    let mut verify_stark_cached_commits =
        sdk.deferral_circuit_cached_commits(VERIFY_STARK_DEF_IDX)?;
    assert_eq!(verify_stark_cached_commits.len(), 1);
    let verify_stark_cached_commit = verify_stark_cached_commits.pop().unwrap();

    let (stdin, def_inputs) = verify_stark_guest_inputs(
        &input_proof.try_into()?,
        child_agg_vk.as_ref().clone(),
        child_baseline,
        verify_stark_cached_commit.into(),
        &num_proves_per_circuit,
    )?;
    let (proof, _) = sdk.prove(exe, stdin, &def_inputs)?;
    VersionedVmStarkProof::new(proof)
}

fn get_num_configured_def_circuits(cached_pk: &SdkCachedProvingKey<SdkVmConfig>) -> usize {
    let num_configured_def_circuits = cached_pk
        .app_pk
        .app_vm_pk
        .vm_config
        .deferral
        .as_ref()
        .map_or(0, |deferral| deferral.circuits.len());
    assert!(num_configured_def_circuits > 0);
    assert!(num_configured_def_circuits <= MAX_VERIFY_STARK_GUEST_DEF_CIRCUITS);
    num_configured_def_circuits
}

fn verify_stark_guest_inputs(
    proof: &VmStarkProof,
    agg_vk: MultiStarkVerifyingKey<SC>,
    baseline: VerificationBaseline,
    verify_stark_cached_commit: Digest,
    num_proves_per_circuit: &[u32],
) -> Result<(StdIn, Vec<DeferralInput>)> {
    let child_vk = VmStarkVerifyingKey {
        mvk: agg_vk,
        baseline,
    };

    let raw_res = get_raw_deferral_results(&child_vk, from_ref(proof), verify_stark_cached_commit)?;
    assert_eq!(raw_res.len(), 1);
    let input_commit: [u8; 32] = raw_res[0].input.clone().try_into().unwrap();

    let mut stdin = StdIn::default();
    stdin.write(&input_commit);
    stdin.write(&(num_proves_per_circuit.len() as u32));

    stdin.deferrals = vec![Default::default(); num_proves_per_circuit.len()];
    let mut def_inputs = vec![DeferralInput::default(); num_proves_per_circuit.len()];

    for (def_idx, &num_proves) in num_proves_per_circuit.iter().enumerate() {
        stdin.write(&num_proves);

        if num_proves == 0 {
            continue;
        }

        let proofs = vec![proof.clone(); num_proves as usize];
        stdin.deferrals[def_idx] = get_deferral_state(
            &child_vk,
            &proofs,
            verify_stark_cached_commit,
            def_idx as u32,
        )?;
        def_inputs[def_idx] = DeferralInput::from_inputs(&proofs);
    }

    Ok((stdin, def_inputs))
}
