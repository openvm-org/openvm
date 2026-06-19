use std::{path::PathBuf, slice::from_ref, sync::Arc};

use eyre::Result;
use openvm_build::GuestOptions;
use openvm_continuations::CommitBytes;
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
    app_params_with_100_bits_security, hook_params_with_100_bits_security,
    internal_params_with_100_bits_security, MAX_APP_LOG_STACKED_HEIGHT,
};
use openvm_verify_stark_circuit::extension::{get_deferral_state, get_raw_deferral_results};
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
) -> Result<(
    SdkCachedProvingKey<SdkVmConfig>,
    SdkVmConfig,
    MultiStarkVerifyingKey<SC>,
)> {
    let default_config = SdkSystemConfig::default();
    let memory_dimensions = default_config.config.memory_config.memory_dimensions();
    let num_user_pvs = default_config.config.num_public_values;
    let verify_prover_params = internal_params_with_100_bits_security();

    let app_params = app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT);
    let agg_config = AggregationConfig::default();

    let deferral_agg_prover = {
        let child_internal_recursive_cached_commit = cached_commit(&child_agg_vk);
        let verify_prover = VerifyProver::new::<E>(
            child_agg_vk,
            child_internal_recursive_cached_commit,
            verify_prover_params,
            memory_dimensions,
            num_user_pvs,
            None,
            VERIFY_STARK_DEF_IDX,
        );
        let verify_circuit_prover = VerifyCircuitProver::new(verify_prover);
        let multi_deferral_circuit_prover = Arc::new(MultiDeferralCircuitProver::new(
            verify_circuit_prover,
            agg_config.clone(),
            hook_params_with_100_bits_security(),
        ));
        DeferralAggProver::new(agg_config.clone(), multi_deferral_circuit_prover)
    };

    let deferral_config = deferral_agg_prover
        .multi_deferral_circuit_prover
        .make_config(vec![SupportedDeferral::VerifyStark]);
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .deferral(deferral_config)
        .build()
        .optimize();

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
    let sdk = Sdk::from_deferral_cached_proving_key(cached_pk)?;
    let elf = sdk.build(
        GuestOptions::default(),
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
) -> Result<VersionedVmStarkProof> {
    let sdk = Sdk::from_deferral_cached_proving_key(cached_pk)?;
    let (stdin, def_input) = verify_stark_guest_inputs(
        &input_proof.try_into()?,
        child_agg_vk.as_ref().clone(),
        child_baseline,
    )?;
    let (proof, _) = sdk.prove(exe, stdin, &[def_input])?;
    VersionedVmStarkProof::new(proof)
}

fn verify_stark_guest_inputs(
    proof: &VmStarkProof,
    agg_vk: MultiStarkVerifyingKey<SC>,
    baseline: VerificationBaseline,
) -> Result<(StdIn, DeferralInput)> {
    let child_vk = VmStarkVerifyingKey {
        mvk: agg_vk,
        baseline,
    };

    let raw_results = get_raw_deferral_results(&child_vk, from_ref(proof))?;
    assert_eq!(raw_results.len(), 1);

    let input_commit: [u8; 32] = raw_results[0].input.clone().try_into().unwrap();

    let mut stdin = StdIn::default();
    stdin.write(&input_commit);
    stdin.deferrals = vec![get_deferral_state(&child_vk, from_ref(proof), 0)?];

    Ok((stdin, DeferralInput::from_inputs(from_ref(proof))))
}

fn cached_commit(child_agg_vk: &Arc<MultiStarkVerifyingKey<SC>>) -> CommitBytes {
    let engine = <E as StarkEngine>::new(child_agg_vk.inner.params.clone());
    commit_child_vk(&engine, &child_agg_vk, true)
        .commitment
        .into()
}
