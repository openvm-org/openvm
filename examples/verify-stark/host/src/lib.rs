use std::{path::PathBuf, slice::from_ref, sync::Arc};

use eyre::Result;
use openvm_build::GuestOptions;
use openvm_continuations::CommitBytes;
use openvm_deferral_circuit::DeferralFn;
use openvm_recursion_circuit::batch_constraint::commit_child_vk;
use openvm_sdk::{
    config::{AggregationConfig, AppConfig},
    keygen::{AppProvingKey, SdkCachedProvingKey},
    openvm_circuit::arch::instructions::exe::VmExe,
    prover::{vm::types::VmProvingKey, DeferralPathProver, DeferralProver},
    types::VersionedVmStarkProof,
    DefaultStarkEngine as E, DeferralInput, Sdk, StdIn, F, SC,
};
use openvm_sdk_config::{SdkSystemConfig, SdkVmConfig};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, StarkEngine};
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, hook_params_with_100_bits_security,
    internal_params_with_100_bits_security, MAX_APP_LOG_STACKED_HEIGHT,
};
use openvm_verify_stark_circuit::extension::{
    get_deferral_state, get_raw_deferral_results, verify_stark_deferral_fn,
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

pub type Proof = VmStarkProof;
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

    let deferral_path_prover = {
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
        let deferral_prover = Arc::new(DeferralProver::new(
            verify_circuit_prover,
            agg_config.clone(),
            hook_params_with_100_bits_security(),
        ));
        DeferralPathProver::from_config(agg_config.clone(), deferral_prover)
    };

    let deferral_fn = Arc::new(DeferralFn::new(verify_stark_deferral_fn));
    let deferral_ext = deferral_path_prover
        .deferral_prover
        .make_extension(vec![deferral_fn]);
    let mut vm_config = SdkVmConfig::riscv32();
    vm_config.deferral = Some(deferral_ext);
    vm_config = vm_config.optimize();

    let sdk = Sdk::builder()
        .app_config(AppConfig::new(vm_config.clone(), app_params))
        .agg_params(agg_config.params)
        .deferral_path_prover(deferral_path_prover)
        .build()?;
    let _ = sdk.app_keygen();
    let _ = sdk.agg_pk();
    let agg_vk = sdk.agg_vk().as_ref().clone();
    Ok((sdk.cached_proving_key(), vm_config, agg_vk))
}

pub fn sdk_from_cache(
    cached_pk: SdkCachedProvingKey<SdkVmConfig>,
    child_agg_vk: Arc<MultiStarkVerifyingKey<SC>>,
) -> Result<Sdk> {
    let app_pk = cached_pk.app_pk.unwrap();
    let agg_pk: openvm_sdk::keygen::AggProvingKey = cached_pk.agg_pk.unwrap();
    let deferral_pk = cached_pk.deferral_pk.unwrap();
    let deferral_agg_pk = cached_pk.deferral_agg_pk.unwrap();

    let default_config = SdkSystemConfig::default();
    let memory_dimensions = default_config.config.memory_config.memory_dimensions();
    let num_user_pvs = default_config.config.num_public_values;

    let child_internal_recursive_cached_commit = cached_commit(&child_agg_vk);
    let verify_prover = VerifyProver::from_pk::<E>(
        child_agg_vk,
        child_internal_recursive_cached_commit,
        deferral_pk.circuits[VERIFY_STARK_DEF_IDX]
            .def_circuit_pk
            .clone(),
        memory_dimensions,
        num_user_pvs,
        None,
        VERIFY_STARK_DEF_IDX,
    );
    let verify_circuit_prover = VerifyCircuitProver::new(verify_prover);
    let deferral_prover = Arc::new(DeferralProver::from_pks(
        verify_circuit_prover,
        deferral_pk.circuits[VERIFY_STARK_DEF_IDX]
            .agg_prefix_pk
            .clone(),
        deferral_pk.def_internal_recursive_pk.clone(),
        deferral_pk.def_hook_pk.clone(),
    ));
    let deferral_path_prover = DeferralPathProver::from_pk(deferral_agg_pk, deferral_prover);

    let deferral_fn = Arc::new(DeferralFn::new(verify_stark_deferral_fn));
    let mut app_vm_config = app_pk.app_vm_pk.vm_config.clone();
    app_vm_config
        .deferral
        .as_mut()
        .expect("cached verify-stark app proving key should include deferral extension")
        .fns = vec![deferral_fn];
    let app_pk = AppProvingKey {
        app_vm_pk: Arc::new(VmProvingKey {
            vm_config: app_vm_config,
            vm_pk: app_pk.app_vm_pk.vm_pk.clone(),
        }),
    };

    let sdk = Sdk::builder()
        .app_pk(app_pk)
        .agg_pk(agg_pk)
        .deferral_path_prover(deferral_path_prover)
        .build()?;
    Ok(sdk)
}

pub fn build(
    cached_pk: SdkCachedProvingKey<SdkVmConfig>,
    child_agg_vk: Arc<MultiStarkVerifyingKey<SC>>,
) -> Result<(Arc<VmExe<F>>, VerificationBaseline)> {
    let sdk = sdk_from_cache(cached_pk, child_agg_vk.clone())?;
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
    let sdk = sdk_from_cache(cached_pk, child_agg_vk.clone())?;
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
