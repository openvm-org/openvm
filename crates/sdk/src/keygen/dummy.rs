use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::{
    instructions::{
        exe::VmExe, instruction::Instruction, program::Program, LocalOpcode, SystemOpcode,
    },
    Executor, MeteredExecutor, PreflightExecutor, SystemConfig, VmBuilder, VmExecutionConfig,
};
use openvm_continuations::RootSC;
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey, p3_field::PrimeField32, prover::ProvingContext,
    StarkEngine, SystemParams, Val,
};
#[cfg(feature = "evm-prove")]
use {
    crate::prover::{EvmProver, RootProver},
    openvm_stark_backend::proof::Proof,
};

use crate::{
    prover::{vm::types::VmProvingKey, AggProver, DeferralPathProver, StarkProver},
    StdIn, F, SC,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_continuations::prover::RootGpuProver as RootInnerProver;
        type RootE = openvm_cuda_backend::BabyBearBn254Poseidon2GpuEngine;
    } else {
        use openvm_continuations::prover::RootCpuProver as RootInnerProver;
        type RootE = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
    }
}

fn dummy_terminate_exe() -> Arc<VmExe<F>> {
    let dummy_program = Program::<F>::from_instructions(&[Instruction::from_isize(
        SystemOpcode::TERMINATE.global_opcode(),
        0,
        0,
        0,
        0,
        0,
    )]);
    Arc::new(VmExe::new(dummy_program))
}

pub fn compute_root_proof_heights<E, VB>(
    vm_builder: VB,
    app_vm_pk: &VmProvingKey<VB::VmConfig>,
    agg_prover: Arc<AggProver>,
    root_params: SystemParams,
    def_prover: Option<Arc<DeferralPathProver>>,
) -> Result<(Vec<usize>, Arc<MultiStarkProvingKey<RootSC>>)>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E>,
    VB::VmConfig: VmExecutionConfig<F> + AsRef<SystemConfig>,
    Val<SC>: PrimeField32,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
{
    let dummy_exe = dummy_terminate_exe();

    let system_config = app_vm_pk.vm_config.as_ref();
    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let num_user_pvs = system_config.num_public_values;

    let def_hook_commit = def_prover.as_ref().map(|p| p.def_hook_commit().into());

    let mut stark_prover = StarkProver::<E, VB>::new(
        vm_builder,
        app_vm_pk,
        dummy_exe,
        agg_prover.clone(),
        def_prover,
    )?;
    let (agg_proof, _) = stark_prover.prove(StdIn::default(), &[])?;

    let root_prover = RootInnerProver::new::<RootE>(
        agg_prover.internal_recursive_prover.get_vk(),
        agg_prover
            .internal_recursive_prover
            .get_self_vk_pcs_data()
            .unwrap()
            .commitment
            .into(),
        root_params,
        memory_dimensions,
        num_user_pvs,
        def_hook_commit,
        None,
    );

    let root_proving_ctx: ProvingContext<<RootE as StarkEngine>::PB> = root_prover
        .generate_proving_ctx(
            agg_proof.inner,
            &agg_proof.user_pvs_proof,
            agg_proof.deferral_merkle_proofs.as_ref(),
            #[cfg(feature = "cuda")]
            None,
        )
        .unwrap();

    let trace_heights = root_proving_ctx
        .into_iter()
        .map(|(_, air_ctx)| air_ctx.height())
        .collect();
    Ok((trace_heights, root_prover.get_pk()))
}

/// Generate a dummy root proof for keygen purposes.
///
/// Runs a trivial TERMINATE-only program through the full EVM prover pipeline
/// (app → aggregation → root) and returns the resulting root proof.
#[cfg(feature = "evm-prove")]
pub fn generate_dummy_root_proof<E, VB>(
    vm_builder: VB,
    app_vm_pk: &VmProvingKey<VB::VmConfig>,
    agg_prover: Arc<AggProver>,
    def_path_prover: Option<Arc<DeferralPathProver>>,
    root_prover: Arc<RootProver>,
) -> Proof<RootSC>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E> + Clone,
    Val<SC>: PrimeField32,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
{
    let dummy_exe = dummy_terminate_exe();

    let mut evm_prover = EvmProver::<E, _>::new(
        vm_builder,
        app_vm_pk,
        dummy_exe,
        agg_prover,
        def_path_prover,
        root_prover,
        None,
    )
    .expect("Failed to create dummy EVM prover");

    evm_prover
        .prove_unwrapped(StdIn::default(), &[])
        .expect("Failed to generate dummy root proof")
}
