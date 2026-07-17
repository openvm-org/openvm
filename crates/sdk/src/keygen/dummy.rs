use std::sync::Arc;

use eyre::Result;
use openvm_circuit::arch::{
    instructions::{
        exe::VmExe, instruction::Instruction, program::Program, LocalOpcode, SystemOpcode,
    },
    SystemConfig,
};
use openvm_continuations::{prover::engine_device_ctx, RootSC};
use openvm_sdk_config::SdkVmBuilder;
use openvm_stark_backend::{
    keygen::types::MultiStarkProvingKey, prover::ProvingContext, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::{app_params_with_100_bits_security, MAX_APP_LOG_STACKED_HEIGHT};
#[cfg(feature = "evm-prove")]
use {
    crate::{
        prover::{vm::types::VmProvingKey, EvmProver, RootProver},
        SC,
    },
    openvm_circuit::arch::{
        Executor, MeteredExecutor, PreflightExecutor, VmBuilder, VmExecutionConfig,
    },
    openvm_stark_backend::{p3_field::PrimeField32, proof::Proof, prover::ProverDevice, Val},
};

use crate::{
    config::{AggregationConfig, AggregationSystemParams, AggregationTreeConfig, AppConfig},
    keygen::AppProvingKey,
    prover::{AggProver, StarkProver},
    DeferralSetup, StdIn, F,
};

type CpuRootE =
    openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        type ChildE = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        type ChildE = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
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

pub(crate) fn compute_root_proof_heights(
    system_config: SystemConfig,
    agg_params: AggregationSystemParams,
    agg_tree_config: AggregationTreeConfig,
    root_params: SystemParams,
    deferral_setup: DeferralSetup,
) -> Result<(Vec<usize>, Arc<MultiStarkProvingKey<RootSC>>)> {
    let dummy_exe = dummy_terminate_exe();

    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let num_user_pvs = system_config.num_public_values;

    let mut app_config = AppConfig::riscv64(app_params_with_100_bits_security(
        MAX_APP_LOG_STACKED_HEIGHT,
    ));
    app_config.app_vm_config.system.config = system_config;

    let def_hook_cached_commit = deferral_setup.hook_cached_commit();
    let def_hook_commit = deferral_setup.hook_commit().map(Into::into);

    let app_pk = AppProvingKey::keygen(app_config)?;

    let agg_prover = Arc::new(AggProver::new(
        Arc::new(app_pk.app_vm_pk.vm_pk.get_vk()),
        AggregationConfig { params: agg_params },
        agg_tree_config,
        def_hook_cached_commit,
    ));

    let mut stark_prover = StarkProver::<ChildE, SdkVmBuilder>::new(
        Default::default(),
        &app_pk.app_vm_pk,
        dummy_exe,
        agg_prover.clone(),
        deferral_setup,
    )?;
    stark_prover.set_program_name("root_keygen");
    let (agg_proof, _) = stark_prover.prove(StdIn::default(), &[])?;

    let root_prover = openvm_continuations::prover::RootCpuProver::new::<CpuRootE>(
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
    let engine = root_prover.create_engine::<CpuRootE>();
    let root_proving_ctx: ProvingContext<<CpuRootE as StarkEngine>::PB> = root_prover
        .generate_proving_ctx(
            agg_proof.inner,
            &agg_proof.user_pvs_proof,
            agg_proof.deferral_merkle_proofs.as_ref(),
            engine_device_ctx(&engine),
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
    deferral_setup: DeferralSetup,
    root_prover: Arc<RootProver>,
) -> Proof<RootSC>
where
    E: StarkEngine<SC = SC>,
    VB: VmBuilder<E> + Clone,
    Val<SC>: PrimeField32,
    <E::PD as ProverDevice<E::PB, E::TS>>::DeviceCtx: 'static,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
{
    let dummy_exe = dummy_terminate_exe();

    let mut evm_prover = EvmProver::<E, _>::new(
        vm_builder,
        app_vm_pk,
        dummy_exe,
        agg_prover,
        deferral_setup,
        root_prover,
        None,
    )
    .expect("Failed to create dummy EVM prover");
    evm_prover.stark_prover.set_program_name("halo2_keygen");

    evm_prover
        .prove_root(StdIn::default(), &[])
        .expect("Failed to generate dummy root proof")
}
