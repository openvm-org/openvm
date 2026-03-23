use std::sync::Arc;

use eyre::Result;
use openvm_circuit::{
    arch::{
        instructions::{
            exe::VmExe, instruction::Instruction, program::Program, LocalOpcode, SystemOpcode,
        },
        SystemConfig,
    },
    system::memory::dimensions::MemoryDimensions,
};
use openvm_continuations::{CommitBytes, RootSC, SC};
use openvm_sdk_config::SdkVmBuilder;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::ProvingContext,
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security,
    baby_bear_poseidon2::{Digest, F},
    MAX_APP_LOG_STACKED_HEIGHT,
};
use openvm_verify_stark_host::NonRootStarkProof;
use tracing::info_span;

use crate::{
    config::{AggregationConfig, AggregationSystemParams, AggregationTreeConfig, AppConfig},
    keygen::AppProvingKey,
    prover::{AggProver, DeferralPathProver, StarkProver},
    StdIn,
};

// CPU engine used for `compute_root_proof_heights` — trace heights are structural
// and backend-independent, so we always use the faster CPU engine for that step.
type CpuRootE =
    openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_continuations::prover::RootGpuProver as RootInnerProver;
        type E = openvm_cuda_backend::BabyBearBn254Poseidon2GpuEngine;
        type ChildE = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        use openvm_continuations::prover::RootCpuProver as RootInnerProver;
        type E = openvm_stark_sdk::config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine;
        type ChildE = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

pub struct RootProver(pub RootInnerProver);

impl RootProver {
    pub fn new(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_dag_commit: CommitBytes,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_vk_commit: Option<Digest>,
        trace_heights: Option<Vec<usize>>,
    ) -> Self {
        let inner = RootInnerProver::new::<E>(
            internal_recursive_vk,
            internal_recursive_dag_commit,
            system_params,
            memory_dimensions,
            num_user_pvs,
            def_hook_vk_commit.map(Into::into),
            trace_heights,
        );
        Self(inner)
    }

    pub fn from_pk(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_dag_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<RootSC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_vk_commit: Option<Digest>,
        trace_heights: Option<Vec<usize>>,
    ) -> Self {
        let inner = RootInnerProver::from_pk::<E>(
            internal_recursive_vk,
            internal_recursive_dag_commit,
            pk,
            memory_dimensions,
            num_user_pvs,
            def_hook_vk_commit.map(Into::into),
            trace_heights,
        );
        Self(inner)
    }

    pub fn generate_proving_ctx(
        &self,
        input: NonRootStarkProof,
    ) -> Option<ProvingContext<<E as StarkEngine>::PB>> {
        let ctx = info_span!("tracegen_attempt", group = format!("root")).in_scope(|| {
            self.0.generate_proving_ctx(
                input.inner,
                &input.user_pvs_proof,
                input.deferral_merkle_proofs.as_ref(),
            )
        });
        ctx
    }

    pub fn prove_from_ctx(
        &self,
        ctx: ProvingContext<<E as StarkEngine>::PB>,
    ) -> Result<Proof<RootSC>> {
        let proof = info_span!("agg_layer", group = format!("root"))
            .in_scope(|| info_span!("root").in_scope(|| self.0.root_prove_from_ctx::<E>(ctx)))?;
        Ok(proof)
    }
}

pub fn compute_root_proof_heights(
    system_config: SystemConfig,
    agg_params: AggregationSystemParams,
    agg_tree_config: AggregationTreeConfig,
    root_params: SystemParams,
    def_prover: Option<Arc<DeferralPathProver>>,
) -> Result<Vec<usize>> {
    let dummy_program = Program::<F>::from_instructions(&[Instruction::from_isize(
        SystemOpcode::TERMINATE.global_opcode(),
        0,
        0,
        0,
        0,
        0,
    )]);
    let dummy_exe = Arc::new(VmExe::new(dummy_program));

    let memory_dimensions = system_config.memory_config.memory_dimensions();
    let num_user_pvs = system_config.num_public_values;

    let mut app_config = AppConfig::riscv32(app_params_with_100_bits_security(
        MAX_APP_LOG_STACKED_HEIGHT,
    ));
    app_config.app_vm_config.system.config = system_config;

    let def_hook_cached_commit = def_prover.as_ref().map(|p| p.def_hook_cached_commit());
    let def_hook_vk_commit = def_prover.as_ref().map(|p| p.def_hook_vk_commit().into());

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
        def_prover,
    )?;
    let (agg_proof, _) = stark_prover.prove(StdIn::default(), &[])?;

    // Use the CPU engine for root keygen + tracegen: trace heights are structural
    // and backend-independent, so we avoid expensive GPU BN254 keygen here.
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
        def_hook_vk_commit,
        None,
    );
    let root_proving_ctx: ProvingContext<<CpuRootE as StarkEngine>::PB> = root_prover
        .generate_proving_ctx(
            agg_proof.inner,
            &agg_proof.user_pvs_proof,
            agg_proof.deferral_merkle_proofs.as_ref(),
        )
        .unwrap();

    let ret = root_proving_ctx
        .into_iter()
        .map(|(_, air_ctx)| air_ctx.height())
        .collect();
    Ok(ret)
}
