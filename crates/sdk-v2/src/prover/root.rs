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
use openvm_sdk_config::SdkVmBuilder;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{CommittedTraceData, ProvingContext},
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use tracing::info_span;
use verify_stark::NonRootStarkProof;

use crate::{
    config::{
        default_app_params, AggregationConfig, AggregationSystemParams, AggregationTreeConfig,
        AppConfig, DEFAULT_APP_LOG_BLOWUP, DEFAULT_APP_L_SKIP,
    },
    keygen::AppProvingKey,
    prover::{AggProver, AppProver},
    StdIn,
};

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use continuations_v2::aggregation::RootGpuProver as RootInnerProver;
        type E = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    } else {
        use continuations_v2::aggregation::RootCpuProver as RootInnerProver;
        type E = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;
    }
}

pub struct RootProver(pub RootInnerProver);

impl RootProver {
    pub fn new(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<crate::SC>>,
        internal_recursive_vk_pcs_data: CommittedTraceData<<E as StarkEngine>::PB>,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        trace_heights: Option<Vec<usize>>,
    ) -> Self {
        let inner = RootInnerProver::new::<E>(
            internal_recursive_vk,
            internal_recursive_vk_pcs_data,
            system_params,
            memory_dimensions,
            num_user_pvs,
            trace_heights,
        );
        Self(inner)
    }

    pub fn from_pk(
        internal_recursive_vk: Arc<MultiStarkVerifyingKey<crate::SC>>,
        internal_recursive_vk_pcs_data: CommittedTraceData<<E as StarkEngine>::PB>,
        pk: Arc<MultiStarkProvingKey<crate::SC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        trace_heights: Option<Vec<usize>>,
    ) -> Self {
        let inner = RootInnerProver::from_pk(
            internal_recursive_vk,
            internal_recursive_vk_pcs_data,
            pk,
            memory_dimensions,
            num_user_pvs,
            trace_heights,
        );
        Self(inner)
    }

    pub fn generate_proving_ctx(
        &self,
        input: NonRootStarkProof,
    ) -> Option<ProvingContext<<E as StarkEngine>::PB>> {
        let ctx = info_span!("tracegen_attempt", group = format!("root")).in_scope(|| {
            self.0
                .generate_proving_ctx(input.inner, &input.user_pvs_proof)
        });
        ctx
    }

    pub fn prove_from_ctx(
        &self,
        ctx: ProvingContext<<E as StarkEngine>::PB>,
    ) -> Result<Proof<crate::SC>> {
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

    let mut app_config = AppConfig::riscv32(default_app_params(
        DEFAULT_APP_LOG_BLOWUP,
        DEFAULT_APP_L_SKIP,
        21 - DEFAULT_APP_L_SKIP,
    ));
    app_config.app_vm_config.system.config = system_config;

    let app_pk = AppProvingKey::keygen(app_config)?;
    let mut app_prover =
        AppProver::<E, SdkVmBuilder>::new(Default::default(), &app_pk.app_vm_pk, dummy_exe)?;
    let app_proof = app_prover.prove(StdIn::default())?;

    let agg_prover = AggProver::new(
        Arc::new(app_pk.app_vm_pk.vm_pk.get_vk()),
        AggregationConfig { params: agg_params },
        agg_tree_config,
    );
    let (agg_proof, _) = agg_prover.prove(app_proof)?;

    let root_prover = RootInnerProver::new::<E>(
        agg_prover.internal_recursive_prover.get_vk(),
        agg_prover
            .internal_recursive_prover
            .get_self_vk_pcs_data()
            .unwrap(),
        root_params,
        memory_dimensions,
        num_user_pvs,
        None,
    );
    let root_proving_ctx = root_prover
        .generate_proving_ctx(agg_proof.inner, &agg_proof.user_pvs_proof)
        .unwrap();

    let ret = root_proving_ctx
        .into_iter()
        .map(|(_, air_ctx)| air_ctx.height())
        .collect();
    Ok(ret)
}
