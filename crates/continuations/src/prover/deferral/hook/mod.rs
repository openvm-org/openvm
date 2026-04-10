use std::sync::Arc;

use eyre::Result;
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
#[cfg(feature = "cuda")]
use openvm_recursion_circuit::system::GpuVerifierTraceGen;
use openvm_recursion_circuit::system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen};
#[cfg(feature = "cuda")]
use openvm_stark_backend::prover::ProverDevice;
#[cfg(feature = "cuda")]
use openvm_stark_backend::EngineDeviceCtx;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{
        CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey, ProverBackend,
    },
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use p3_field::{Field, PrimeField32};
use tracing::instrument;

use crate::{
    circuit::{
        deferral::hook::{DeferralHookCircuit, DeferralHookTraceGen, DeferralIoCommit},
        Circuit,
    },
    prover::trace_heights_tracing_info,
    CommitBytes, VkCommitBytes, SC,
};

mod trace;

pub struct DeferralHookProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    d_pk: DeviceMultiStarkProvingKey<PB>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<DeferralHookCircuit<S>>,
}

impl<S, T> DeferralHookProver<CpuBackend<SC>, S, T>
where
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<SC>, SC>,
    T: DeferralHookTraceGen<CpuBackend<SC>>,
    <CpuBackend<SC> as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn prove<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        &self,
        proof: Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> Result<Proof<SC>> {
        let engine = E::new(self.pk.params.clone());
        let proving_ctx = self.generate_proving_ctx(proof, leaf_children);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, SC>(&proving_ctx.per_trace, &self.circuit.airs());
        }
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            crate::prover::debug_constraints(&self.circuit, &proving_ctx, &engine);
        }
        let proof = engine.prove(&self.d_pk, proving_ctx)?;
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            engine.verify(&self.vk, &proof)?;
        }
        Ok(proof)
    }
    pub fn new<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        system_params: SystemParams,
    ) -> Self
    where
        <CpuBackend<SC> as ProverBackend>::Val: Field + PrimeField32,
        <CpuBackend<SC> as ProverBackend>::Commitment: Into<CommitBytes>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                ..Default::default()
            },
        );
        let engine = E::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            vk_pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(DeferralHookCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let d_pk = engine.device().transport_pk_to_device(&pk);

        Self {
            pk: Arc::new(pk),
            d_pk,
            vk: Arc::new(vk),
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
        }
    }

    pub fn from_pk<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<SC>>,
    ) -> Self
    where
        <CpuBackend<SC> as ProverBackend>::Val: Field + PrimeField32,
        <CpuBackend<SC> as ProverBackend>::Commitment: Into<CommitBytes>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                ..Default::default()
            },
        );
        let engine = E::new(pk.params.clone());
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            vk_pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(DeferralHookCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
        ));
        let vk = Arc::new(pk.get_vk());
        let d_pk = engine.device().transport_pk_to_device(pk.as_ref());
        Self {
            pk,
            d_pk,
            vk,
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
        }
    }
}

#[cfg(feature = "cuda")]
impl<S, T> DeferralHookProver<GpuBackend, S, T>
where
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<SC>, SC>
        + GpuVerifierTraceGen<GpuBackend, SC>,
    T: DeferralHookTraceGen<CpuBackend<SC>>,
    <GpuBackend as ProverBackend>::Val: Field + PrimeField32,
    <GpuBackend as ProverBackend>::Matrix: Clone,
    <GpuBackend as ProverBackend>::Commitment: Into<CommitBytes>,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn prove<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        &self,
        proof: Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> Result<Proof<SC>>
    where
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
    {
        let engine = E::new(self.pk.params.clone());
        let device_ctx: openvm_cuda_common::stream::GpuDeviceCtx =
            engine.device().device_ctx().clone().into();
        let proving_ctx = self.generate_proving_ctx(proof, leaf_children, &device_ctx);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, SC>(&proving_ctx.per_trace, &self.circuit.airs());
        }
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            crate::prover::debug_constraints(&self.circuit, &proving_ctx, &engine);
        }
        let proof = engine.prove(&self.d_pk, proving_ctx)?;
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            engine.verify(&self.vk, &proof)?;
        }
        Ok(proof)
    }

    pub fn new<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        system_params: SystemParams,
    ) -> Self
    where
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                ..Default::default()
            },
        );
        let engine = E::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk_gpu(&engine, &child_vk);
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            vk_pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(DeferralHookCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let d_pk = engine.device().transport_pk_to_device(&pk);

        Self {
            pk: Arc::new(pk),
            d_pk,
            vk: Arc::new(vk),
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
        }
    }

    pub fn from_pk<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<SC>>,
    ) -> Self
    where
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                ..Default::default()
            },
        );
        let engine = E::new(pk.params.clone());
        let child_vk_pcs_data = verifier_circuit.commit_child_vk_gpu(&engine, &child_vk);
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            vk_pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(DeferralHookCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
        ));
        let vk = Arc::new(pk.get_vk());
        let d_pk = engine.device().transport_pk_to_device(pk.as_ref());
        Self {
            pk,
            d_pk,
            vk,
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
        }
    }
}

impl<PB, S, T> DeferralHookProver<PB, S, T>
where
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    PB::Matrix: Clone,
{
    pub fn get_circuit(&self) -> Arc<DeferralHookCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKey<SC>> {
        self.pk.clone()
    }

    pub fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.vk.clone()
    }

    pub fn get_cached_commit(&self) -> <PB as ProverBackend>::Commitment {
        self.child_vk_pcs_data.commitment
    }
}
