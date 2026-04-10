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
use openvm_verify_stark_host::pvs::{DeferralPvs, VkCommit};
use tracing::instrument;

use crate::{
    circuit::{
        inner::{InnerCircuit, InnerTraceGen, ProofsType},
        Circuit,
    },
    prover::trace_heights_tracing_info,
    SC,
};

mod trace;

/// Generates an aggregation proof for inner layers (leaf and internal).
pub struct InnerAggregationProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    d_pk: DeviceMultiStarkProvingKey<PB>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    // TODO: tracegen currently requires storing these, we should revisit this
    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<InnerCircuit<S>>,

    self_vk_pcs_data: Option<CommittedTraceData<PB>>,
}

/// Struct to determine if InnerAggregationProver is proving a special case,
/// i.e. if the child_vk is the app_vk or if it should use its own vk as child.
#[derive(Clone, Copy)]
pub enum ChildVkKind {
    Standard,
    App,
    RecursiveSelf,
}

impl<S, T> InnerAggregationProver<CpuBackend<SC>, S, T>
where
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<SC>, SC>,
    T: InnerTraceGen<CpuBackend<SC>>,
    <CpuBackend<SC> as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn agg_prove<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
    ) -> Result<Proof<SC>> {
        let engine = E::new(self.pk.params.clone());
        let ctx = self.generate_proving_ctx(proofs, child_vk_kind, proofs_type, absent_trace_pvs);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, SC>(&ctx.per_trace, &self.circuit.airs());
        }
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            crate::prover::debug_constraints(&self.circuit, &ctx, &engine);
        }
        let proof = engine.prove(&self.d_pk, ctx)?;
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            engine.verify(&self.vk, &proof)?;
        }
        Ok(proof)
    }

    pub fn agg_prove_no_def<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
    ) -> Result<Proof<SC>> {
        self.agg_prove::<E>(proofs, child_vk_kind, ProofsType::Vm, None)
    }
    pub fn new<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                ..Default::default()
            },
        );
        let engine = E::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(InnerCircuit::new(
            Arc::new(verifier_circuit),
            def_hook_cached_commit.map(|d| d.into()),
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        let agg_node_tracegen = InnerTraceGen::new(def_hook_cached_commit.is_some());
        Self {
            pk: Arc::new(pk),
            d_pk,
            vk: Arc::new(vk),
            agg_node_tracegen,
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }

    pub fn from_pk<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        pk: Arc<MultiStarkProvingKey<SC>>,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                ..Default::default()
            },
        );
        let engine = E::new(pk.params.clone());
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(InnerCircuit::new(
            Arc::new(verifier_circuit),
            def_hook_cached_commit.map(|d| d.into()),
        ));
        let vk = Arc::new(pk.get_vk());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        let agg_node_tracegen = InnerTraceGen::new(def_hook_cached_commit.is_some());
        Self {
            pk,
            d_pk,
            vk,
            agg_node_tracegen,
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }
}

#[cfg(feature = "cuda")]
impl<S, T> InnerAggregationProver<GpuBackend, S, T>
where
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<SC>, SC>
        + GpuVerifierTraceGen<GpuBackend, SC>,
    T: InnerTraceGen<CpuBackend<SC>>,
    <GpuBackend as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn agg_prove<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
    ) -> Result<Proof<SC>>
    where
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
    {
        let engine = E::new(self.pk.params.clone());
        let device_ctx: openvm_cuda_common::stream::GpuDeviceCtx =
            engine.device().device_ctx().clone().into();
        let ctx = self.generate_proving_ctx(
            proofs,
            child_vk_kind,
            proofs_type,
            absent_trace_pvs,
            &device_ctx,
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, SC>(&ctx.per_trace, &self.circuit.airs());
        }
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            crate::prover::debug_constraints(&self.circuit, &ctx, &engine);
        }
        let proof = engine.prove(&self.d_pk, ctx)?;
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            engine.verify(&self.vk, &proof)?;
        }
        Ok(proof)
    }

    pub fn agg_prove_no_def<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
    ) -> Result<Proof<SC>>
    where
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
    {
        self.agg_prove::<E>(proofs, child_vk_kind, ProofsType::Vm, None)
    }

    pub fn new<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
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
        let circuit = Arc::new(InnerCircuit::new(
            Arc::new(verifier_circuit),
            def_hook_cached_commit.map(|d| d.into()),
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk_gpu(&engine, &vk))
        } else {
            None
        };
        let agg_node_tracegen = InnerTraceGen::new(def_hook_cached_commit.is_some());
        Self {
            pk: Arc::new(pk),
            d_pk,
            vk: Arc::new(vk),
            agg_node_tracegen,
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }

    pub fn from_pk<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        pk: Arc<MultiStarkProvingKey<SC>>,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
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
        let circuit = Arc::new(InnerCircuit::new(
            Arc::new(verifier_circuit),
            def_hook_cached_commit.map(|d| d.into()),
        ));
        let vk = Arc::new(pk.get_vk());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk_gpu(&engine, &vk))
        } else {
            None
        };
        let agg_node_tracegen = InnerTraceGen::new(def_hook_cached_commit.is_some());
        Self {
            pk,
            d_pk,
            vk,
            agg_node_tracegen,
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }
}

impl<PB, S, T> InnerAggregationProver<PB, S, T>
where
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    PB::Matrix: Clone,
{
    pub fn get_circuit(&self) -> Arc<InnerCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKey<SC>> {
        self.pk.clone()
    }

    pub fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.vk.clone()
    }

    pub fn deferral_enabled(&self) -> bool {
        self.circuit.def_hook_cached_commit.is_some()
    }

    pub fn get_vk_commit(&self, is_self_recursive: bool) -> VkCommit<PB::Val> {
        if is_self_recursive {
            VkCommit {
                cached_commit: self.self_vk_pcs_data.as_ref().unwrap().commitment,
                vk_pre_hash: self.vk.pre_hash,
            }
        } else {
            VkCommit {
                cached_commit: self.child_vk_pcs_data.commitment,
                vk_pre_hash: self.child_vk.pre_hash,
            }
        }
    }

    pub fn get_self_vk_pcs_data(&self) -> Option<CommittedTraceData<PB>>
    where
        CommittedTraceData<PB>: Clone,
    {
        self.self_vk_pcs_data.clone()
    }
}
