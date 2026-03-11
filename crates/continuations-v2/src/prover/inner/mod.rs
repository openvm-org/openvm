use std::sync::Arc;

use eyre::Result;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{
        CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey, ProverBackend,
    },
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use recursion_circuit::system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen};
use tracing::instrument;
use verify_stark::pvs::{DagCommit, DeferralPvs};

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
    T: InnerTraceGen<PB>,
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
pub enum ChildVkKind {
    Standard,
    App,
    RecursiveSelf,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: InnerTraceGen<PB>,
    > InnerAggregationProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn agg_prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
    ) -> Result<Proof<SC>> {
        let ctx = self.generate_proving_ctx(proofs, child_vk_kind, proofs_type, absent_trace_pvs);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, SC>(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        crate::prover::debug_constraints(&self.circuit, &ctx, &engine);
        let proof = engine.prove(&self.d_pk, ctx)?;
        #[cfg(debug_assertions)]
        engine.verify(&self.vk, &proof)?;
        Ok(proof)
    }

    pub fn agg_prove_no_def<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
    ) -> Result<Proof<SC>> {
        self.agg_prove::<E>(proofs, child_vk_kind, ProofsType::Vm, None)
    }
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: InnerTraceGen<PB>,
    > InnerAggregationProver<PB, S, T>
{
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_commit: Option<Digest>,
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
            def_hook_commit.map(|d| d.into()),
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        let agg_node_tracegen = InnerTraceGen::new(def_hook_commit.is_some());
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

    pub fn from_pk<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        pk: Arc<MultiStarkProvingKey<SC>>,
        is_self_recursive: bool,
        def_hook_commit: Option<Digest>,
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
            def_hook_commit.map(|d| d.into()),
        ));
        let vk = Arc::new(pk.get_vk());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        let agg_node_tracegen = InnerTraceGen::new(def_hook_commit.is_some());
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
        self.circuit.def_hook_commit.is_some()
    }

    pub fn get_dag_commit(&self, is_self_recursive: bool) -> DagCommit<PB::Val> {
        if is_self_recursive {
            DagCommit {
                cached_commit: self.self_vk_pcs_data.as_ref().unwrap().commitment,
                vk_pre_hash: self.vk.pre_hash,
            }
        } else {
            DagCommit {
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
