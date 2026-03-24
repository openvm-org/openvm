use std::sync::Arc;

use eyre::Result;
use openvm_recursion_circuit::system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen};
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{CommittedTraceData, DeviceMultiStarkProvingKey, ProverBackend},
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use openvm_verify_stark_host::pvs::DagCommit;
use tracing::instrument;

use crate::{
    circuit::{
        deferral::inner::{DeferralInnerCircuit, DeferralInnerTraceGen},
        Circuit,
    },
    prover::{keygen_for_proving_backend, trace_heights_tracing_info, transport_pk},
    SC,
};

mod trace;

pub enum DeferralChildVkKind {
    /// Child proofs are deferral verify proofs (consume DeferralCircuitPvs at air 0).
    DeferralCircuit,
    /// Child proofs are deferral aggregation inner proofs (consume DeferralAggregationPvs at air
    /// 1).
    DeferralAggregation,
    /// Same as DeferralAggregation but uses this prover's own vk as child vk.
    RecursiveSelf,
}

pub struct DeferralInnerProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T: DeferralInnerTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    d_pk: DeviceMultiStarkProvingKey<PB>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<DeferralInnerCircuit<S>>,

    self_vk_pcs_data: Option<CommittedTraceData<PB>>,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: DeferralInnerTraceGen<PB>,
    > DeferralInnerProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn agg_prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: DeferralChildVkKind,
        child_merkle_depth: Option<usize>,
    ) -> Result<Proof<SC>> {
        let ctx = self.generate_proving_ctx(proofs, child_vk_kind, child_merkle_depth);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, SC>(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
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
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: DeferralInnerTraceGen<PB>,
    > DeferralInnerProver<PB, S, T>
{
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
    ) -> Self {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                ..Default::default()
            },
        );
        let engine = E::new(system_params.clone());
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(DeferralInnerCircuit::new(Arc::new(verifier_circuit)));
        let airs = circuit.airs();
        let (pk, vk) = keygen_for_proving_backend(&engine, &airs, || {
            // Reuse CPU keygen for the backend-agnostic PK, then upload it for GPU proving.
            BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(system_params).keygen(&airs)
        });
        let d_pk = transport_pk(&engine, &pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        Self {
            pk: Arc::new(pk),
            d_pk,
            vk: Arc::new(vk),
            agg_node_tracegen: DeferralInnerTraceGen::new(),
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
        let circuit = Arc::new(DeferralInnerCircuit::new(Arc::new(verifier_circuit)));
        let vk = Arc::new(pk.get_vk());
        let d_pk = transport_pk(&engine, &pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        Self {
            pk,
            d_pk,
            vk,
            agg_node_tracegen: DeferralInnerTraceGen::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }

    pub fn get_circuit(&self) -> Arc<DeferralInnerCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKey<SC>> {
        self.pk.clone()
    }

    pub fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.vk.clone()
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
}
