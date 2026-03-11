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
use p3_field::{Field, PrimeField32};
use recursion_circuit::system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen};
use tracing::instrument;

use crate::{
    bn254::{CommitBytes, DagCommitBytes},
    circuit::{
        deferral::aggregation::hook::{
            DeferralHookCircuit, DeferralHookTraceGen, DeferralIoCommit,
        },
        Circuit,
    },
    prover::trace_heights_tracing_info,
    SC,
};

mod trace;

pub struct DeferralHookProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T: DeferralHookTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    d_pk: DeviceMultiStarkProvingKey<PB>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<DeferralHookCircuit<S>>,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: DeferralHookTraceGen<PB>,
    > DeferralHookProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proof: Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> Result<Proof<SC>> {
        let ctx = self.generate_proving_ctx(proof, leaf_children);
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
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: DeferralHookTraceGen<PB>,
    > DeferralHookProver<PB, S, T>
{
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
    ) -> Self
    where
        PB::Val: Field + PrimeField32,
        PB::Matrix: Clone,
        PB::Commitment: Into<CommitBytes>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                has_cached: true,
                ..Default::default()
            },
        );
        let engine = E::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let internal_recursive_dag_commit = DagCommitBytes {
            cached_commit: child_vk_pcs_data.commitment.into(),
            pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(DeferralHookCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_dag_commit,
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

    pub fn from_pk<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<SC>>,
    ) -> Self
    where
        PB::Val: Field + PrimeField32,
        PB::Matrix: Clone,
        PB::Commitment: Into<CommitBytes>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                has_cached: true,
                ..Default::default()
            },
        );
        let engine = E::new(pk.params.clone());
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let internal_recursive_dag_commit = DagCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(DeferralHookCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_dag_commit,
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
