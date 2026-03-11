use std::sync::Arc;

use eyre::Result;
use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{CommittedTraceData, DeviceDataTransporter, ProverBackend},
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, EF, F};
use p3_field::{Field, PrimeField32};
use recursion_circuit::system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen};
use tracing::instrument;

use crate::{
    bn254::{CommitBytes, DagCommitBytes},
    circuit::{
        deferral::verify::{DeferralMerkleProofs, DeferredVerifyCircuit, DeferredVerifyTraceGen},
        Circuit,
    },
    prover::trace_heights_tracing_info,
    SC,
};

mod trace;

pub struct DeferredVerifyProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T: DeferredVerifyTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<DeferredVerifyCircuit<S>>,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: DeferredVerifyTraceGen<PB>,
    > DeferredVerifyProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<PB::Val>>,
    ) -> Result<Proof<SC>> {
        let ctx = self.generate_proving_ctx(proof, user_pvs_proof, deferral_merkle_proofs);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, SC>(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        crate::prover::debug_constraints(&self.circuit, &ctx, &engine);
        let d_pk = engine.device().transport_pk_to_device(self.pk.as_ref());
        let proof = engine.prove(&d_pk, ctx)?;
        #[cfg(debug_assertions)]
        engine.verify(&self.vk, &proof)?;
        Ok(proof)
    }

    #[instrument(name = "total_proof", skip_all)]
    pub fn prove_no_def<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> Result<Proof<SC>> {
        self.prove::<E>(proof, user_pvs_proof, None)
    }
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC>,
        T: DeferredVerifyTraceGen<PB>,
    > DeferredVerifyProver<PB, S, T>
{
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        child_vk_pcs_data: CommittedTraceData<PB>,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<PB::Commitment>,
    ) -> Self
    where
        E::PD: DeviceDataTransporter<SC, PB> + Clone,
        PB::Val: Field + PrimeField32,
        PB::Matrix: Clone,
        PB::Commitment: Into<CommitBytes>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                final_state_bus_enabled: true,
            },
        );
        let engine = E::new(system_params);
        let internal_recursive_dag_commit = DagCommitBytes {
            cached_commit: child_vk_pcs_data.commitment.into(),
            pre_hash: child_vk.pre_hash.into(),
        };
        let def_hook_commit = def_hook_commit.map(Into::into);
        let circuit = Arc::new(DeferredVerifyCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_dag_commit,
            def_hook_commit,
            memory_dimensions,
            num_user_pvs,
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        Self {
            pk: Arc::new(pk),
            vk: Arc::new(vk),
            agg_node_tracegen: T::new(def_hook_commit.is_some()),
            child_vk,
            child_vk_pcs_data,
            circuit,
        }
    }

    pub fn from_pk(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        child_vk_pcs_data: CommittedTraceData<PB>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<SC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<PB::Commitment>,
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
                final_state_bus_enabled: true,
            },
        );
        let def_hook_commit = def_hook_commit.map(Into::into);
        let internal_recursive_dag_commit = DagCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            pre_hash: child_vk.pre_hash.into(),
        };
        let circuit = Arc::new(DeferredVerifyCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_dag_commit,
            def_hook_commit,
            memory_dimensions,
            num_user_pvs,
        ));
        let vk = Arc::new(pk.get_vk());
        Self {
            pk,
            vk,
            agg_node_tracegen: T::new(def_hook_commit.is_some()),
            child_vk,
            child_vk_pcs_data,
            circuit,
        }
    }

    pub fn get_circuit(&self) -> Arc<DeferredVerifyCircuit<S>> {
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
