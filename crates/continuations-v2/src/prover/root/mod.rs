use std::sync::Arc;

use eyre::Result;
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{CommittedTraceData, DeviceDataTransporter, ProverBackend, ProvingContext},
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use p3_field::{Field, PrimeField32};
use recursion_circuit::system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen};
use tracing::instrument;

use crate::{
    bn254::CommitBytes,
    circuit::{
        root::{RootCircuit, RootTraceGen},
        Circuit,
    },
    prover::trace_heights_tracing_info,
    SC,
};

mod trace;

pub struct RootProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T: RootTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<RootCircuit<S>>,
    trace_heights: Option<Vec<usize>>,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: RootTraceGen<PB>,
    > RootProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn root_prove_from_ctx<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        ctx: ProvingContext<PB>,
    ) -> Result<Proof<SC>> {
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
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
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: RootTraceGen<PB>,
    > RootProver<PB, S, T>
{
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        child_vk_pcs_data: CommittedTraceData<PB>,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        trace_heights: Option<Vec<usize>>,
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
                has_cached: true,
                ..Default::default()
            },
        );
        let engine = E::new(system_params);
        let internal_recursive_dag_commit = child_vk_pcs_data.commitment.into();
        let circuit = Arc::new(RootCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_dag_commit,
            memory_dimensions,
            num_user_pvs,
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        Self {
            pk: Arc::new(pk),
            vk: Arc::new(vk),
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            trace_heights,
        }
    }

    pub fn from_pk(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        child_vk_pcs_data: CommittedTraceData<PB>,
        pk: Arc<MultiStarkProvingKey<SC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        trace_heights: Option<Vec<usize>>,
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
        let internal_recursive_dag_commit = child_vk_pcs_data.commitment.into();
        let circuit = Arc::new(RootCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_dag_commit,
            memory_dimensions,
            num_user_pvs,
        ));
        let vk = Arc::new(pk.get_vk());
        Self {
            pk,
            vk,
            agg_node_tracegen: T::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            trace_heights,
        }
    }

    pub fn get_circuit(&self) -> Arc<RootCircuit<S>> {
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

    pub fn get_trace_heights(&self) -> Option<Vec<usize>> {
        self.trace_heights.clone()
    }
}
