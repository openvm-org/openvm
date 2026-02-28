use std::sync::Arc;

use eyre::Result;
use itertools::Itertools;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{
        CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey, ProverBackend,
    },
    AirRef, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use recursion_circuit::system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen};
use tracing::instrument;

use crate::{
    circuit::nonroot::{receiver::UserPvsReceiverAir, verifier::VerifierPvsAir, NonRootTraceGen},
    prover::{trace_heights_tracing_info, Circuit},
    SC,
};

mod trace;

#[derive(derive_new::new, Clone)]
pub struct NonRootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<S: AggregationSubCircuit> Circuit for NonRootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let public_values_bus = bus_inventory.public_values_bus;
        [Arc::new(VerifierPvsAir {
            public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
        }) as AirRef<SC>]
        .into_iter()
        .chain(self.verifier_circuit.airs())
        .chain([Arc::new(UserPvsReceiverAir { public_values_bus }) as AirRef<SC>])
        .collect_vec()
    }
}

/// Generates an aggregation proof for non-root layers (leaf and internal).
pub struct NonRootAggregationProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T: NonRootTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    d_pk: DeviceMultiStarkProvingKey<PB>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    // TODO: tracegen currently requires storing these, we should revisit this
    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<NonRootCircuit<S>>,

    self_vk_pcs_data: Option<CommittedTraceData<PB>>,
}

/// Struct to determine if NonRootAggregationProver is proving a special case,
/// i.e. if the child_vk is the app_vk or if it should use its own vk as child.
pub enum ChildVkKind {
    Standard,
    App,
    RecursiveSelf,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: NonRootTraceGen<PB>,
    > NonRootAggregationProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn agg_prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
    ) -> Result<Proof<SC>> {
        let ctx = self.generate_proving_ctx(proofs, child_vk_kind);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
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
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: NonRootTraceGen<PB>,
    > NonRootAggregationProver<PB, S, T>
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
                has_cached: true,
                ..Default::default()
            },
        );
        let engine = E::new(system_params);
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(NonRootCircuit::new(Arc::new(verifier_circuit)));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        Self {
            pk: Arc::new(pk),
            d_pk,
            vk: Arc::new(vk),
            agg_node_tracegen: NonRootTraceGen::new(),
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
                has_cached: true,
                ..Default::default()
            },
        );
        let engine = E::new(pk.params.clone());
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(NonRootCircuit::new(Arc::new(verifier_circuit)));
        let vk = Arc::new(pk.get_vk());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        Self {
            pk,
            d_pk,
            vk,
            agg_node_tracegen: NonRootTraceGen::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }

    pub fn get_circuit(&self) -> Arc<NonRootCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKey<SC>> {
        self.pk.clone()
    }

    pub fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.vk.clone()
    }

    pub fn get_cached_commit(&self, is_self_recursive: bool) -> PB::Commitment {
        if is_self_recursive {
            self.self_vk_pcs_data.as_ref().unwrap().commitment
        } else {
            self.child_vk_pcs_data.commitment
        }
    }

    pub fn get_self_vk_pcs_data(&self) -> Option<CommittedTraceData<PB>>
    where
        CommittedTraceData<PB>: Clone,
    {
        self.self_vk_pcs_data.clone()
    }
}
