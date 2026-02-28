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
    circuit::deferral::aggregation::nonroot::{
        bus::{InputOrMerkleCommitBus, PvAirConsistencyBus},
        def_pvs::DeferralPvsAir,
        input::InputCommitAir,
        verifier::NonRootPvsAir,
        DeferralNonRootTraceGen,
    },
    prover::{trace_heights_tracing_info, Circuit},
    SC,
};

mod trace;

#[derive(derive_new::new, Clone)]
pub struct DeferralNonRootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
}

impl<S: AggregationSubCircuit> Circuit for DeferralNonRootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();
        let input_or_merkle_commit_bus = InputOrMerkleCommitBus::new(next_bus_idx);
        let pv_air_consistency_bus = PvAirConsistencyBus::new(next_bus_idx + 1);

        let verifier_pvs_air = NonRootPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            pv_air_consistency_bus,
        };

        let def_pvs_air = DeferralPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            input_or_merkle_commit_bus,
            pv_air_consistency_bus,
        };

        let input_commit_air = InputCommitAir {
            public_values_bus: bus_inventory.public_values_bus,
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            input_or_merkle_commit_bus,
        };

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain([Arc::new(def_pvs_air) as AirRef<SC>])
            .chain(self.verifier_circuit.airs())
            .chain([Arc::new(input_commit_air) as AirRef<SC>])
            .collect_vec()
    }
}

pub enum DeferralChildVkKind {
    /// Child proofs are deferral verify proofs (consume DeferralCircuitPvs at air 0).
    DeferralCircuit,
    /// Child proofs are deferral aggregation nonroot proofs (consume DeferralAggregationPvs at air
    /// 1).
    DeferralAggregation,
    /// Same as DeferralAggregation but uses this prover's own vk as child vk.
    RecursiveSelf,
}

pub struct DeferralNonRootProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T: DeferralNonRootTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    d_pk: DeviceMultiStarkProvingKey<PB>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<DeferralNonRootCircuit<S>>,

    self_vk_pcs_data: Option<CommittedTraceData<PB>>,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: DeferralNonRootTraceGen<PB>,
    > DeferralNonRootProver<PB, S, T>
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
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        crate::prover::debug_constraints(&self.circuit, &ctx, &engine);
        let proof = engine.prove(&self.d_pk, ctx);
        #[cfg(debug_assertions)]
        engine.verify(&self.vk, &proof)?;
        Ok(proof)
    }
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: DeferralNonRootTraceGen<PB>,
    > DeferralNonRootProver<PB, S, T>
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
        let circuit = Arc::new(DeferralNonRootCircuit::new(Arc::new(verifier_circuit)));
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
            agg_node_tracegen: DeferralNonRootTraceGen::new(),
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
        let circuit = Arc::new(DeferralNonRootCircuit::new(Arc::new(verifier_circuit)));
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
            agg_node_tracegen: DeferralNonRootTraceGen::new(),
            child_vk,
            child_vk_pcs_data,
            circuit,
            self_vk_pcs_data,
        }
    }

    pub fn get_circuit(&self) -> Arc<DeferralNonRootCircuit<S>> {
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
}
