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
use p3_field::{Field, PrimeField32};
use recursion_circuit::system::{AggregationSubCircuit, VerifierConfig, VerifierTraceGen};
use tracing::instrument;

use crate::{
    bn254::CommitBytes,
    circuit::{
        deferral::aggregation::root::{
            bus::{DefVkCommitBus, IoCommitBus, OnionResultBus},
            decommit::MerkleDecommitAir,
            onion::OnionHashAir,
            verifier::DeferralRootPvsAir,
            DeferralIoCommit, DeferralRootTraceGen,
        },
        subair::{MerkleRootBus, MerkleTreeInternalBus, MerkleTreeSubAir},
    },
    prover::{trace_heights_tracing_info, Circuit},
    SC,
};

mod trace;

#[derive(derive_new::new, Clone)]
pub struct DeferralRootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    internal_recursive_dag_commit: CommitBytes,
}

impl<S: AggregationSubCircuit> Circuit for DeferralRootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();
        let io_commit_bus = IoCommitBus::new(next_bus_idx);
        let onion_res_bus = OnionResultBus::new(next_bus_idx + 1);
        let def_vk_commit_bus = DefVkCommitBus::new(next_bus_idx + 2);
        let merkle_root_bus = MerkleRootBus::new(next_bus_idx + 3);
        let merkle_tree_internal_bus = MerkleTreeInternalBus::new(next_bus_idx + 4);

        let verifier_pvs_air = DeferralRootPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            def_vk_commit_bus,
            merkle_root_bus,
            onion_res_bus,
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
        };

        let decommit_air = MerkleDecommitAir {
            subair: MerkleTreeSubAir::new(
                bus_inventory.poseidon2_compress_bus,
                merkle_root_bus,
                merkle_tree_internal_bus,
                0,
            ),
            io_commit_bus,
        };

        let onion_air = OnionHashAir {
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            def_vk_commit_bus,
            io_commit_bus,
            onion_res_bus,
        };

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain(self.verifier_circuit.airs())
            .chain([Arc::new(decommit_air) as AirRef<SC>])
            .chain([Arc::new(onion_air) as AirRef<SC>])
            .collect_vec()
    }
}

pub struct DeferralRootProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T: DeferralRootTraceGen<PB>,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    d_pk: DeviceMultiStarkProvingKey<PB>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<DeferralRootCircuit<S>>,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: DeferralRootTraceGen<PB>,
    > DeferralRootProver<PB, S, T>
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
        T: DeferralRootTraceGen<PB>,
    > DeferralRootProver<PB, S, T>
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
        let internal_recursive_dag_commit = child_vk_pcs_data.commitment.into();
        let circuit = Arc::new(DeferralRootCircuit::new(
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
        let internal_recursive_dag_commit = child_vk_pcs_data.commitment.into();
        let circuit = Arc::new(DeferralRootCircuit::new(
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

    pub fn get_circuit(&self) -> Arc<DeferralRootCircuit<S>> {
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
