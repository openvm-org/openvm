use std::{iter::once, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{CommittedTraceData, DeviceDataTransporter, ProverBackend, ProvingContext},
    AirRef, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, DIGEST_SIZE, EF, F,
};
use p3_field::{Field, PrimeField32};
use recursion_circuit::system::{AggregationSubCircuit, VerifierTraceGen};
use tracing::instrument;

use crate::{
    aggregation::{trace_heights_tracing_info, Circuit},
    bn254::CommitBytes,
    circuit::root::{
        bus::{MemoryMerkleCommitBus, UserPvsCommitBus, UserPvsCommitTreeBus},
        commit::UserPvsCommitAir,
        memory::UserPvsInMemoryAir,
        verifier::RootVerifierPvsAir,
        RootTraceGen,
    },
    SC,
};

#[derive(derive_new::new, Clone)]
pub struct RootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    internal_recursive_dag_commit: CommitBytes,
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
}

impl<S: AggregationSubCircuit> Circuit for RootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();

        let user_pvs_commit_bus = UserPvsCommitBus::new(next_bus_idx);
        let user_pvs_commit_tree_bus = UserPvsCommitTreeBus::new(next_bus_idx + 1);
        let memory_merkle_commit_bus = MemoryMerkleCommitBus::new(next_bus_idx + 2);

        let verifier_pvs_air = RootVerifierPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            memory_merkle_commit_bus,
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
        };
        let user_pvs_commit_air = UserPvsCommitAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            user_pvs_commit_tree_bus,
            self.num_user_pvs,
        );
        let user_pvs_memory_air = UserPvsInMemoryAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            memory_merkle_commit_bus,
            self.memory_dimensions,
            self.num_user_pvs,
        );

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain(self.verifier_circuit.airs())
            .chain([Arc::new(user_pvs_commit_air) as AirRef<SC>])
            .chain([Arc::new(user_pvs_memory_air) as AirRef<SC>])
            .collect_vec()
    }
}

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
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> Option<ProvingContext<PB>> {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        // These AIRs should have the same height regardless of proof or user_pvs_proof
        let (verifier_pvs_ctx, mut poseidon2_inputs) =
            self.agg_node_tracegen.generate_verifier_pvs_ctx(&proof);
        let (agg_other_ctxs, other_inputs) = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(user_pvs_proof, self.circuit.memory_dimensions);
        poseidon2_inputs.extend(other_inputs);

        let verifier_trace_heights = self.trace_heights.as_ref().map(|v| {
            let num_airs = v.len();
            &v[1..(num_airs - agg_other_ctxs.len())]
        });

        let subcircuit_ctxs = self.circuit.verifier_circuit.generate_proving_ctxs(
            &self.child_vk,
            self.child_vk_pcs_data.clone(),
            &[proof],
            &poseidon2_inputs,
            verifier_trace_heights,
            default_duplex_sponge_recorder(),
        );

        subcircuit_ctxs.map(|subcircuit_ctxs| ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(agg_other_ctxs)
                .enumerate()
                .collect_vec(),
        })
    }

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
        crate::aggregation::debug_constraints(&self.circuit, &ctx, &engine);
        let d_pk = engine.device().transport_pk_to_device(self.pk.as_ref());
        let proof = engine.prove(&d_pk, ctx);
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
        let verifier_circuit = S::new(child_vk.clone(), true, true);
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
        let verifier_circuit = S::new(child_vk.clone(), true, true);
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
