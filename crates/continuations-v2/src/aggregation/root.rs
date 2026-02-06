use std::{iter::once, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
use openvm_stark_backend::AirRef;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_field::{Field, PrimeField32};
use recursion_circuit::system::{AggregationSubCircuit, VerifierTraceGen};
use stark_backend_v2::{
    DIGEST_SIZE, StarkWhirEngine, SystemParams,
    keygen::types::{MultiStarkProvingKeyV2, MultiStarkVerifyingKeyV2},
    poseidon2::sponge::DuplexSpongeRecorder,
    proof::Proof,
    prover::{CommittedTraceDataV2, DeviceDataTransporterV2, ProverBackendV2, ProvingContextV2},
};
use tracing::instrument;

use crate::{
    aggregation::{Circuit, trace_heights_tracing_info},
    bn254::CommitBytes,
    circuit::root::{
        RootTraceGen,
        bus::{MemoryMerkleCommitBus, UserPvsCommitBus, UserPvsCommitTreeBus},
        commit::UserPvsCommitAir,
        memory::UserPvsInMemoryAir,
        verifier::RootVerifierPvsAir,
    },
};

#[derive(derive_new::new, Clone)]
pub struct RootCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    internal_recursive_vk_commit: CommitBytes,
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
}

impl<S: AggregationSubCircuit> Circuit for RootCircuit<S> {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();

        let user_pvs_commit_bus = UserPvsCommitBus::new(next_bus_idx);
        let user_pvs_commit_tree_bus = UserPvsCommitTreeBus::new(next_bus_idx + 1);
        let memory_merkle_commit_bus = MemoryMerkleCommitBus::new(next_bus_idx + 2);

        let verifier_pvs_air = RootVerifierPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            memory_merkle_commit_bus,
            expected_internal_recursive_vk_commit: self.internal_recursive_vk_commit.clone().into(),
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

        [Arc::new(verifier_pvs_air) as AirRef<BabyBearPoseidon2Config>]
            .into_iter()
            .chain(self.verifier_circuit.airs())
            .chain([Arc::new(user_pvs_commit_air) as AirRef<BabyBearPoseidon2Config>])
            .chain([Arc::new(user_pvs_memory_air) as AirRef<BabyBearPoseidon2Config>])
            .collect_vec()
    }
}

pub struct RootProver<PB: ProverBackendV2, S: AggregationSubCircuit, T: RootTraceGen<PB>> {
    pk: Arc<MultiStarkProvingKeyV2>,
    vk: Arc<MultiStarkVerifyingKeyV2>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKeyV2>,
    child_vk_pcs_data: CommittedTraceDataV2<PB>,
    circuit: Arc<RootCircuit<S>>,
}

impl<PB: ProverBackendV2, S: AggregationSubCircuit + VerifierTraceGen<PB>, T: RootTraceGen<PB>>
    RootProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> ProvingContextV2<PB> {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        let (verifier_pvs_ctx, mut poseidon2_inputs) =
            self.agg_node_tracegen.generate_verifier_pvs_ctx(&proof);
        let (agg_other_ctxs, other_inputs) = self
            .agg_node_tracegen
            .generate_other_proving_ctxs(user_pvs_proof, self.circuit.memory_dimensions);
        poseidon2_inputs.extend(other_inputs);

        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs::<DuplexSpongeRecorder>(
                &self.child_vk,
                self.child_vk_pcs_data.clone(),
                &[proof],
                &poseidon2_inputs,
            );

        ProvingContextV2 {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(agg_other_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }

    #[instrument(name = "total_proof", skip_all)]
    pub fn root_prove<E: StarkWhirEngine<PB = PB>>(
        &self,
        proof: Proof,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> Result<Proof> {
        let ctx = self.generate_proving_ctx(proof, user_pvs_proof);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        crate::aggregation::debug_constraints(&self.circuit, &ctx.per_trace, &engine);
        let d_pk = engine.device().transport_pk_to_device(self.pk.as_ref());
        let proof = engine.prove(&d_pk, ctx);
        #[cfg(debug_assertions)]
        engine.verify(&self.vk, &proof)?;
        Ok(proof)
    }
}

impl<PB: ProverBackendV2, S: AggregationSubCircuit + VerifierTraceGen<PB>, T: RootTraceGen<PB>>
    RootProver<PB, S, T>
{
    pub fn new<E: StarkWhirEngine<PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKeyV2>,
        child_vk_pcs_data: CommittedTraceDataV2<PB>,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
    ) -> Self
    where
        E::PD: DeviceDataTransporterV2<PB> + Clone,
        PB::Val: Field + PrimeField32,
        PB::Matrix: Clone,
        PB::Commitment: Into<CommitBytes>,
    {
        let verifier_circuit = S::new(child_vk.clone(), true, true);
        let engine = E::new(system_params);
        let internal_recursive_vk_commit = child_vk_pcs_data.commitment.clone().into();
        let circuit = Arc::new(RootCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
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
        }
    }

    pub fn from_pk(
        child_vk: Arc<MultiStarkVerifyingKeyV2>,
        child_vk_pcs_data: CommittedTraceDataV2<PB>,
        pk: Arc<MultiStarkProvingKeyV2>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
    ) -> Self
    where
        PB::Val: Field + PrimeField32,
        PB::Matrix: Clone,
        PB::Commitment: Into<CommitBytes>,
    {
        let verifier_circuit = S::new(child_vk.clone(), true, true);
        let internal_recursive_vk_commit = child_vk_pcs_data.commitment.clone().into();
        let circuit = Arc::new(RootCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
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
        }
    }

    pub fn get_circuit(&self) -> Arc<RootCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKeyV2> {
        self.pk.clone()
    }

    pub fn get_vk(&self) -> Arc<MultiStarkVerifyingKeyV2> {
        self.vk.clone()
    }

    pub fn get_cached_commit(&self) -> <PB as ProverBackendV2>::Commitment {
        self.child_vk_pcs_data.commitment.clone()
    }
}
