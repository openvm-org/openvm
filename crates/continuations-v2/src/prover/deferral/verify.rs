use std::{iter::once, sync::Arc};

use eyre::Result;
use itertools::Itertools;
use openvm_circuit::{
    arch::POSEIDON2_WIDTH,
    system::memory::{dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof},
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
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32};
use recursion_circuit::system::{
    AggregationSubCircuit, CachedTraceCtx, VerifierConfig, VerifierExternalData, VerifierTraceGen,
};
use tracing::instrument;

use crate::{
    bn254::CommitBytes,
    circuit::{
        deferral::verify::{
            bus::{OutputCommitBus, OutputValBus},
            output::DeferralOutputCommitAir,
            verifier::DeferredVerifyPvsAir,
            DeferredVerifyTraceGen, PreVerifierData,
        },
        root::bus::{MemoryMerkleCommitBus, UserPvsCommitBus, UserPvsCommitTreeBus},
        user_pvs::{commit::UserPvsCommitAir, memory::UserPvsInMemoryAir},
    },
    prover::{trace_heights_tracing_info, Circuit},
    SC,
};

#[derive(derive_new::new, Clone)]
pub struct DeferredVerifyCircuit<S: AggregationSubCircuit> {
    pub verifier_circuit: Arc<S>,
    internal_recursive_dag_commit: CommitBytes,
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
}

impl<S: AggregationSubCircuit> Circuit for DeferredVerifyCircuit<S> {
    fn airs(&self) -> Vec<AirRef<SC>> {
        let bus_inventory = self.verifier_circuit.bus_inventory();
        let next_bus_idx = self.verifier_circuit.next_bus_idx();

        let user_pvs_commit_bus = UserPvsCommitBus::new(next_bus_idx);
        let user_pvs_commit_tree_bus = UserPvsCommitTreeBus::new(next_bus_idx + 1);
        let memory_merkle_commit_bus = MemoryMerkleCommitBus::new(next_bus_idx + 2);
        let output_val_bus = OutputValBus::new(next_bus_idx + 3);
        let output_commit_bus = OutputCommitBus::new(next_bus_idx + 4);

        let verifier_pvs_air = DeferredVerifyPvsAir {
            public_values_bus: bus_inventory.public_values_bus,
            cached_commit_bus: bus_inventory.cached_commit_bus,
            poseidon2_compress_bus: bus_inventory.poseidon2_compress_bus,
            memory_merkle_commit_bus,
            output_val_bus,
            output_commit_bus,
            final_state_bus: bus_inventory.final_state_bus,
            expected_internal_recursive_dag_commit: self.internal_recursive_dag_commit,
        };
        let user_pvs_commit_air = UserPvsCommitAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            user_pvs_commit_tree_bus,
            Some(output_val_bus),
            self.num_user_pvs,
        );
        let user_pvs_memory_air = UserPvsInMemoryAir::new(
            bus_inventory.poseidon2_compress_bus,
            user_pvs_commit_bus,
            memory_merkle_commit_bus,
            self.memory_dimensions,
            self.num_user_pvs,
        );
        let output_commit_air = DeferralOutputCommitAir {
            poseidon2_bus: bus_inventory.poseidon2_compress_bus,
            range_bus: bus_inventory.range_checker_bus,
            output_val_bus,
            output_commit_bus,
        };

        [Arc::new(verifier_pvs_air) as AirRef<SC>]
            .into_iter()
            .chain(self.verifier_circuit.airs())
            .chain([Arc::new(user_pvs_commit_air) as AirRef<SC>])
            .chain([Arc::new(user_pvs_memory_air) as AirRef<SC>])
            .chain([Arc::new(output_commit_air) as AirRef<SC>])
            .collect_vec()
    }
}

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
        S: AggregationSubCircuit + VerifierTraceGen<PB>,
        T: DeferredVerifyTraceGen<PB>,
    > DeferredVerifyProver<PB, S, T>
where
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> ProvingContext<PB> {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        let PreVerifierData {
            proving_ctxs: agg_other_ctxs,
            poseidon2_inputs,
            range_inputs,
            verifier_pvs_record,
            output_commit,
        } = self.agg_node_tracegen.pre_verifier_subcircuit_tracegen(
            &proof,
            user_pvs_proof,
            self.circuit.memory_dimensions,
        );

        let mut final_transcript_state = [F::ZERO; POSEIDON2_WIDTH];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_inputs,
            range_check_inputs: &range_inputs,
            required_heights: None,
            final_transcript_state: Some(&mut final_transcript_state),
        };

        let proof_slice = &[proof];
        let cached_trace_ctx = CachedTraceCtx::PcsData(self.child_vk_pcs_data.clone());
        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs(
                &self.child_vk,
                cached_trace_ctx,
                proof_slice,
                &mut external_data,
                default_duplex_sponge_recorder(),
            )
            .unwrap();

        let verifier_pvs_ctx = self.agg_node_tracegen.generate_verifier_pvs_ctx(
            &proof_slice[0],
            verifier_pvs_record,
            final_transcript_state,
            output_commit,
        );

        ProvingContext {
            per_trace: once(verifier_pvs_ctx)
                .chain(subcircuit_ctxs)
                .chain(agg_other_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }

    #[instrument(name = "total_proof", skip_all)]
    pub fn prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
    ) -> Result<Proof<SC>> {
        let ctx = self.generate_proving_ctx(proof, user_pvs_proof);
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info(&ctx.per_trace, &self.circuit.airs());
        }
        let engine = E::new(self.pk.params.clone());
        #[cfg(debug_assertions)]
        crate::prover::debug_constraints(&self.circuit, &ctx, &engine);
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
        T: DeferredVerifyTraceGen<PB>,
    > DeferredVerifyProver<PB, S, T>
{
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        child_vk_pcs_data: CommittedTraceData<PB>,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
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
                has_cached: true,
            },
        );
        let engine = E::new(system_params);
        let internal_recursive_dag_commit = child_vk_pcs_data.commitment.into();
        let circuit = Arc::new(DeferredVerifyCircuit::new(
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
        }
    }

    pub fn from_pk(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        child_vk_pcs_data: CommittedTraceData<PB>,
        pk: Arc<MultiStarkProvingKey<SC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
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
                has_cached: true,
            },
        );
        let internal_recursive_dag_commit = child_vk_pcs_data.commitment.into();
        let circuit = Arc::new(DeferredVerifyCircuit::new(
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
