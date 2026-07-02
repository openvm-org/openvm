use std::{marker::PhantomData, sync::Arc};

use eyre::Result;
use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
#[cfg(debug_assertions)]
use openvm_continuations::prover::debug_constraints;
use openvm_continuations::{
    circuit::{deferral::DeferralMerkleProofs, Circuit},
    prover::{DeferralCircuitProver, DeferralCircuitProverKey},
    CommitBytes, VkCommitBytes, SC,
};
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, VerifierConfig, VerifierSubCircuit, VerifierTraceGen,
};
use openvm_stark_backend::{
    codec::{Decode, Encode},
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{CommittedTraceData, DeviceDataTransporter, ProverBackend, ProverDevice},
    EngineDeviceCtx, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2CpuEngine, Digest, DIGEST_SIZE, EF, F,
};
use openvm_verify_stark_host::VmStarkProof;
use p3_field::{Field, PrimeField32};
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{DeferredVerifyCircuit, DeferredVerifyTraceGen, DeferredVerifyTraceGenImpl};

mod trace;

pub type DeferredVerifyCpuProver =
    DeferredVerifyProver<CpuBackend<SC>, VerifierSubCircuit<1>, DeferredVerifyTraceGenImpl>;
pub type DeferredVerifyCpuCircuitProver = DeferredVerifyCircuitProver<
    BabyBearPoseidon2CpuEngine,
    VerifierSubCircuit<1>,
    DeferredVerifyTraceGenImpl,
>;

#[cfg(feature = "cuda")]
pub type DeferredVerifyGpuProver =
    DeferredVerifyProver<GpuBackend, VerifierSubCircuit<1>, DeferredVerifyTraceGenImpl>;
#[cfg(feature = "cuda")]
pub type DeferredVerifyGpuCircuitProver = DeferredVerifyCircuitProver<
    BabyBearPoseidon2GpuEngine,
    VerifierSubCircuit<1>,
    DeferredVerifyTraceGenImpl,
>;

pub struct DeferredVerifyProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<DeferredVerifyCircuit<S>>,
}

impl<PB, S, T> DeferredVerifyProver<PB, S, T>
where
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    PB::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<PB::Val>>,
    ) -> Result<Proof<SC>>
    where
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: DeferredVerifyTraceGen<PB, EngineDeviceCtx<E>>,
    {
        assert!(
            deferral_merkle_proofs.is_none() || self.circuit.def_hook_commit.is_some(),
            "def_hook_commit must be defined to verify child proof with deferrals"
        );
        let engine = E::new(self.pk.params.clone());
        let ctx = self.generate_proving_ctx(
            proof,
            user_pvs_proof,
            deferral_merkle_proofs,
            engine.device().device_ctx(),
        );
        #[cfg(debug_assertions)]
        debug_constraints(&self.circuit, &ctx, &engine);
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
    ) -> Result<Proof<SC>>
    where
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: DeferredVerifyTraceGen<PB, EngineDeviceCtx<E>>,
    {
        self.prove::<E>(proof, user_pvs_proof, None)
    }

    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<CommitBytes>,
        def_idx: usize,
    ) -> Self
    where
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: DeferredVerifyTraceGen<PB, EngineDeviceCtx<E>>,
        E::PD: DeviceDataTransporter<SC, PB> + Clone,
        PB::Val: Field + PrimeField32,
        PB::Matrix: Clone,
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
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            vk_pre_hash: child_vk.pre_hash.into(),
        };
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(DeferredVerifyCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
            def_hook_commit,
            memory_dimensions,
            num_user_pvs,
            def_idx,
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

    #[allow(clippy::too_many_arguments)]
    pub fn from_pk<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<SC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<CommitBytes>,
        def_idx: usize,
    ) -> Self
    where
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: DeferredVerifyTraceGen<PB, EngineDeviceCtx<E>>,
        PB::Matrix: Clone,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                final_state_bus_enabled: true,
                has_cached: true,
            },
        );
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            vk_pre_hash: child_vk.pre_hash.into(),
        };
        let engine = E::new(pk.params.clone());
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        // WARNING: def_idx must match the original def_idx used when generating the pk,
        // or else the generated proof will be incorrect.
        let circuit = Arc::new(DeferredVerifyCircuit::new(
            Arc::new(verifier_circuit),
            internal_recursive_vk_commit,
            def_hook_commit,
            memory_dimensions,
            num_user_pvs,
            def_idx,
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

pub struct DeferredVerifyCircuitProver<
    E: StarkEngine<SC = SC>,
    S: AggregationSubCircuit + VerifierTraceGen<E::PB, SC, EngineDeviceCtx<E>>,
    T: DeferredVerifyTraceGen<E::PB, EngineDeviceCtx<E>>,
> {
    prover: DeferredVerifyProver<E::PB, S, T>,
    phantom: PhantomData<E>,
}

impl<E, S, T> DeferredVerifyCircuitProver<E, S, T>
where
    E: StarkEngine<SC = SC>,
    E::PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit + VerifierTraceGen<E::PB, SC, EngineDeviceCtx<E>>,
    T: DeferredVerifyTraceGen<E::PB, EngineDeviceCtx<E>>,
{
    pub fn new(prover: DeferredVerifyProver<E::PB, S, T>) -> Self {
        Self {
            prover,
            phantom: PhantomData,
        }
    }
}

impl<PB, S, T, E> DeferralCircuitProver<SC> for DeferredVerifyCircuitProver<E, S, T>
where
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit + VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
    T: DeferredVerifyTraceGen<PB, EngineDeviceCtx<E>>,
    E: StarkEngine<PB = PB, SC = SC>,
    PB::Matrix: Clone,
{
    fn from_pk(pk: DeferralCircuitProverKey<SC>) -> Self {
        let aux = DeferralVerifyProvingAux::decode(&mut pk.aux.as_slice())
            .expect("failed to decode verify-stark deferral proving aux");
        Self::new(DeferredVerifyProver::from_pk::<E>(
            aux.child_vk,
            aux.internal_recursive_cached_commit,
            pk.base_pk,
            aux.memory_dimensions,
            aux.num_user_pvs,
            aux.def_hook_commit,
            aux.def_idx,
        ))
    }

    fn get_pk(&self) -> Arc<DeferralCircuitProverKey<SC>> {
        let aux = DeferralVerifyProvingAux {
            child_vk: self.prover.child_vk.clone(),
            internal_recursive_cached_commit: self
                .prover
                .circuit
                .internal_recursive_vk_commit
                .cached_commit,
            memory_dimensions: self.prover.circuit.memory_dimensions,
            num_user_pvs: self.prover.circuit.num_user_pvs,
            def_hook_commit: self.prover.circuit.def_hook_commit,
            def_idx: self.prover.circuit.def_idx,
        };
        let mut encoded_aux = Vec::new();
        aux.encode(&mut encoded_aux)
            .expect("failed to encode verify-stark deferral proving aux");
        Arc::new(DeferralCircuitProverKey {
            base_pk: self.prover.get_pk(),
            aux: encoded_aux,
        })
    }

    fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.prover.get_vk()
    }

    fn prove(&self, input_bytes: &[u8]) -> Proof<SC> {
        let vm_proof = VmStarkProof::decode_from_bytes(input_bytes).unwrap();
        self.prover
            .prove::<E>(
                vm_proof.inner,
                &vm_proof.user_pvs_proof,
                vm_proof.deferral_merkle_proofs.as_ref(),
            )
            .expect("DeferredVerifyProver::prove failed")
    }

    fn get_def_idx(&self) -> usize {
        self.prover.circuit.def_idx
    }

    fn cached_commits(&self) -> Vec<CommitBytes> {
        vec![self.prover.get_cached_commit().into()]
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct DeferralVerifyProvingAux {
    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    internal_recursive_cached_commit: CommitBytes,
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
    def_hook_commit: Option<CommitBytes>,
    def_idx: usize,
}

impl Encode for DeferralVerifyProvingAux {
    fn encode<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        let bytes = bitcode::serialize(self).map_err(std::io::Error::other)?;
        std::io::Write::write_all(writer, &(bytes.len() as u64).to_le_bytes())?;
        std::io::Write::write_all(writer, &bytes)
    }
}

impl Decode for DeferralVerifyProvingAux {
    fn decode<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut len_bytes = [0u8; 8];
        std::io::Read::read_exact(reader, &mut len_bytes)?;
        let len = u64::from_le_bytes(len_bytes) as usize;
        let mut bytes = vec![0u8; len];
        std::io::Read::read_exact(reader, &mut bytes)?;
        bitcode::deserialize(&bytes).map_err(std::io::Error::other)
    }
}
