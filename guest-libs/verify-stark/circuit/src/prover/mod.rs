use std::{marker::PhantomData, sync::Arc};

use eyre::Result;
use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
use openvm_continuations::{
    circuit::{deferral::DeferralMerkleProofs, Circuit},
    prover::{debug_constraints, DeferralCircuitProver},
    CommitBytes, VkCommitBytes, SC,
};
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, VerifierConfig, VerifierSubCircuit, VerifierTraceGen,
};
use openvm_stark_backend::{
    codec::Decode,
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
pub type DeferredVerifyGpuProver = DeferredVerifyProver<
    GpuBackend,
    VerifierSubCircuit<1>,
    DeferredVerifyTraceGenImpl,
    openvm_cuda_common::stream::GpuDeviceCtx,
>;
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
    DC: Clone + Send + Sync = (),
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: CommittedTraceData<PB>,
    circuit: Arc<DeferredVerifyCircuit<S>>,
    _phantom_dc: PhantomData<DC>,
}

impl<
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
        S: AggregationSubCircuit + VerifierTraceGen<PB, SC, DC>,
        T: DeferredVerifyTraceGen<PB, DC>,
        DC: Clone + Send + Sync,
    > DeferredVerifyProver<PB, S, T, DC>
where
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
        DC: From<EngineDeviceCtx<E>>,
    {
        let engine = E::new(self.pk.params.clone());
        let device_ctx: DC = engine.device().device_ctx().clone().into();
        let ctx =
            self.generate_proving_ctx(proof, user_pvs_proof, deferral_merkle_proofs, &device_ctx);
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
        DC: From<EngineDeviceCtx<E>>,
    {
        self.prove::<E>(proof, user_pvs_proof, None)
    }
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<PB::Commitment>,
        def_idx: usize,
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
        let internal_recursive_vk_commit = VkCommitBytes {
            cached_commit: internal_recursive_cached_commit,
            vk_pre_hash: child_vk.pre_hash.into(),
        };
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let def_hook_commit = def_hook_commit.map(Into::into);
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
            _phantom_dc: PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_pk<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<SC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<PB::Commitment>,
        def_idx: usize,
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
        let def_hook_commit = def_hook_commit.map(Into::into);
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
            _phantom_dc: PhantomData,
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
    prover: DeferredVerifyProver<E::PB, S, T, EngineDeviceCtx<E>>,
    phantom: PhantomData<E>,
}

impl<E, S, T> DeferredVerifyCircuitProver<E, S, T>
where
    E: StarkEngine<SC = SC>,
    E::PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit + VerifierTraceGen<E::PB, SC, EngineDeviceCtx<E>>,
    T: DeferredVerifyTraceGen<E::PB, EngineDeviceCtx<E>>,
{
    pub fn new(prover: DeferredVerifyProver<E::PB, S, T, EngineDeviceCtx<E>>) -> Self {
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
    EngineDeviceCtx<E>: From<EngineDeviceCtx<E>>,
{
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.prover.get_vk()
    }

    fn prove(&self, input_bytes: &[u8]) -> Proof<SC> {
        let vm_proof = VmStarkProof::decode_from_bytes(input_bytes).unwrap();
        self.prover
            .prove_no_def::<E>(vm_proof.inner, &vm_proof.user_pvs_proof)
            .expect("DeferredVerifyProver::prove_no_def failed")
    }

    fn get_def_idx(&self) -> usize {
        self.prover.circuit.def_idx
    }
}
