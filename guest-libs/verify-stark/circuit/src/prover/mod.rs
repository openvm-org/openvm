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
#[cfg(feature = "cuda")]
use openvm_recursion_circuit::system::GpuVerifierTraceGen;
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, VerifierConfig, VerifierSubCircuit, VerifierTraceGen,
};
#[cfg(feature = "cuda")]
use openvm_stark_backend::prover::ProverDevice;
#[cfg(feature = "cuda")]
use openvm_stark_backend::EngineDeviceCtx;
use openvm_stark_backend::{
    codec::Decode,
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{CommittedTraceData, DeviceDataTransporter, ProverBackend},
    StarkEngine, SystemParams,
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

impl<S, T> DeferredVerifyProver<CpuBackend<SC>, S, T>
where
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<SC>, SC>,
    T: DeferredVerifyTraceGen<CpuBackend<SC>>,
    <CpuBackend<SC> as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn prove<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> Result<Proof<SC>> {
        let engine = E::new(self.pk.params.clone());
        let ctx = self.generate_proving_ctx(proof, user_pvs_proof, deferral_merkle_proofs);
        #[cfg(debug_assertions)]
        debug_constraints(&self.circuit, &ctx, &engine);
        let d_pk = engine.device().transport_pk_to_device(self.pk.as_ref());
        let proof = engine.prove(&d_pk, ctx)?;
        #[cfg(debug_assertions)]
        engine.verify(&self.vk, &proof)?;
        Ok(proof)
    }

    #[instrument(name = "total_proof", skip_all)]
    pub fn prove_no_def<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
    ) -> Result<Proof<SC>> {
        self.prove::<E>(proof, user_pvs_proof, None)
    }
    pub fn new<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<<CpuBackend<SC> as ProverBackend>::Commitment>,
        def_idx: usize,
    ) -> Self
    where
        E::PD: DeviceDataTransporter<SC, CpuBackend<SC>> + Clone,
        <CpuBackend<SC> as ProverBackend>::Val: Field + PrimeField32,
        <CpuBackend<SC> as ProverBackend>::Commitment: Into<CommitBytes>,
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
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_pk<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<SC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<<CpuBackend<SC> as ProverBackend>::Commitment>,
        def_idx: usize,
    ) -> Self
    where
        E::PD: DeviceDataTransporter<SC, CpuBackend<SC>> + Clone,
        <CpuBackend<SC> as ProverBackend>::Val: Field + PrimeField32,
        <CpuBackend<SC> as ProverBackend>::Commitment: Into<CommitBytes>,
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
        }
    }
}

#[cfg(feature = "cuda")]
impl<S, T> DeferredVerifyProver<GpuBackend, S, T>
where
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<SC>, SC>
        + GpuVerifierTraceGen<GpuBackend, SC>,
    T: DeferredVerifyTraceGen<CpuBackend<SC>>,
    <GpuBackend as ProverBackend>::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn prove<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> Result<Proof<SC>>
    where
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
    {
        let engine = E::new(self.pk.params.clone());
        let device_ctx: openvm_cuda_common::stream::GpuDeviceCtx =
            engine.device().device_ctx().clone().into();
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
    pub fn prove_no_def<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
    ) -> Result<Proof<SC>>
    where
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
    {
        self.prove::<E>(proof, user_pvs_proof, None)
    }

    pub fn new<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        system_params: SystemParams,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<<GpuBackend as ProverBackend>::Commitment>,
        def_idx: usize,
    ) -> Self
    where
        E::PD: DeviceDataTransporter<SC, GpuBackend> + Clone,
        <GpuBackend as ProverBackend>::Val: Field + PrimeField32,
        <GpuBackend as ProverBackend>::Commitment: Into<CommitBytes>,
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
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
        let child_vk_pcs_data = verifier_circuit.commit_child_vk_gpu(&engine, &child_vk);
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
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_pk<E: StarkEngine<SC = SC, PB = GpuBackend>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        internal_recursive_cached_commit: CommitBytes,
        pk: Arc<MultiStarkProvingKey<SC>>,
        memory_dimensions: MemoryDimensions,
        num_user_pvs: usize,
        def_hook_commit: Option<<GpuBackend as ProverBackend>::Commitment>,
        def_idx: usize,
    ) -> Self
    where
        E::PD: DeviceDataTransporter<SC, GpuBackend> + Clone,
        <GpuBackend as ProverBackend>::Val: Field + PrimeField32,
        <GpuBackend as ProverBackend>::Commitment: Into<CommitBytes>,
        EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
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
        let child_vk_pcs_data = verifier_circuit.commit_child_vk_gpu(&engine, &child_vk);
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
}

impl<PB, S, T> DeferredVerifyProver<PB, S, T>
where
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    PB::Matrix: Clone,
{
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

pub struct DeferredVerifyCircuitProver<E: StarkEngine<SC = SC>, S: AggregationSubCircuit, T> {
    prover: DeferredVerifyProver<E::PB, S, T>,
    phantom: PhantomData<E>,
}

trait DeferredVerifyCircuitBackend<E, S, T>
where
    E: StarkEngine<SC = SC>,
    E::PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
{
    fn prove_no_def(
        prover: &DeferredVerifyProver<E::PB, S, T>,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
    ) -> Result<Proof<SC>>;
}

impl<E, S, T> DeferredVerifyCircuitBackend<E, S, T> for CpuBackend<SC>
where
    E: StarkEngine<PB = CpuBackend<SC>, SC = SC>,
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<SC>, SC>,
    T: DeferredVerifyTraceGen<CpuBackend<SC>>,
    <CpuBackend<SC> as ProverBackend>::Matrix: Clone,
{
    fn prove_no_def(
        prover: &DeferredVerifyProver<CpuBackend<SC>, S, T>,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
    ) -> Result<Proof<SC>> {
        prover.prove_no_def::<E>(proof, user_pvs_proof)
    }
}

#[cfg(feature = "cuda")]
impl<E, S, T> DeferredVerifyCircuitBackend<E, S, T> for GpuBackend
where
    E: StarkEngine<PB = GpuBackend, SC = SC>,
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<SC>, SC>
        + GpuVerifierTraceGen<GpuBackend, SC>,
    T: DeferredVerifyTraceGen<CpuBackend<SC>>,
    <GpuBackend as ProverBackend>::Matrix: Clone,
    EngineDeviceCtx<E>: Into<openvm_cuda_common::stream::GpuDeviceCtx>,
{
    fn prove_no_def(
        prover: &DeferredVerifyProver<GpuBackend, S, T>,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
    ) -> Result<Proof<SC>> {
        prover.prove_no_def::<E>(proof, user_pvs_proof)
    }
}

impl<E, S, T> DeferredVerifyCircuitProver<E, S, T>
where
    E: StarkEngine<SC = SC>,
    E::PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
{
    pub fn new(prover: DeferredVerifyProver<E::PB, S, T>) -> Self {
        Self {
            prover,
            phantom: PhantomData,
        }
    }
}

impl<S, T, E> DeferralCircuitProver<SC> for DeferredVerifyCircuitProver<E, S, T>
where
    E: StarkEngine<SC = SC>,
    E::PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>
        + DeferredVerifyCircuitBackend<E, S, T>,
    <E::PB as ProverBackend>::Matrix: Clone,
    S: AggregationSubCircuit,
{
    fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.prover.get_vk()
    }

    fn prove(&self, input_bytes: &[u8]) -> Proof<SC> {
        let vm_proof = VmStarkProof::decode_from_bytes(input_bytes).unwrap();
        <E::PB as DeferredVerifyCircuitBackend<E, S, T>>::prove_no_def(
            &self.prover,
            vm_proof.inner,
            &vm_proof.user_pvs_proof,
        )
        .expect("DeferredVerifyProver::prove_no_def failed")
    }

    fn get_def_idx(&self) -> usize {
        self.prover.circuit.def_idx
    }
}
