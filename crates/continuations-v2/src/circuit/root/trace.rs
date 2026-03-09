use std::borrow::Borrow;

use itertools::Itertools;
use openvm_circuit::{
    arch::POSEIDON2_WIDTH,
    system::memory::{dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof},
};
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ProverBackend},
    StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeField32;
use verify_stark::pvs::{DeferralPvs, DEF_PVS_AIR_ID};

use crate::circuit::{
    deferral::verify::DeferralMerkleProofs,
    root::{commit, memory},
};

// Trait that root provers use to remain generic in PB. Tracegen returns both the AIR proving
// contexts and the Poseidon2 compress inputs that are to be fed to Poseidon2Air.
pub trait RootTraceGen<PB: ProverBackend> {
    fn new(deferral_enabled: bool) -> Self;
    fn generate_pre_verifier_subcircuit_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        memory_dimensions: MemoryDimensions,
    ) -> (Vec<AirProvingContext<PB>>, Vec<[PB::Val; POSEIDON2_WIDTH]>);
    fn generate_other_proving_ctxs(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<PB::Val>>,
    ) -> (Vec<AirProvingContext<PB>>, Vec<[PB::Val; POSEIDON2_WIDTH]>);
}

pub struct RootTraceGenImpl {
    pub deferral_enabled: bool,
}

impl<SC: StarkProtocolConfig<F = F>> RootTraceGen<CpuBackend<SC>> for RootTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
    ) -> (
        Vec<AirProvingContext<CpuBackend<SC>>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (verifier_ctx, verifier_p2_inputs) =
            super::verifier::generate_proving_ctx(proof, self.deferral_enabled);
        let (commit_ctx, commit_inputs) =
            commit::generate_proving_ctx(user_pvs_proof.public_values.clone());
        let (memory_ctx, memory_inputs) = memory::generate_proving_input(
            user_pvs_proof.public_values_commit,
            &user_pvs_proof.proof,
            memory_dimensions,
            user_pvs_proof.public_values.len(),
        );
        (
            vec![verifier_ctx, commit_ctx, memory_ctx],
            verifier_p2_inputs
                .into_iter()
                .chain(commit_inputs)
                .chain(memory_inputs)
                .collect_vec(),
        )
    }

    fn generate_other_proving_ctxs(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> (
        Vec<AirProvingContext<CpuBackend<SC>>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (paths_ctx, paths_inputs) = if let Some(deferral_merkle_proofs) = deferral_merkle_proofs
        {
            assert!(self.deferral_enabled);
            let def_pvs: &DeferralPvs<F> = proof.public_values[DEF_PVS_AIR_ID].as_slice().borrow();
            let depth = def_pvs.depth.as_canonical_u32() as usize;
            let (ctx, inputs) = super::def_paths::generate_proving_input(
                def_pvs.initial_acc_hash,
                def_pvs.final_acc_hash,
                &deferral_merkle_proofs.initial_merkle_proof,
                &deferral_merkle_proofs.final_merkle_proof,
                memory_dimensions,
                depth,
                depth == 0,
            );
            (Some(ctx), inputs)
        } else {
            assert!(!self.deferral_enabled);
            (None, vec![])
        };
        (paths_ctx.into_iter().collect_vec(), paths_inputs)
    }
}

#[cfg(feature = "cuda")]
impl RootTraceGen<GpuBackend> for RootTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
    ) -> (
        Vec<AirProvingContext<GpuBackend>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (cpu_ctxs, inputs) =
            self.generate_pre_verifier_subcircuit_ctx(proof, user_pvs_proof, memory_dimensions);
        let gpu_ctxs = cpu_ctxs
            .into_iter()
            .map(transport_air_proving_ctx_to_device)
            .collect_vec();
        (gpu_ctxs, inputs)
    }

    fn generate_other_proving_ctxs(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> (
        Vec<AirProvingContext<GpuBackend>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (cpu_ctxs, inputs) =
            self.generate_other_proving_ctxs(proof, memory_dimensions, deferral_merkle_proofs);
        let gpu_ctxs = cpu_ctxs
            .into_iter()
            .map(transport_air_proving_ctx_to_device)
            .collect_vec();
        (gpu_ctxs, inputs)
    }
}
