use std::borrow::Borrow;

use itertools::Itertools;
use openvm_circuit::{
    arch::POSEIDON2_WIDTH,
    system::memory::{dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof},
};
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{
    data_transporter::transport_air_proving_ctx_to_device, BabyBearBn254Poseidon2HashScheme,
    GenericGpuBackend,
};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
    StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use openvm_verify_stark_host::pvs::{DeferralPvs, DEF_PVS_AIR_ID};
use p3_field::PrimeField32;

use crate::circuit::{
    deferral::verify::DeferralMerkleProofs,
    root::{commit, memory},
    SingleAirTraceData, SubCircuitTraceData,
};

// Trait that root provers use to remain generic in PB. Tracegen returns the AIR proving
// contexts, Poseidon2 compress inputs, and Poseidon2 permute inputs to be fed to Poseidon2Air.
pub trait RootTraceGen<PB: ProverBackend> {
    fn new(deferral_enabled: bool) -> Self;
    fn generate_pre_verifier_subcircuit_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        memory_dimensions: MemoryDimensions,
    ) -> SubCircuitTraceData<PB>;
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
    ) -> SubCircuitTraceData<CpuBackend<SC>> {
        let SingleAirTraceData {
            air_proving_ctx: verifier_ctx,
            poseidon2_compress_inputs: verifier_compress_inputs,
            poseidon2_permute_inputs: verifier_permute_inputs,
        } = super::verifier::generate_proving_ctx(proof, self.deferral_enabled);
        let (commit_ctx, commit_inputs) =
            commit::generate_proving_ctx(user_pvs_proof.public_values.clone());
        let (memory_ctx, memory_inputs) = memory::generate_proving_input(
            user_pvs_proof.public_values_commit,
            &user_pvs_proof.proof,
            memory_dimensions,
            user_pvs_proof.public_values.len(),
        );
        SubCircuitTraceData {
            air_proving_ctxs: vec![verifier_ctx, commit_ctx, memory_ctx],
            poseidon2_compress_inputs: verifier_compress_inputs
                .into_iter()
                .chain(commit_inputs)
                .chain(memory_inputs)
                .collect_vec(),
            poseidon2_permute_inputs: verifier_permute_inputs,
        }
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
impl RootTraceGen<GenericGpuBackend<BabyBearBn254Poseidon2HashScheme>> for RootTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctx(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
    ) -> SubCircuitTraceData<GenericGpuBackend<BabyBearBn254Poseidon2HashScheme>> {
        let data =
            self.generate_pre_verifier_subcircuit_ctx(proof, user_pvs_proof, memory_dimensions);
        SubCircuitTraceData {
            air_proving_ctxs: data
                .air_proving_ctxs
                .into_iter()
                .map(transport_air_proving_ctx_to_device::<BabyBearBn254Poseidon2HashScheme>)
                .collect_vec(),
            poseidon2_compress_inputs: data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: data.poseidon2_permute_inputs,
        }
    }

    fn generate_other_proving_ctxs(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> (
        Vec<AirProvingContext<GenericGpuBackend<BabyBearBn254Poseidon2HashScheme>>>,
        Vec<[F; POSEIDON2_WIDTH]>,
    ) {
        let (cpu_ctxs, inputs) =
            self.generate_other_proving_ctxs(proof, memory_dimensions, deferral_merkle_proofs);
        let gpu_ctxs = cpu_ctxs
            .into_iter()
            .map(transport_air_proving_ctx_to_device::<BabyBearBn254Poseidon2HashScheme>)
            .collect_vec();
        (gpu_ctxs, inputs)
    }
}
