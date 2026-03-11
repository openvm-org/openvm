use std::{borrow::Borrow, iter::once};

use itertools::Itertools;
use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::prelude::{DIGEST_SIZE, F, SC};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
};
use p3_field::PrimeField32;
use verify_stark::pvs::{DeferralPvs, DEF_PVS_AIR_ID};

use crate::circuit::{
    deferral::verify::{
        output::DeferralOutputCtx,
        verifier::{generate_record, DeferredVerifyPvsRecord},
    },
    root::{def_paths, memory},
};

pub struct PreVerifierData<PB: ProverBackend> {
    pub pre_verifier_ctxs: [AirProvingContext<PB>; 2],
    pub post_verifier_ctxs: Vec<AirProvingContext<PB>>,
    pub poseidon2_compress_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub poseidon2_permute_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub range_inputs: Vec<usize>,
    pub verifier_pvs_record: DeferredVerifyPvsRecord<PB::Val>,
    pub output_commit: [PB::Val; DIGEST_SIZE],
}

pub struct DeferralMerkleProofs<T> {
    pub initial_merkle_proof: Vec<[T; DIGEST_SIZE]>,
    pub final_merkle_proof: Vec<[T; DIGEST_SIZE]>,
}

// Trait used to remain generic in PB
pub trait DeferredVerifyTraceGen<PB: ProverBackend> {
    fn new(deferral_enabled: bool) -> Self;

    // Returns the AIR proving contexts, Poseidon2 and range inputs, and the data
    // needed to compute the DeferredVerifyPvsAir trace later
    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> PreVerifierData<PB>;

    fn generate_verifier_pvs_ctx(
        &self,
        proof: &Proof<SC>,
        record: DeferredVerifyPvsRecord<PB::Val>,
        final_transcript_state: [PB::Val; POSEIDON2_WIDTH],
        output_commit: [PB::Val; DIGEST_SIZE],
    ) -> AirProvingContext<PB>;
}

pub struct DeferredVerifyTraceGenImpl {
    pub deferral_enabled: bool,
}

impl DeferredVerifyTraceGen<CpuBackend<SC>> for DeferredVerifyTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> PreVerifierData<CpuBackend<SC>> {
        let (verifier_pvs_record, verifier_p2_compress_inputs, verifier_p2_permute_inputs) =
            generate_record(proof);
        let (commit_ctx, commit_p2_inputs) =
            super::commit::generate_proving_ctx(user_pvs_proof.public_values.clone());
        let (memory_ctx, memory_p2_inputs) = memory::generate_proving_input(
            user_pvs_proof.public_values_commit,
            &user_pvs_proof.proof,
            memory_dimensions,
            user_pvs_proof.public_values.len(),
        );
        let DeferralOutputCtx {
            proving_ctx: output_ctx,
            poseidon2_inputs: output_p2_inputs,
            range_inputs,
            output_commit,
        } = super::output::generate_proving_ctx(
            verifier_pvs_record.app_exe_commit,
            verifier_pvs_record.app_vk_commit,
            user_pvs_proof.public_values.clone(),
        );

        let (paths_ctx, paths_p2_inputs) = if let Some(deferral_merkle_proofs) =
            deferral_merkle_proofs
        {
            assert!(self.deferral_enabled);
            let def_pvs: &DeferralPvs<_> = proof.public_values[DEF_PVS_AIR_ID].as_slice().borrow();
            let depth = def_pvs.depth.as_canonical_u32() as usize;
            let (acc_merkle_paths_ctx, acc_merkle_paths_p2_inputs) =
                def_paths::generate_proving_input(
                    def_pvs.initial_acc_hash,
                    def_pvs.final_acc_hash,
                    &deferral_merkle_proofs.initial_merkle_proof,
                    &deferral_merkle_proofs.final_merkle_proof,
                    memory_dimensions,
                    depth,
                    depth == 0,
                );
            (Some(acc_merkle_paths_ctx), acc_merkle_paths_p2_inputs)
        } else {
            assert!(!self.deferral_enabled);
            (None, vec![])
        };

        PreVerifierData {
            pre_verifier_ctxs: [commit_ctx, memory_ctx],
            post_verifier_ctxs: once(output_ctx).chain(paths_ctx).collect_vec(),
            poseidon2_compress_inputs: verifier_p2_compress_inputs
                .into_iter()
                .chain(commit_p2_inputs)
                .chain(memory_p2_inputs)
                .chain(output_p2_inputs)
                .chain(paths_p2_inputs)
                .collect_vec(),
            poseidon2_permute_inputs: verifier_p2_permute_inputs,
            range_inputs,
            verifier_pvs_record,
            output_commit,
        }
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proof: &Proof<SC>,
        record: DeferredVerifyPvsRecord<F>,
        final_transcript_state: [F; POSEIDON2_WIDTH],
        output_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContext<CpuBackend<SC>> {
        super::verifier::generate_proving_ctx(
            proof,
            record,
            final_transcript_state,
            output_commit,
            self.deferral_enabled,
        )
    }
}

#[cfg(feature = "cuda")]
impl DeferredVerifyTraceGen<GpuBackend> for DeferredVerifyTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
    ) -> PreVerifierData<GpuBackend> {
        let PreVerifierData {
            pre_verifier_ctxs,
            post_verifier_ctxs: post_verifier_ctx,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            range_inputs,
            verifier_pvs_record,
            output_commit,
        } = <Self as DeferredVerifyTraceGen<CpuBackend<SC>>>::pre_verifier_subcircuit_tracegen(
            self,
            proof,
            user_pvs_proof,
            memory_dimensions,
            deferral_merkle_proofs,
        );

        PreVerifierData {
            pre_verifier_ctxs: pre_verifier_ctxs.map(transport_air_proving_ctx_to_device),
            post_verifier_ctxs: post_verifier_ctx
                .into_iter()
                .map(transport_air_proving_ctx_to_device)
                .collect_vec(),
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            range_inputs,
            verifier_pvs_record,
            output_commit,
        }
    }

    fn generate_verifier_pvs_ctx(
        &self,
        proof: &Proof<SC>,
        record: DeferredVerifyPvsRecord<F>,
        final_transcript_state: [F; POSEIDON2_WIDTH],
        output_commit: [F; DIGEST_SIZE],
    ) -> AirProvingContext<GpuBackend> {
        transport_air_proving_ctx_to_device(super::verifier::generate_proving_ctx(
            proof,
            record,
            final_transcript_state,
            output_commit,
            self.deferral_enabled,
        ))
    }
}
