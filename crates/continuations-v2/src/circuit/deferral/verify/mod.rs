use itertools::Itertools;
use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
};
use recursion_circuit::prelude::{DIGEST_SIZE, F, SC};

use crate::circuit::{
    deferral::verify::verifier::{generate_record, DeferredVerifyPvsRecord},
    user_pvs::{commit, memory},
};

pub mod bus;
pub mod output;
pub mod verifier;

pub struct PreVerifierData<PB: ProverBackend> {
    pub proving_ctxs: Vec<AirProvingContext<PB>>,
    pub poseidon2_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub range_inputs: Vec<usize>,
    pub verifier_pvs_record: DeferredVerifyPvsRecord<PB::Val>,
    pub output_commit: [PB::Val; DIGEST_SIZE],
}

// Trait used to remain generic in PB
pub trait DeferredVerifyTraceGen<PB: ProverBackend> {
    // Returns the AIR proving contexts, Poseidon2 and range inputs, and the data
    // needed to compute the DeferredVerifyPvsAir trace later
    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        memory_dimensions: MemoryDimensions,
    ) -> PreVerifierData<PB>;

    fn generate_verifier_pvs_ctx(
        &self,
        proof: &Proof<SC>,
        record: DeferredVerifyPvsRecord<PB::Val>,
        final_transcript_state: [PB::Val; POSEIDON2_WIDTH],
        output_commit: [PB::Val; DIGEST_SIZE],
    ) -> AirProvingContext<PB>;
}

pub struct DeferredVerifyTraceGenImpl;

impl DeferredVerifyTraceGen<CpuBackend<SC>> for DeferredVerifyTraceGenImpl {
    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: MemoryDimensions,
    ) -> PreVerifierData<CpuBackend<SC>> {
        let (verifier_pvs_record, verifier_p2_inputs) = generate_record(proof);
        let (commit_ctx, commit_p2_inputs) =
            commit::generate_proving_ctx(user_pvs_proof.public_values.clone(), false);
        let (memory_ctx, memory_p2_inputs) = memory::generate_proving_input(
            user_pvs_proof.public_values_commit,
            &user_pvs_proof.proof,
            memory_dimensions,
            user_pvs_proof.public_values.len(),
        );
        let (output_ctx, output_p2_inputs, range_inputs, output_commit) =
            output::generate_proving_ctx(
                verifier_pvs_record.app_exe_commit,
                verifier_pvs_record.app_vk_commit,
                user_pvs_proof.public_values.clone(),
            );

        PreVerifierData {
            proving_ctxs: vec![commit_ctx, memory_ctx, output_ctx],
            poseidon2_inputs: verifier_p2_inputs
                .into_iter()
                .chain(commit_p2_inputs)
                .chain(memory_p2_inputs)
                .chain(output_p2_inputs)
                .collect_vec(),
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
        verifier::generate_proving_ctx(proof, record, final_transcript_state, output_commit)
    }
}
