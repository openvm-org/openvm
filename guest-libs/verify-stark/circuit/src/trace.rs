use std::{borrow::Borrow, iter::once};

use itertools::Itertools;
use openvm_circuit::system::{
    memory::dimensions::MemoryDimensions, public_values::proof::PublicValuesOpening,
};
use openvm_continuations::circuit::{deferral::DeferralMerkleProofs, root::def_paths};
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::prelude::{DIGEST_SIZE, F, SC};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ProverBackend},
};
use openvm_verify_stark_host::pvs::{DeferralPvs, DEF_PVS_AIR_ID};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
#[cfg(feature = "cuda")]
use {
    openvm_circuit_primitives::hybrid_chip::cpu_proving_ctx_to_gpu,
    openvm_cuda_backend::GpuBackend, openvm_cuda_common::stream::GpuDeviceCtx,
};

use crate::{
    commit::generate_proving_ctx as generate_commit_proving_ctx,
    output::DeferralOutputCtx,
    verifier::{generate_record, DeferredVerifyPvsRecord},
};

pub struct PreVerifierData<PB: ProverBackend> {
    pub pre_verifier_ctxs: [AirProvingContext<PB>; 1],
    pub post_verifier_ctxs: Vec<AirProvingContext<PB>>,
    pub poseidon2_compress_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub poseidon2_permute_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub range_inputs: Vec<usize>,
    pub verifier_pvs_record: DeferredVerifyPvsRecord<PB::Val>,
    pub output_commit: [PB::Val; DIGEST_SIZE],
}

// Trait used to remain generic in PB
pub trait DeferredVerifyTraceGen<PB: ProverBackend, DC: Clone + Send + Sync> {
    fn new(deferral_enabled: bool) -> Self;

    // Returns the AIR proving contexts, Poseidon2 and range inputs, and the data
    // needed to compute the DeferredVerifyPvsAir trace later
    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        public_values_opening: &PublicValuesOpening<PB::Val>,
        memory_dimensions: MemoryDimensions,
        def_idx: usize,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
        device_ctx: &DC,
    ) -> PreVerifierData<PB>;

    fn generate_verifier_pvs_ctx(
        &self,
        proof: &Proof<SC>,
        record: DeferredVerifyPvsRecord<PB::Val>,
        final_transcript_state: [PB::Val; POSEIDON2_WIDTH],
        output_commit: [PB::Val; DIGEST_SIZE],
        def_idx: usize,
        device_ctx: &DC,
    ) -> AirProvingContext<PB>;
}

pub struct DeferredVerifyTraceGenImpl {
    pub deferral_enabled: bool,
}

impl DeferredVerifyTraceGen<CpuBackend<SC>, ()> for DeferredVerifyTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        public_values_opening: &PublicValuesOpening<F>,
        memory_dimensions: MemoryDimensions,
        def_idx: usize,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
        _device_ctx: &(),
    ) -> PreVerifierData<CpuBackend<SC>> {
        let (
            verifier_pvs_record,
            verifier_p2_compress_inputs,
            verifier_p2_permute_inputs,
            verifier_range_inputs,
        ) = generate_record(proof);
        let (commit_ctx, commit_p2_inputs) = generate_commit_proving_ctx(
            &public_values_opening.public_values,
            public_values_opening.num_values,
        );
        let output_user_pvs = std::iter::once(F::from_usize(public_values_opening.num_values))
            .chain(std::iter::once(F::ZERO))
            .chain(public_values_opening.public_values.iter().copied())
            .collect();
        let DeferralOutputCtx {
            proving_ctx: output_ctx,
            poseidon2_inputs: output_p2_inputs,
            range_inputs,
            output_commit,
        } = super::output::generate_proving_ctx(
            verifier_pvs_record.app_exe_commit,
            verifier_pvs_record.app_vm_commit,
            output_user_pvs,
            def_idx,
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
            pre_verifier_ctxs: [commit_ctx],
            post_verifier_ctxs: once(output_ctx).chain(paths_ctx).collect_vec(),
            poseidon2_compress_inputs: verifier_p2_compress_inputs
                .into_iter()
                .chain(commit_p2_inputs)
                .chain(paths_p2_inputs)
                .collect_vec(),
            poseidon2_permute_inputs: verifier_p2_permute_inputs
                .into_iter()
                .chain(output_p2_inputs)
                .collect_vec(),
            range_inputs: verifier_range_inputs
                .into_iter()
                .chain(range_inputs)
                .collect_vec(),
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
        def_idx: usize,
        _device_ctx: &(),
    ) -> AirProvingContext<CpuBackend<SC>> {
        super::verifier::generate_proving_ctx(
            proof,
            record,
            final_transcript_state,
            output_commit,
            def_idx,
            self.deferral_enabled,
        )
    }
}

#[cfg(feature = "cuda")]
impl DeferredVerifyTraceGen<GpuBackend, GpuDeviceCtx> for DeferredVerifyTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        public_values_opening: &PublicValuesOpening<F>,
        memory_dimensions: MemoryDimensions,
        def_idx: usize,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
        device_ctx: &GpuDeviceCtx,
    ) -> PreVerifierData<GpuBackend> {
        let PreVerifierData {
            pre_verifier_ctxs,
            post_verifier_ctxs: post_verifier_ctx,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            range_inputs,
            verifier_pvs_record,
            output_commit,
        } = <Self as DeferredVerifyTraceGen<CpuBackend<SC>, ()>>::pre_verifier_subcircuit_tracegen(
            self,
            proof,
            public_values_opening,
            memory_dimensions,
            def_idx,
            deferral_merkle_proofs,
            &(),
        );

        PreVerifierData {
            pre_verifier_ctxs: pre_verifier_ctxs
                .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx)),
            post_verifier_ctxs: post_verifier_ctx
                .into_iter()
                .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx))
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
        def_idx: usize,
        device_ctx: &GpuDeviceCtx,
    ) -> AirProvingContext<GpuBackend> {
        cpu_proving_ctx_to_gpu(
            super::verifier::generate_proving_ctx(
                proof,
                record,
                final_transcript_state,
                output_commit,
                def_idx,
                self.deferral_enabled,
            ),
            device_ctx,
        )
    }
}
