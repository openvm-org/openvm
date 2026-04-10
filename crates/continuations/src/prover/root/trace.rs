use itertools::Itertools;
use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProof;
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::hybrid_chip::cpu_proving_ctx_to_gpu;
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{hash_scheme::GpuHashScheme, GenericGpuBackend};
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
#[cfg(feature = "cuda")]
use openvm_recursion_circuit::system::GpuVerifierTraceGen;
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, CachedTraceCtx, VerifierExternalData, VerifierTraceGen,
};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    proof::Proof,
    prover::{AirProvingContext, ProverBackend, ProvingContext},
};
#[cfg(feature = "cuda")]
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, DIGEST_SIZE, EF, F,
};
use tracing::instrument;

use super::RootProver;
use crate::{
    circuit::{deferral::DeferralMerkleProofs, root::RootTraceGen, SubCircuitTraceData},
    RootSC, SC,
};

#[doc(hidden)]
pub trait RootTracegenBackend<PB, S, T>
where
    PB: ProverBackend<Val = F, Challenge = EF>,
    S: AggregationSubCircuit,
{
    type DeviceCtx: Clone + Send + Sync;

    fn generate_pre_verifier_subcircuit_ctx(
        tracegen: &T,
        proof: &Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        memory_dimensions: openvm_circuit::system::memory::dimensions::MemoryDimensions,
        device_ctx: &Self::DeviceCtx,
    ) -> SubCircuitTraceData<PB>;

    fn generate_other_proving_ctxs(
        tracegen: &T,
        proof: &Proof<SC>,
        memory_dimensions: openvm_circuit::system::memory::dimensions::MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<PB::Val>>,
        device_ctx: &Self::DeviceCtx,
    ) -> (
        Vec<AirProvingContext<PB>>,
        Vec<[PB::Val; openvm_circuit::arch::POSEIDON2_WIDTH]>,
    );

    fn generate_verifier_subcircuit_ctxs(
        verifier_circuit: &S,
        child_vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
        cached_trace_ctx: CachedTraceCtx<PB>,
        external_data: &mut VerifierExternalData,
        device_ctx: &Self::DeviceCtx,
    ) -> Option<Vec<AirProvingContext<PB>>>;
}

impl<S, T> RootTracegenBackend<CpuBackend<RootSC>, S, T> for CpuBackend<RootSC>
where
    S: AggregationSubCircuit + VerifierTraceGen<CpuBackend<RootSC>, RootSC>,
    T: RootTraceGen<CpuBackend<RootSC>>,
{
    type DeviceCtx = ();

    fn generate_pre_verifier_subcircuit_ctx(
        tracegen: &T,
        proof: &Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: openvm_circuit::system::memory::dimensions::MemoryDimensions,
        _device_ctx: &Self::DeviceCtx,
    ) -> SubCircuitTraceData<CpuBackend<RootSC>> {
        tracegen.generate_pre_verifier_subcircuit_ctx(proof, user_pvs_proof, memory_dimensions)
    }

    fn generate_other_proving_ctxs(
        tracegen: &T,
        proof: &Proof<SC>,
        memory_dimensions: openvm_circuit::system::memory::dimensions::MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
        _device_ctx: &Self::DeviceCtx,
    ) -> (
        Vec<AirProvingContext<CpuBackend<RootSC>>>,
        Vec<[F; openvm_circuit::arch::POSEIDON2_WIDTH]>,
    ) {
        tracegen.generate_other_proving_ctxs(proof, memory_dimensions, deferral_merkle_proofs)
    }

    fn generate_verifier_subcircuit_ctxs(
        verifier_circuit: &S,
        child_vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
        cached_trace_ctx: CachedTraceCtx<CpuBackend<RootSC>>,
        external_data: &mut VerifierExternalData,
        _device_ctx: &Self::DeviceCtx,
    ) -> Option<Vec<AirProvingContext<CpuBackend<RootSC>>>> {
        verifier_circuit.generate_proving_ctxs(
            child_vk,
            cached_trace_ctx,
            std::slice::from_ref(proof),
            external_data,
            default_duplex_sponge_recorder(),
        )
    }
}

#[cfg(feature = "cuda")]
impl<HS, S, T> RootTracegenBackend<GenericGpuBackend<HS>, S, T> for GenericGpuBackend<HS>
where
    HS: GpuHashScheme<SC = RootSC>,
    S: AggregationSubCircuit
        + VerifierTraceGen<CpuBackend<RootSC>, RootSC>
        + GpuVerifierTraceGen<GenericGpuBackend<HS>, RootSC>,
    T: RootTraceGen<CpuBackend<BabyBearPoseidon2Config>>,
{
    type DeviceCtx = GpuDeviceCtx;

    fn generate_pre_verifier_subcircuit_ctx(
        tracegen: &T,
        proof: &Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, F>,
        memory_dimensions: openvm_circuit::system::memory::dimensions::MemoryDimensions,
        device_ctx: &Self::DeviceCtx,
    ) -> SubCircuitTraceData<GenericGpuBackend<HS>> {
        let data =
            tracegen.generate_pre_verifier_subcircuit_ctx(proof, user_pvs_proof, memory_dimensions);
        SubCircuitTraceData {
            air_proving_ctxs: data
                .air_proving_ctxs
                .into_iter()
                .map(|ctx| cpu_proving_ctx_to_gpu::<HS>(ctx, device_ctx))
                .collect_vec(),
            poseidon2_compress_inputs: data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: data.poseidon2_permute_inputs,
        }
    }

    fn generate_other_proving_ctxs(
        tracegen: &T,
        proof: &Proof<SC>,
        memory_dimensions: openvm_circuit::system::memory::dimensions::MemoryDimensions,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<F>>,
        device_ctx: &Self::DeviceCtx,
    ) -> (
        Vec<AirProvingContext<GenericGpuBackend<HS>>>,
        Vec<[F; openvm_circuit::arch::POSEIDON2_WIDTH]>,
    ) {
        let (ctxs, inputs) =
            tracegen.generate_other_proving_ctxs(proof, memory_dimensions, deferral_merkle_proofs);
        (
            ctxs.into_iter()
                .map(|ctx| cpu_proving_ctx_to_gpu::<HS>(ctx, device_ctx))
                .collect_vec(),
            inputs,
        )
    }

    fn generate_verifier_subcircuit_ctxs(
        verifier_circuit: &S,
        child_vk: &MultiStarkVerifyingKey<SC>,
        proof: &Proof<SC>,
        cached_trace_ctx: CachedTraceCtx<GenericGpuBackend<HS>>,
        external_data: &mut VerifierExternalData,
        device_ctx: &Self::DeviceCtx,
    ) -> Option<Vec<AirProvingContext<GenericGpuBackend<HS>>>> {
        verifier_circuit.generate_proving_ctxs_gpu(
            child_vk,
            cached_trace_ctx,
            std::slice::from_ref(proof),
            external_data,
            device_ctx,
            default_duplex_sponge_recorder(),
        )
    }
}

impl<S: AggregationSubCircuit, T> RootProver<S, T> {
    pub fn generate_proving_ctx<PB>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        deferral_merkle_proofs: Option<&DeferralMerkleProofs<PB::Val>>,
        device_ctx: &<PB as RootTracegenBackend<PB, S, T>>::DeviceCtx,
    ) -> Option<ProvingContext<PB>>
    where
        PB: ProverBackend<Val = F, Challenge = EF> + RootTracegenBackend<PB, S, T>,
        PB::Matrix: Clone,
    {
        assert_eq!(
            user_pvs_proof.public_values.len(),
            self.circuit.num_user_pvs
        );

        // These AIRs should have the same height regardless of proof or user_pvs_proof.
        let mut pre_data =
            <PB as RootTracegenBackend<PB, S, T>>::generate_pre_verifier_subcircuit_ctx(
                &self.agg_node_tracegen,
                &proof,
                user_pvs_proof,
                self.circuit.memory_dimensions,
                device_ctx,
            );
        let (post_verifier_subcircuit_ctxs, other_compress_inputs) =
            <PB as RootTracegenBackend<PB, S, T>>::generate_other_proving_ctxs(
                &self.agg_node_tracegen,
                &proof,
                self.circuit.memory_dimensions,
                deferral_merkle_proofs,
                device_ctx,
            );
        pre_data
            .poseidon2_compress_inputs
            .extend(other_compress_inputs);

        // Get the verifier sub-circuit trace heights. If deferrals are enabled, there is
        // an additional AIR at the end.
        let verifier_trace_heights = self.trace_heights.as_ref().map(|v| {
            let num_airs = v.len() - deferral_merkle_proofs.is_some() as usize;
            &v[3..num_airs]
        });

        let range_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &pre_data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: &pre_data.poseidon2_permute_inputs,
            range_check_inputs: &range_check_inputs,
            required_heights: verifier_trace_heights,
            final_transcript_state: None,
        };

        let subcircuit_ctxs =
            <PB as RootTracegenBackend<PB, S, T>>::generate_verifier_subcircuit_ctxs(
                &self.circuit.verifier_circuit,
                &self.child_vk,
                &proof,
                CachedTraceCtx::Records(self.cached_trace_record.clone()),
                &mut external_data,
                device_ctx,
            );

        subcircuit_ctxs.map(|subcircuit_ctxs| ProvingContext {
            per_trace: pre_data
                .air_proving_ctxs
                .into_iter()
                .chain(subcircuit_ctxs)
                .chain(post_verifier_subcircuit_ctxs)
                .enumerate()
                .collect_vec(),
        })
    }

    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx_no_def<PB>(
        &self,
        proof: Proof<SC>,
        user_pvs_proof: &UserPublicValuesProof<DIGEST_SIZE, PB::Val>,
        device_ctx: &<PB as RootTracegenBackend<PB, S, T>>::DeviceCtx,
    ) -> Option<ProvingContext<PB>>
    where
        PB: ProverBackend<Val = F, Challenge = EF> + RootTracegenBackend<PB, S, T>,
        PB::Matrix: Clone,
    {
        assert!(
            self.circuit.def_hook_commit.is_none(),
            "deferral-enabled root prover requires generate_proving_ctx_with_deferrals"
        );
        self.generate_proving_ctx(proof, user_pvs_proof, None, device_ctx)
    }
}
