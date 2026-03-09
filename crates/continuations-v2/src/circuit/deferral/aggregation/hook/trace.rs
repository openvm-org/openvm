use std::borrow::Borrow;

use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ProverBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use p3_field::PrimeCharacteristicRing;
use verify_stark::pvs::VerifierBasePvs;

use crate::SC;

pub type DeferralIoCommit<F> = ([F; DIGEST_SIZE], [F; DIGEST_SIZE]);

pub struct DeferralHookPreCtx<PB: ProverBackend> {
    pub verifier_pvs_ctx: AirProvingContext<PB>,
    pub decommit_ctx: AirProvingContext<PB>,
    pub onion_ctx: AirProvingContext<PB>,
    pub poseidon2_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
}

// Trait used to remain generic in PB.
pub trait DeferralHookTraceGen<PB: ProverBackend> {
    fn new() -> Self;

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<PB::Val>>,
    ) -> DeferralHookPreCtx<PB>;
}

pub struct DeferralHookTraceGenImpl;

fn normalize_leaf_children(
    mut leaf_children: Vec<DeferralIoCommit<F>>,
) -> (Vec<DeferralIoCommit<F>>, usize) {
    assert!(
        !leaf_children.is_empty(),
        "deferral hook requires at least one leaf commit"
    );
    let num_real_leaves = leaf_children.len();
    let target_len = leaf_children.len().next_power_of_two();
    leaf_children.resize(target_len, ([F::ZERO; DIGEST_SIZE], [F::ZERO; DIGEST_SIZE]));
    (leaf_children, num_real_leaves)
}

impl DeferralHookTraceGen<CpuBackend<BabyBearPoseidon2Config>> for DeferralHookTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> DeferralHookPreCtx<CpuBackend<BabyBearPoseidon2Config>> {
        let (leaf_children, num_real_leaves) = normalize_leaf_children(leaf_children);
        let super::decommit::MerkleDecommitTraceCtx {
            proving_ctx: decommit_ctx,
            poseidon2_inputs: decommit_p2_inputs,
            io_commits,
            merkle_root: computed_merkle_root,
        } = super::decommit::generate_proving_ctx(leaf_children, num_real_leaves);

        let def_pvs: &crate::circuit::deferral::DeferralAggregationPvs<F> =
            proof.public_values[1].as_slice().borrow();
        assert_eq!(
            computed_merkle_root, def_pvs.merkle_commit,
            "leaf_children do not match the child proof merkle_commit"
        );

        let verifier_pvs: &VerifierBasePvs<F> = proof.public_values[0].as_slice().borrow();
        let (_, def_vk_commit) = super::verifier::def_vk_commit_from_verifier_pvs(verifier_pvs);

        let super::onion::OnionTraceCtx {
            proving_ctx: onion_ctx,
            poseidon2_inputs: onion_p2_inputs,
            input_onion,
            output_onion,
        } = super::onion::generate_proving_ctx(def_vk_commit, io_commits);

        let (verifier_pvs_ctx, verifier_p2_inputs, _) =
            super::verifier::generate_proving_ctx(proof, input_onion, output_onion);

        DeferralHookPreCtx {
            verifier_pvs_ctx,
            decommit_ctx,
            onion_ctx,
            poseidon2_inputs: verifier_p2_inputs
                .into_iter()
                .chain(decommit_p2_inputs)
                .chain(onion_p2_inputs)
                .collect(),
        }
    }
}

#[cfg(feature = "cuda")]
impl DeferralHookTraceGen<GpuBackend> for DeferralHookTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proof: &Proof<SC>,
        leaf_children: Vec<DeferralIoCommit<F>>,
    ) -> DeferralHookPreCtx<GpuBackend> {
        let DeferralHookPreCtx {
            verifier_pvs_ctx,
            decommit_ctx,
            onion_ctx,
            poseidon2_inputs,
        } = <Self as DeferralHookTraceGen<CpuBackend<BabyBearPoseidon2Config>>>::pre_verifier_subcircuit_tracegen(self, proof, leaf_children);

        DeferralHookPreCtx {
            verifier_pvs_ctx: transport_air_proving_ctx_to_device(verifier_pvs_ctx),
            decommit_ctx: transport_air_proving_ctx_to_device(decommit_ctx),
            onion_ctx: transport_air_proving_ctx_to_device(onion_ctx),
            poseidon2_inputs,
        }
    }
}
