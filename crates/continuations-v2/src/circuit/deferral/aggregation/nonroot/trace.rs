use std::borrow::Borrow;

use openvm_circuit::arch::POSEIDON2_WIDTH;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{data_transporter::transport_air_proving_ctx_to_device, GpuBackend};
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, CpuBackend, ProverBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};

use crate::{
    circuit::deferral::{
        DeferralAggregationPvs, DeferralCircuitPvs, DEF_AGG_PVS_AIR_ID, DEF_CIRCUIT_PVS_AIR_ID,
    },
    utils::{digests_to_poseidon2_input, zero_hash},
};

pub struct DeferralNonRootPreCtx<PB: ProverBackend> {
    pub verifier_pvs_ctx: AirProvingContext<PB>,
    pub def_pvs_ctx: AirProvingContext<PB>,
    pub input_ctx: AirProvingContext<PB>,
    pub poseidon2_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
}

fn fold_leaf_input_commit(
    proof: &Proof<BabyBearPoseidon2Config>,
    init: [F; DIGEST_SIZE],
) -> [F; DIGEST_SIZE] {
    proof
        .trace_vdata
        .iter()
        .flatten()
        .flat_map(|vdata| vdata.cached_commitments.iter().copied())
        .fold(init, |acc, cached_commit| {
            poseidon2_compress_with_capacity(acc, cached_commit).0
        })
}

fn child_merkle_commit(
    proof: &Proof<BabyBearPoseidon2Config>,
    child_is_def: bool,
) -> [F; DIGEST_SIZE] {
    if child_is_def {
        let child_pvs: &DeferralAggregationPvs<F> =
            proof.public_values[DEF_AGG_PVS_AIR_ID].as_slice().borrow();
        child_pvs.merkle_commit
    } else {
        let child_pvs: &DeferralCircuitPvs<F> = proof.public_values[DEF_CIRCUIT_PVS_AIR_ID]
            .as_slice()
            .borrow();
        let folded_input_commit = fold_leaf_input_commit(proof, child_pvs.input_commit);
        poseidon2_compress_with_capacity(folded_input_commit, child_pvs.output_commit).0
    }
}

fn generate_poseidon2_inputs(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_def: bool,
    child_merkle_depth: Option<usize>,
) -> Vec<[F; POSEIDON2_WIDTH]> {
    let mut poseidon2_inputs: Vec<[F; POSEIDON2_WIDTH]> = Vec::new();

    for proof in proofs {
        if child_is_def {
            continue;
        }
        let child_pvs: &DeferralCircuitPvs<F> = proof.public_values[DEF_CIRCUIT_PVS_AIR_ID]
            .as_slice()
            .borrow();

        // InputCommitAir: hash input_commit with each cached trace commitment.
        let mut current_commit = child_pvs.input_commit;
        for cached_commit in proof
            .trace_vdata
            .iter()
            .flatten()
            .flat_map(|vdata| vdata.cached_commitments.iter().copied())
        {
            poseidon2_inputs.push(digests_to_poseidon2_input(current_commit, cached_commit));
            current_commit = poseidon2_compress_with_capacity(current_commit, cached_commit).0;
        }

        // DeferralPvsAir (leaf): hash folded input_commit and output_commit into merkle_commit.
        poseidon2_inputs.push(digests_to_poseidon2_input(
            current_commit,
            child_pvs.output_commit,
        ));
    }

    // DeferralPvsAir: hash child merkle commits when this is not a wrapper.
    if let Some(depth) = child_merkle_depth {
        let left_merkle = child_merkle_commit(&proofs[0], child_is_def);
        let right_merkle = if proofs.len() == 2 {
            child_merkle_commit(&proofs[1], child_is_def)
        } else {
            zero_hash(depth + 1)
        };
        poseidon2_inputs.push(digests_to_poseidon2_input(left_merkle, right_merkle));
    }

    poseidon2_inputs
}

// Trait used to remain generic in PB
pub trait DeferralNonRootTraceGen<PB: ProverBackend> {
    fn new() -> Self;
    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: PB::Commitment,
        child_merkle_depth: Option<usize>,
    ) -> DeferralNonRootPreCtx<PB>;
}

pub struct DeferralNonRootTraceGenImpl;

impl DeferralNonRootTraceGen<CpuBackend<BabyBearPoseidon2Config>> for DeferralNonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: [F; DIGEST_SIZE],
        child_merkle_depth: Option<usize>,
    ) -> DeferralNonRootPreCtx<CpuBackend<BabyBearPoseidon2Config>> {
        DeferralNonRootPreCtx {
            verifier_pvs_ctx: super::verifier::generate_proving_ctx(
                proofs,
                child_is_def,
                child_dag_commit,
            ),
            def_pvs_ctx: super::def_pvs::generate_proving_ctx(
                proofs,
                child_is_def,
                child_merkle_depth,
            ),
            input_ctx: super::input::generate_proving_ctx(proofs, child_is_def),
            poseidon2_inputs: generate_poseidon2_inputs(proofs, child_is_def, child_merkle_depth),
        }
    }
}

#[cfg(feature = "cuda")]
impl DeferralNonRootTraceGen<GpuBackend> for DeferralNonRootTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: [F; DIGEST_SIZE],
        child_merkle_depth: Option<usize>,
    ) -> DeferralNonRootPreCtx<GpuBackend> {
        let DeferralNonRootPreCtx {
            verifier_pvs_ctx,
            def_pvs_ctx,
            input_ctx,
            poseidon2_inputs,
        } = <Self as DeferralNonRootTraceGen<CpuBackend<BabyBearPoseidon2Config>>>::pre_verifier_subcircuit_tracegen(
            self,
            proofs,
            child_is_def,
            child_dag_commit,
            child_merkle_depth,
        );
        DeferralNonRootPreCtx {
            verifier_pvs_ctx: transport_air_proving_ctx_to_device(verifier_pvs_ctx),
            def_pvs_ctx: transport_air_proving_ctx_to_device(def_pvs_ctx),
            input_ctx: transport_air_proving_ctx_to_device(input_ctx),
            poseidon2_inputs,
        }
    }
}
