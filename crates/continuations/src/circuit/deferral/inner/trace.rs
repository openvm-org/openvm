use std::{borrow::Borrow, iter::once};

use itertools::Itertools;
use openvm_circuit::arch::POSEIDON2_WIDTH;
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::hybrid_chip::cpu_proving_ctx_to_gpu;
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
use openvm_recursion_circuit::utils::poseidon2_hash_slice_with_states;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ProverBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::pvs::DagCommit;

use crate::{
    circuit::deferral::{
        DeferralAggregationPvs, DeferralCircuitPvs, DEF_AGG_PVS_AIR_ID, DEF_CIRCUIT_PVS_AIR_ID,
    },
    utils::{digests_to_poseidon2_input, zero_hash},
};

pub struct DeferralInnerPreCtx<PB: ProverBackend> {
    pub verifier_pvs_ctx: AirProvingContext<PB>,
    pub def_pvs_ctx: AirProvingContext<PB>,
    pub input_ctx: AirProvingContext<PB>,
    pub poseidon2_compress_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
    pub poseidon2_permute_inputs: Vec<[PB::Val; POSEIDON2_WIDTH]>,
}

fn fold_leaf_input_commit(
    proof: &Proof<BabyBearPoseidon2Config>,
    init: [F; DIGEST_SIZE],
) -> ([F; DIGEST_SIZE], Vec<[F; POSEIDON2_WIDTH]>) {
    let values = once(init)
        .chain(
            proof
                .trace_vdata
                .iter()
                .flatten()
                .flat_map(|vdata| vdata.cached_commitments.iter().copied()),
        )
        .flatten()
        .collect_vec();
    let (folded_input_commit, pre_states, _) = poseidon2_hash_slice_with_states(&values);
    (folded_input_commit, pre_states)
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
        let (folded_input_commit, _) = fold_leaf_input_commit(proof, child_pvs.input_commit);
        poseidon2_compress_with_capacity(folded_input_commit, child_pvs.output_commit).0
    }
}

fn generate_poseidon2_inputs(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    child_is_def: bool,
    child_merkle_depth: Option<usize>,
) -> (Vec<[F; POSEIDON2_WIDTH]>, Vec<[F; POSEIDON2_WIDTH]>) {
    let mut poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]> = Vec::new();
    let mut poseidon2_permute_inputs: Vec<[F; POSEIDON2_WIDTH]> = Vec::new();

    for proof in proofs {
        if child_is_def {
            continue;
        }
        let child_pvs: &DeferralCircuitPvs<F> = proof.public_values[DEF_CIRCUIT_PVS_AIR_ID]
            .as_slice()
            .borrow();

        // InputCommitAir: sponge-hash input_commit and cached trace commitments.
        let (folded_input_commit, input_permute_inputs) =
            fold_leaf_input_commit(proof, child_pvs.input_commit);
        poseidon2_permute_inputs.extend(input_permute_inputs);

        // DeferralAggPvsAir (leaf): hash folded input_commit and output_commit into merkle_commit.
        poseidon2_compress_inputs.push(digests_to_poseidon2_input(
            folded_input_commit,
            child_pvs.output_commit,
        ));
    }

    // DeferralAggPvsAir: hash child merkle commits when this is not a wrapper.
    if let Some(depth) = child_merkle_depth {
        let left_merkle = child_merkle_commit(&proofs[0], child_is_def);
        let right_merkle = if proofs.len() == 2 {
            child_merkle_commit(&proofs[1], child_is_def)
        } else {
            zero_hash(depth + 1)
        };
        poseidon2_compress_inputs.push(digests_to_poseidon2_input(left_merkle, right_merkle));
    }

    (poseidon2_compress_inputs, poseidon2_permute_inputs)
}

// Trait used to remain generic in PB
pub trait DeferralInnerTraceGen<PB: ProverBackend> {
    fn new() -> Self;
    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: DagCommit<F>,
        child_merkle_depth: Option<usize>,
    ) -> DeferralInnerPreCtx<PB>;
}

pub struct DeferralInnerTraceGenImpl;

impl DeferralInnerTraceGen<CpuBackend<BabyBearPoseidon2Config>> for DeferralInnerTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: DagCommit<F>,
        child_merkle_depth: Option<usize>,
    ) -> DeferralInnerPreCtx<CpuBackend<BabyBearPoseidon2Config>> {
        let (poseidon2_compress_inputs, poseidon2_permute_inputs) =
            generate_poseidon2_inputs(proofs, child_is_def, child_merkle_depth);
        DeferralInnerPreCtx {
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
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        }
    }
}

#[cfg(feature = "cuda")]
impl DeferralInnerTraceGen<GpuBackend> for DeferralInnerTraceGenImpl {
    fn new() -> Self {
        Self
    }

    fn pre_verifier_subcircuit_tracegen(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        child_is_def: bool,
        child_dag_commit: DagCommit<F>,
        child_merkle_depth: Option<usize>,
    ) -> DeferralInnerPreCtx<GpuBackend> {
        let DeferralInnerPreCtx {
            verifier_pvs_ctx,
            def_pvs_ctx,
            input_ctx,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        } = <Self as DeferralInnerTraceGen<CpuBackend<BabyBearPoseidon2Config>>>::pre_verifier_subcircuit_tracegen(
            self,
            proofs,
            child_is_def,
            child_dag_commit,
            child_merkle_depth,
        );
        DeferralInnerPreCtx {
            verifier_pvs_ctx: cpu_proving_ctx_to_gpu(verifier_pvs_ctx),
            def_pvs_ctx: cpu_proving_ctx_to_gpu(def_pvs_ctx),
            input_ctx: cpu_proving_ctx_to_gpu(input_ctx),
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        }
    }
}
