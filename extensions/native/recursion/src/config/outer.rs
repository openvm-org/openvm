use openvm_native_compiler::ir::Config;
use openvm_stark_backend::{
    config::Com,
    keygen::types::{MultiStarkVerifyingKey, StarkVerifyingKey},
    p3_challenger::MultiField32Challenger,
    p3_commit::ExtensionMmcs,
    p3_field::extension::BinomialExtensionField,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2_root::BabyBearPoseidon2RootConfig,
    p3_baby_bear::BabyBear,
    p3_bn254_fr::{Bn254Fr, Poseidon2Bn254},
};
use p3_dft::Radix2DitParallel;
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof, TwoAdicFriPcs};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::{MultiField32PaddingFreeSponge, TruncatedPermutation};
use serde::{Deserialize, Serialize};

use crate::{
    digest::DigestVal,
    types::{
        MultiStarkVerificationAdvice, StarkVerificationAdvice,
        VerifierSinglePreprocessedDataInProgram,
    },
};

const WIDTH: usize = 3;
const RATE: usize = 16;
const DIGEST_WIDTH: usize = 1;

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct OuterConfig;

impl Config for OuterConfig {
    type N = Bn254Fr;
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
}

/// A configuration for outer recursion.
pub type OuterVal = BabyBear;
pub type OuterChallenge = BinomialExtensionField<OuterVal, 4>;
pub type OuterPerm = Poseidon2Bn254<WIDTH>;
pub type OuterHash =
    MultiField32PaddingFreeSponge<OuterVal, Bn254Fr, OuterPerm, WIDTH, RATE, DIGEST_WIDTH>;
pub type OuterDigest = [Bn254Fr; 1];
pub type OuterCompress = TruncatedPermutation<OuterPerm, 2, 1, WIDTH>;
pub type OuterValMmcs = MerkleTreeMmcs<BabyBear, Bn254Fr, OuterHash, OuterCompress, 1>;
pub type OuterChallengeMmcs = ExtensionMmcs<OuterVal, OuterChallenge, OuterValMmcs>;
pub type OuterDft = Radix2DitParallel<OuterVal>;
pub type OuterChallenger = MultiField32Challenger<OuterVal, Bn254Fr, OuterPerm, WIDTH, 2>;
pub type OuterPcs = TwoAdicFriPcs<OuterVal, OuterDft, OuterValMmcs, OuterChallengeMmcs>;
pub type OuterInputProof = Vec<OuterBatchOpening>;
pub type OuterQueryProof = QueryProof<OuterChallenge, OuterChallengeMmcs, OuterInputProof>;
pub type OuterCommitPhaseStep = CommitPhaseProofStep<OuterChallenge, OuterChallengeMmcs>;
pub type OuterFriProof = FriProof<OuterChallenge, OuterChallengeMmcs, OuterVal, OuterInputProof>;
pub type OuterBatchOpening = BatchOpening<OuterVal, OuterValMmcs>;

pub(crate) fn new_from_outer_vkv2(
    vk: StarkVerifyingKey<BabyBear, Com<BabyBearPoseidon2RootConfig>>,
) -> StarkVerificationAdvice<OuterConfig> {
    let StarkVerifyingKey {
        preprocessed_data,
        params,
        quotient_degree,
        symbolic_constraints,
        rap_phase_seq_kind: _,
    } = vk;
    StarkVerificationAdvice {
        preprocessed_data: preprocessed_data.map(|data| {
            let commit: [Bn254Fr; DIGEST_WIDTH] = data.commit.into();
            VerifierSinglePreprocessedDataInProgram {
                commit: DigestVal::N(commit.to_vec()),
            }
        }),
        width: params.width,
        quotient_degree: quotient_degree as usize,
        num_public_values: params.num_public_values,
        num_challenges_to_sample: params.num_challenges_to_sample,
        num_exposed_values_after_challenge: params.num_exposed_values_after_challenge,
        symbolic_constraints: symbolic_constraints.constraints,
    }
}

/// Create MultiStarkVerificationAdvice for the outer config.
pub fn new_from_outer_multi_vk(
    vk: &MultiStarkVerifyingKey<BabyBearPoseidon2RootConfig>,
) -> MultiStarkVerificationAdvice<OuterConfig> {
    let num_challenges_to_sample = vk.num_challenges_per_phase();
    MultiStarkVerificationAdvice {
        per_air: vk
            .per_air
            .clone()
            .into_iter()
            .map(new_from_outer_vkv2)
            .collect(),
        num_challenges_to_sample,
        trace_height_constraints: vk.trace_height_constraints.clone(),
        log_up_pow_bits: vk.log_up_pow_bits,
    }
}
