use afs_compiler::ir::{Config, DIGEST_SIZE};
use afs_stark_backend::{
    air_builders::symbolic::symbolic_expression::SymbolicExpression,
    config::Com,
    keygen::{
        types::TraceWidth,
        v2::types::{MultiStarkVerifyingKeyV2, StarkVerifyingKeyV2},
    },
};
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;

use crate::{digest::DigestVal, types::VerifierSinglePreprocessedDataInProgram};

/// Constants determined by AIRs.
pub struct StarkVerificationAdviceV2<C: Config> {
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedDataInProgram<C>>,
    /// Trace sub-matrix widths
    pub width: TraceWidth,
    /// The factor to multiply the trace degree by to get the degree of the quotient polynomial. Determined from the max constraint degree of the AIR constraints.
    /// This is equivalently the number of chunks the quotient polynomial is split into.
    pub quotient_degree: usize,
    /// Number of public values for this STARK only
    pub num_public_values: usize,
    /// For only this RAP, how many challenges are needed in each trace challenge phase
    pub num_challenges_to_sample: Vec<usize>,
    /// Number of values to expose to verifier in each trace challenge phase
    pub num_exposed_values_after_challenge: Vec<usize>,
    /// Symbolic representation of all AIR constraints, including logup constraints
    pub symbolic_constraints: Vec<SymbolicExpression<C::F>>,
}

/// Create StarkVerificationAdvice for an inner config.
// TODO: the bound C::F = Val<SC> is very awkward
pub(crate) fn new_from_inner_vkv2<SC: StarkGenericConfig, C: Config<F = Val<SC>>>(
    vk: StarkVerifyingKeyV2<SC>,
) -> StarkVerificationAdviceV2<C>
where
    Com<SC>: Into<[C::F; DIGEST_SIZE]>,
{
    let StarkVerifyingKeyV2::<SC> {
        preprocessed_data,
        params,
        quotient_degree,
        symbolic_constraints,
    } = vk;
    StarkVerificationAdviceV2 {
        preprocessed_data: preprocessed_data.map(|data| VerifierSinglePreprocessedDataInProgram {
            commit: DigestVal::F(data.commit.clone().into().to_vec()),
        }),
        width: params.width,
        quotient_degree,
        num_public_values: params.num_public_values,
        num_challenges_to_sample: params.num_challenges_to_sample,
        num_exposed_values_after_challenge: params.num_exposed_values_after_challenge,
        symbolic_constraints: symbolic_constraints.constraints,
    }
}

/// Constants determined by multiple AIRs.
pub struct MultiStarkVerificationAdviceV2<C: Config> {
    pub per_air: Vec<StarkVerificationAdviceV2<C>>,
    pub num_challenges_to_sample: Vec<usize>,
}

/// Create MultiStarkVerificationAdvice for an inner config.
// TODO: the bound C::F = Val<SC> is very awkward
pub fn new_from_inner_multi_vkv2<SC: StarkGenericConfig, C: Config<F = Val<SC>>>(
    vk: &MultiStarkVerifyingKeyV2<SC>,
) -> MultiStarkVerificationAdviceV2<C>
where
    Com<SC>: Into<[C::F; DIGEST_SIZE]>,
{
    let num_challenges_to_sample = vk.num_challenges_to_sample();
    let MultiStarkVerifyingKeyV2::<SC> { per_air } = vk;
    MultiStarkVerificationAdviceV2 {
        per_air: per_air
            .clone()
            .into_iter()
            .map(new_from_inner_vkv2)
            .collect(),
        num_challenges_to_sample,
    }
}

impl<C: Config> StarkVerificationAdviceV2<C> {
    pub fn log_quotient_degree(&self) -> usize {
        log2_strict_usize(self.quotient_degree)
    }
}
