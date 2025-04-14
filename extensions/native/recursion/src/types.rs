use openvm_native_compiler::{
    asm::AsmConfig,
    ir::{Config, DIGEST_SIZE},
};
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicExpressionDag,
    config::{Com, StarkGenericConfig, Val},
    keygen::types::{LinearConstraint, MultiStarkVerifyingKey, StarkVerifyingKey, TraceWidth},
    p3_util::log2_strict_usize,
};

use crate::{
    digest::DigestVal,
    hints::{InnerChallenge, InnerVal},
};

pub type InnerConfig = AsmConfig<InnerVal, InnerChallenge>;

/// Constants determined by AIRs.
pub struct StarkVerificationAdvice<C: Config> {
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedDataInProgram<C>>,
    /// Trace sub-matrix widths
    pub width: TraceWidth,
    /// The factor to multiply the trace degree by to get the degree of the quotient polynomial.
    /// Determined from the max constraint degree of the AIR constraints. This is equivalently
    /// the number of chunks the quotient polynomial is split into.
    pub quotient_degree: usize,
    /// Number of public values for this STARK only
    pub num_public_values: usize,
    /// For only this RAP, how many challenges are needed in each trace challenge phase
    pub num_challenges_to_sample: Vec<usize>,
    /// Number of values to expose to verifier in each trace challenge phase
    pub num_exposed_values_after_challenge: Vec<usize>,
    /// Symbolic representation of all AIR constraints, including logup constraints
    pub symbolic_constraints: SymbolicExpressionDag<C::F>,
}

/// Create StarkVerificationAdvice for an inner config.
pub(crate) fn new_from_inner_vk<SC: StarkGenericConfig, C: Config<F = Val<SC>>>(
    vk: StarkVerifyingKey<Val<SC>, Com<SC>>,
) -> StarkVerificationAdvice<C>
where
    Com<SC>: Into<[C::F; DIGEST_SIZE]>,
{
    let StarkVerifyingKey {
        preprocessed_data,
        params,
        quotient_degree,
        symbolic_constraints,
        rap_phase_seq_kind: _,
    } = vk;
    StarkVerificationAdvice {
        preprocessed_data: preprocessed_data.map(|data| VerifierSinglePreprocessedDataInProgram {
            commit: DigestVal::F(data.commit.clone().into().to_vec()),
        }),
        width: params.width,
        quotient_degree: quotient_degree as usize,
        num_public_values: params.num_public_values,
        num_challenges_to_sample: params.num_challenges_to_sample,
        num_exposed_values_after_challenge: params.num_exposed_values_after_challenge,
        symbolic_constraints: symbolic_constraints.constraints,
    }
}

/// Constants determined by multiple AIRs.
pub struct MultiStarkVerificationAdvice<C: Config> {
    pub per_air: Vec<StarkVerificationAdvice<C>>,
    pub num_challenges_to_sample: Vec<usize>,
    pub trace_height_constraints: Vec<LinearConstraint>,
    pub log_up_pow_bits: usize,
    pub pre_hash: DigestVal<C>,
}

/// Create MultiStarkVerificationAdvice for an inner config.
pub fn new_from_inner_multi_vk<SC: StarkGenericConfig, C: Config<F = Val<SC>>>(
    vk: &MultiStarkVerifyingKey<SC>,
) -> MultiStarkVerificationAdvice<C>
where
    Com<SC>: Into<[C::F; DIGEST_SIZE]>,
{
    let num_challenges_to_sample = vk.num_challenges_per_phase();
    MultiStarkVerificationAdvice {
        per_air: vk
            .inner
            .per_air
            .iter()
            .map(|vk| new_from_inner_vk::<SC, C>(vk.clone()))
            .collect(),
        num_challenges_to_sample,
        trace_height_constraints: vk.inner.trace_height_constraints.clone(),
        log_up_pow_bits: vk.inner.log_up_pow_bits,
        pre_hash: DigestVal::F(vk.pre_hash.clone().into().to_vec()),
    }
}

impl<C: Config> StarkVerificationAdvice<C> {
    pub fn log_quotient_degree(&self) -> usize {
        log2_strict_usize(self.quotient_degree)
    }
}

pub struct VerifierSinglePreprocessedDataInProgram<C: Config> {
    pub commit: DigestVal<C>,
}
