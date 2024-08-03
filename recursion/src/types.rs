use afs_compiler::{
    asm::AsmConfig,
    ir::{Array, Builder, Config, Ext, Felt, Var},
    prelude::*,
};
use afs_stark_backend::{
    air_builders::symbolic::SymbolicConstraints,
    commit::MatrixCommitmentPointers,
    config::Com,
    keygen::types::{CommitmentToAirGraph, MultiStarkVerifyingKey, StarkVerifyingKey, TraceWidth},
    prover::types::Proof,
};
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;

use crate::{
    fri::types::{DigestVariable, TwoAdicPcsProofVariable},
    hints::{InnerChallenge, InnerVal},
};

pub type InnerConfig = AsmConfig<InnerVal, InnerChallenge>;

/// The maximum number of elements that can be stored in the public values vec.  Both SP1 and recursive
/// proofs need to pad their public_values vec to this length.  This is required since the recursion
/// verification program expects the public values vec to be fixed length.
pub const PROOF_MAX_NUM_PVS: usize = 240;

pub struct VerifierInput<SC: StarkGenericConfig> {
    pub proof: Proof<SC>,
    pub log_degree_per_air: Vec<usize>,
    pub public_values: Vec<Vec<Val<SC>>>,
}

#[derive(DslVariable, Clone)]
pub struct VerifierInputVariable<C: Config> {
    pub proof: StarkProofVariable<C>,
    pub log_degree_per_air: Array<C, Usize<C::N>>,
    pub public_values: Array<C, Array<C, Felt<C::F>>>,
}

#[derive(DslVariable, Clone)]
pub struct TraceWidthVariable<C: Config> {
    pub preprocessed: Array<C, Var<C::N>>,
    pub partitioned_main: Array<C, Var<C::N>>,
    pub after_challenge: Array<C, Var<C::N>>,
}

#[derive(DslVariable, Clone)]
pub struct CommitmentsVariable<C: Config> {
    pub main_trace: Array<C, DigestVariable<C>>,
    pub after_challenge: Array<C, DigestVariable<C>>,
    pub quotient: DigestVariable<C>,
}

#[derive(DslVariable, Clone)]
pub struct StarkProofVariable<C: Config> {
    pub commitments: CommitmentsVariable<C>,
    pub opening: OpeningProofVariable<C>,
    #[allow(clippy::type_complexity)]
    pub exposed_values_after_challenge: Array<C, Array<C, Array<C, Ext<C::F, C::EF>>>>,
}

#[derive(DslVariable, Clone)]
pub struct OpeningProofVariable<C: Config> {
    pub proof: TwoAdicPcsProofVariable<C>,
    pub values: OpenedValuesVariable<C>,
}

#[allow(clippy::type_complexity)]
#[derive(DslVariable, Clone)]
pub struct OpenedValuesVariable<C: Config> {
    pub preprocessed: Array<C, AdjacentOpenedValuesVariable<C>>,
    pub main: Array<C, Array<C, AdjacentOpenedValuesVariable<C>>>,
    pub after_challenge: Array<C, Array<C, AdjacentOpenedValuesVariable<C>>>,
    pub quotient: Array<C, Array<C, Array<C, Ext<C::F, C::EF>>>>,
}

#[derive(DslVariable, Debug, Clone)]
pub struct AdjacentOpenedValuesVariable<C: Config> {
    pub local: Array<C, Ext<C::F, C::EF>>,
    pub next: Array<C, Ext<C::F, C::EF>>,
}

pub struct VerifierSinglePreprocessedDataInProgram<C: Config> {
    pub commit: Vec<C::F>,
}

/// Constants determined by AIRs.
pub struct StarkVerificationAdvice<C: Config> {
    /// Preprocessed trace data, if any
    pub preprocessed_data: Option<VerifierSinglePreprocessedDataInProgram<C>>,
    /// Trace sub-matrix widths
    pub width: TraceWidth,
    /// [MatrixCommitmentPointers] for partitioned main trace matrix
    pub main_graph: MatrixCommitmentPointers,
    /// The factor to multiple the trace degree by to get the degree of the quotient polynomial. Determined from the max constraint degree of the AIR constraints.
    /// This is equivalently the number of chunks the quotient polynomial is split into.
    pub quotient_degree: usize,
    /// Number of public values for this STARK only
    pub num_public_values: usize,
    /// For only this RAP, how many challenges are needed in each trace challenge phase
    pub num_challenges_to_sample: Vec<usize>,
    /// Number of values to expose to verifier in each trace challenge phase
    pub num_exposed_values_after_challenge: Vec<usize>,
    /// Symbolic representation of all AIR constraints, including logup constraints
    pub symbolic_constraints: SymbolicConstraints<C::F>,
    /// TODO: remove this once dyn Rap is no longer necessary
    pub(crate) interaction_chunk_size: usize,
}

// TODO: the bound C::F = Val<SC> is very awkward
pub(crate) fn new_from_vk<
    SC: StarkGenericConfig,
    C: Config<F = Val<SC>>,
    const DIGEST_SIZE: usize,
>(
    vk: StarkVerifyingKey<SC>,
) -> StarkVerificationAdvice<C>
where
    Com<SC>: Into<[C::F; DIGEST_SIZE]>,
{
    let StarkVerifyingKey::<SC> {
        preprocessed_data,
        params,
        main_graph,
        quotient_degree,
        symbolic_constraints,
        interaction_chunk_size,
        ..
    } = vk;
    StarkVerificationAdvice {
        preprocessed_data: preprocessed_data.map(|data| VerifierSinglePreprocessedDataInProgram {
            commit: data.commit.clone().into().to_vec(),
        }),
        width: params.width,
        main_graph,
        quotient_degree,
        num_public_values: params.num_public_values,
        num_challenges_to_sample: params.num_challenges_to_sample,
        num_exposed_values_after_challenge: params.num_exposed_values_after_challenge,
        symbolic_constraints,
        interaction_chunk_size,
    }
}

impl<C: Config> StarkVerificationAdvice<C> {
    pub fn log_quotient_degree(&self) -> usize {
        log2_strict_usize(self.quotient_degree)
    }
}

/// Constants determined by multiple AIRs.
pub struct MultiStarkVerificationAdvice<C: Config> {
    pub per_air: Vec<StarkVerificationAdvice<C>>,
    /// Number of multi-matrix commitments that hold commitments to the partitioned main trace matrices across all AIRs.
    pub num_main_trace_commitments: usize,
    /// Mapping from commit_idx to global AIR index for matrix in commitment, in order.
    pub main_commit_to_air_graph: CommitmentToAirGraph,
    /// The number of challenges to sample in each challenge phase.
    /// The length determines the global number of challenge phases.
    pub num_challenges_to_sample: Vec<usize>,
}

// TODO: the bound C::F = Val<SC> is very awkward
pub fn new_from_multi_vk<SC: StarkGenericConfig, C: Config<F = Val<SC>>, const DIGEST_SIZE: usize>(
    vk: &MultiStarkVerifyingKey<SC>,
) -> MultiStarkVerificationAdvice<C>
where
    Com<SC>: Into<[C::F; DIGEST_SIZE]>,
{
    let MultiStarkVerifyingKey::<SC> {
        per_air,
        num_main_trace_commitments,
        main_commit_to_air_graph,
        num_challenges_to_sample,
        // TODO: add support for interaction_chunk_size
        ..
    } = vk;
    MultiStarkVerificationAdvice {
        per_air: per_air.clone().into_iter().map(new_from_vk).collect(),
        num_main_trace_commitments: *num_main_trace_commitments,
        main_commit_to_air_graph: main_commit_to_air_graph.clone(),
        num_challenges_to_sample: num_challenges_to_sample.clone(),
    }
}
