use axvm_native_compiler::prelude::*;

use crate::{digest::DigestVariable, fri::TwoAdicMultiplicativeCosetVariable};

#[derive(Clone)]
pub struct FriConfigVariable<C: Config> {
    pub log_blowup: usize,
    pub blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub generators: Array<C, Felt<C::F>>,
    pub subgroups: Array<C, TwoAdicMultiplicativeCosetVariable<C>>,
}

impl<C: Config> FriConfigVariable<C> {
    pub fn get_subgroup(
        &self,
        builder: &mut Builder<C>,
        log_degree: impl Into<RVar<C::N>>,
    ) -> TwoAdicMultiplicativeCosetVariable<C> {
        builder.get(&self.subgroups, log_degree)
    }

    pub fn get_two_adic_generator(
        &self,
        builder: &mut Builder<C>,
        bits: impl Into<RVar<C::N>>,
    ) -> Felt<C::F> {
        builder.get(&self.generators, bits)
    }
}

#[derive(DslVariable, Clone)]
pub struct FriProofVariable<C: Config> {
    pub commit_phase_commits: Array<C, DigestVariable<C>>,
    pub query_proofs: Array<C, FriQueryProofVariable<C>>,
    pub final_poly: Ext<C::F, C::EF>,
    pub pow_witness: Felt<C::F>,
}

#[derive(DslVariable, Clone)]
pub struct FriQueryProofVariable<C: Config> {
    pub commit_phase_openings: Array<C, FriCommitPhaseProofStepVariable<C>>,
}

#[derive(DslVariable, Clone)]
pub struct FriCommitPhaseProofStepVariable<C: Config> {
    pub sibling_value: Ext<C::F, C::EF>,
    pub opening_proof: Array<C, DigestVariable<C>>,
}

#[derive(DslVariable, Clone)]
pub struct FriChallengesVariable<C: Config> {
    pub query_indices: Array<C, Array<C, Var<C::N>>>,
    pub betas: Array<C, Ext<C::F, C::EF>>,
}

#[derive(DslVariable, Clone)]
pub struct DimensionsVariable<C: Config> {
    pub height: Usize<C::N>,
}

#[derive(DslVariable, Clone)]
pub struct TwoAdicPcsProofVariable<C: Config> {
    pub fri_proof: FriProofVariable<C>,
    pub query_openings: Array<C, Array<C, BatchOpeningVariable<C>>>,
}

#[derive(DslVariable, Clone)]
pub struct BatchOpeningVariable<C: Config> {
    #[allow(clippy::type_complexity)]
    pub opened_values: Array<C, Array<C, Felt<C::F>>>,
    pub opening_proof: Array<C, DigestVariable<C>>,
}

#[derive(DslVariable, Clone)]
pub struct TwoAdicPcsRoundVariable<C: Config> {
    pub batch_commit: DigestVariable<C>,
    pub mats: Array<C, TwoAdicPcsMatsVariable<C>>,
    /// Optional. `permutation` could be empty if `mats` is already sorted. Otherwise, it's a
    /// permutation of indexes of mats which domains are sorted by degree in descending order.
    pub permutation: Array<C, Usize<C::N>>,
}

#[derive(DslVariable, Clone)]
pub struct TwoAdicPcsMatsVariable<C: Config> {
    pub domain: TwoAdicMultiplicativeCosetVariable<C>,
    pub points: Array<C, Ext<C::F, C::EF>>,
    #[allow(clippy::type_complexity)]
    pub values: Array<C, Array<C, Ext<C::F, C::EF>>>,
}
