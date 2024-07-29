use afs_stark_backend::air_builders::symbolic::SymbolicConstraints;
use p3_field::{AbstractField, PrimeField32, TwoAdicField};
use p3_uni_stark::{StarkGenericConfig, Val};
use p3_util::log2_strict_usize;

use afs_compiler::asm::AsmConfig;
use afs_compiler::ir::{Array, Builder, Config, Ext, ExtConst, Felt, Var};
use afs_compiler::prelude::*;
use afs_stark_backend::commit::MatrixCommitmentPointers;
use afs_stark_backend::config::Com;
use afs_stark_backend::keygen::types::{
    CommitmentToAirGraph, MultiStarkVerifyingKey, StarkVerifyingKey, TraceWidth,
};
use afs_stark_backend::prover::types::Proof;

use crate::challenger::DuplexChallengerVariable;
use crate::fri::types::{DigestVariable, TwoAdicPcsProofVariable};
use crate::fri::TwoAdicFriPcsVariable;
use crate::hints::{InnerChallenge, InnerVal};
use crate::stark::{DynRapForRecursion, StarkVerifier, VerifierProgram};

pub type InnerConfig = AsmConfig<InnerVal, InnerChallenge>;

/// The maximum number of elements that can be stored in the public values vec.  Both SP1 and recursive
/// proofs need to pad their public_values vec to this length.  This is required since the recursion
/// verification program expects the public values vec to be fixed length.
pub const PROOF_MAX_NUM_PVS: usize = 240;

impl<C: Config> VerifierProgram<C>
where
    C::F: PrimeField32 + TwoAdicField,
{
    /// Reference: [afs_stark_backend::verifier::MultiTraceStarkVerifier::verify].
    pub fn verify(
        builder: &mut Builder<C>,
        pcs: &TwoAdicFriPcsVariable<C>,
        raps: Vec<&dyn DynRapForRecursion<C>>,
        constants: MultiStarkVerificationAdvice<C>,
        input: &VerifierProgramInputVariable<C>,
    ) {
        let proof = &input.proof;

        let cumulative_sum: Ext<C::F, C::EF> = builder.eval(C::F::zero());
        let num_phases = constants.num_challenges_to_sample.len();
        // Currently only support 0 or 1 phase is supported.
        assert!(num_phases <= 1);
        // Tmp solution to support 0 or 1 phase.
        if num_phases > 0 {
            builder
                .range(0, proof.exposed_values_after_challenge.len())
                .for_each(|i, builder| {
                    let exposed_values = builder.get(&proof.exposed_values_after_challenge, i);

                    // Verifier does not support more than 1 challenge phase
                    builder.assert_usize_eq(exposed_values.len(), 1);

                    let values = builder.get(&exposed_values, 0);

                    // Only exposed value should be cumulative sum
                    builder.assert_usize_eq(values.len(), 1);

                    let summand = builder.get(&values, 0);
                    builder.assign(cumulative_sum, cumulative_sum + summand);
                });
        }
        builder.assert_ext_eq(cumulative_sum, C::EF::zero().cons());

        let mut challenger = DuplexChallengerVariable::new(builder);

        StarkVerifier::<C>::verify_raps(builder, pcs, raps, constants, &mut challenger, input);

        builder.halt();

        // TODO: bind public inputs
        // Get the public inputs from the proof.
        // let public_values_elements = (0..RECURSIVE_PROOF_NUM_PV_ELTS)
        //     .map(|i| builder.get(&input.proof.public_values, i))
        //     .collect::<Vec<Felt<_>>>();
        // let public_values: &RecursionPublicValues<Felt<C::F>> =
        //     public_values_elements.as_slice().borrow();

        // Check that the public values digest is correct.
        // verify_public_values_hash(builder, public_values);

        // Assert that the proof is complete.
        //
        // *Remark*: here we are assuming on that the program we are verifying indludes the check
        // of completeness conditions are satisfied if the flag is set to one, so we are only
        // checking the `is_complete` flag in this program.
        // builder.assert_felt_eq(public_values.is_complete, C::F::one());

        // If the proof is a compress proof, assert that the vk is the same as the compress vk from
        // the public values.
        // if is_compress {
        //     let vk_digest = hash_vkey(builder, &vk);
        //     for (i, reduce_digest_elem) in public_values.compress_vk_digest.iter().enumerate() {
        //         let vk_digest_elem = builder.get(&vk_digest, i);
        //         builder.assert_felt_eq(vk_digest_elem, *reduce_digest_elem);
        //     }
        // }

        // commit_public_values(builder, public_values);
    }
}

pub struct VerifierProgramInput<SC: StarkGenericConfig> {
    pub proof: Proof<SC>,
    pub log_degree_per_air: Vec<usize>,
    pub public_values: Vec<Vec<Val<SC>>>,
}

#[derive(DslVariable, Clone)]
pub struct VerifierProgramInputVariable<C: Config> {
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
    } = vk;
    MultiStarkVerificationAdvice {
        per_air: per_air.clone().into_iter().map(new_from_vk).collect(),
        num_main_trace_commitments: *num_main_trace_commitments,
        main_commit_to_air_graph: main_commit_to_air_graph.clone(),
        num_challenges_to_sample: num_challenges_to_sample.clone(),
    }
}
