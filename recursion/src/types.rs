use p3_field::{AbstractField, PrimeField32, TwoAdicField};
use p3_uni_stark::{StarkGenericConfig, Val};

use afs_compiler::asm::AsmConfig;
use afs_compiler::ir::{Array, Builder, Config, Ext, ExtConst, Felt, Var};
use afs_compiler::prelude::*;
use afs_stark_backend::keygen::types::MultiStarkPartialVerifyingKey;
use afs_stark_backend::prover::types::Proof;

use crate::challenger::DuplexChallengerVariable;
use crate::fri::TwoAdicFriPcsVariable;
use crate::fri::types::{DigestVariable, TwoAdicPcsProofVariable};
use crate::hints::{InnerChallenge, InnerVal};
use crate::stark::{AxiomStarkVerifier, AxiomVerifier, DynRapForRecursion};

pub type InnerConfig = AsmConfig<InnerVal, InnerChallenge>;

/// The maximum number of elements that can be stored in the public values vec.  Both SP1 and recursive
/// proofs need to pad their public_values vec to this length.  This is required since the recursion
/// verification program expects the public values vec to be fixed length.
pub const PROOF_MAX_NUM_PVS: usize = 240;

impl<C: Config> AxiomVerifier<C>
where
    C::F: PrimeField32 + TwoAdicField,
{
    /// Reference: [afs_stark_backend::verifier::MultiTraceStarkVerifier::verify].
    pub fn verify(
        builder: &mut Builder<C>,
        pcs: &TwoAdicFriPcsVariable<C>,
        raps: Vec<&dyn DynRapForRecursion<C>>,
        chip_dims: Vec<ChipDimensions>,
        input: &AxiomMemoryLayoutVariable<C>,
    ) {
        let proof = &input.proof;

        let cumulative_sum: Ext<C::F, C::EF> = builder.eval(C::F::zero());
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
        builder.assert_ext_eq(cumulative_sum, C::EF::zero().cons());

        let mut challenger = DuplexChallengerVariable::new(builder);

        AxiomStarkVerifier::<C>::verify_raps(
            builder,
            pcs,
            raps,
            chip_dims,
            &mut challenger,
            &input.proof,
            &input.vk,
            input.public_values.clone(),
        );

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

pub struct AxiomMemoryLayout<SC: StarkGenericConfig> {
    pub proof: Proof<SC>,
    pub vk: MultiStarkPartialVerifyingKey<SC>,
    pub public_values: Vec<Vec<Val<SC>>>,
}

#[derive(DslVariable, Clone)]
pub struct AxiomMemoryLayoutVariable<C: Config> {
    pub proof: AxiomProofVariable<C>,
    pub vk: MultiStarkPartialVerifyingKeyVariable<C>,
    pub public_values: Array<C, Array<C, Felt<C::F>>>,
}

#[derive(DslVariable, Clone)]
pub struct MultiStarkPartialVerifyingKeyVariable<C: Config> {
    pub per_air: Array<C, StarkPartialVerifyingKeyVariable<C>>,
    pub num_challenges_to_sample: Array<C, Var<C::N>>,
    pub num_main_trace_commitments: Var<C::N>,
}

#[derive(DslVariable, Clone)]
pub struct StarkPartialVerifyingKeyVariable<C: Config> {
    pub log_degree: Var<C::N>,
    pub degree: Var<C::N>,
    pub log_quotient_degree: Var<C::N>,
    pub quotient_degree: Var<C::N>,
    pub width: TraceWidthVariable<C>,
    pub num_exposed_values_after_challenge: Array<C, Var<C::N>>,
}

#[derive(DslVariable, Clone)]
pub struct TraceWidthVariable<C: Config> {
    pub preprocessed: Array<C, Var<C::N>>,
    pub partitioned_main: Array<C, Var<C::N>>,
    pub after_challenge: Array<C, Var<C::N>>,
}

#[derive(DslVariable, Clone)]
pub struct AxiomCommitmentsVariable<C: Config> {
    pub main_trace: Array<C, DigestVariable<C>>,
    pub after_challenge: Array<C, DigestVariable<C>>,
    pub quotient: DigestVariable<C>,
}

#[derive(DslVariable, Clone)]
pub struct AxiomProofVariable<C: Config> {
    pub commitments: AxiomCommitmentsVariable<C>,
    pub opening: OpeningProofVariable<C>,
    pub exposed_values_after_challenge: Array<C, Array<C, Array<C, Ext<C::F, C::EF>>>>,
}

#[derive(DslVariable, Clone)]
pub struct OpeningProofVariable<C: Config> {
    pub proof: TwoAdicPcsProofVariable<C>,
    pub values: OpenedValuesVariable<C>,
}

#[derive(DslVariable, Clone)]
pub struct OpenedValuesVariable<C: Config> {
    // pub preprocessed: Array<C, AirOpenedValuesVariable<C>>,
    pub main: Array<C, Array<C, AdjacentOpenedValuesVariable<C>>>,
    pub after_challenge: Array<C, Array<C, AdjacentOpenedValuesVariable<C>>>,
    pub quotient: Array<C, Array<C, Array<C, Ext<C::F, C::EF>>>>,
}

#[derive(DslVariable, Debug, Clone)]
pub struct AdjacentOpenedValuesVariable<C: Config> {
    pub local: Array<C, Ext<C::F, C::EF>>,
    pub next: Array<C, Ext<C::F, C::EF>>,
}

#[derive(Debug, Clone, Copy)]
pub struct ChipDimensions {
    pub preprocessed_width: usize,
    pub main_width: usize,
    pub permutation_width: usize,
    pub log_quotient_degree: usize,
}
