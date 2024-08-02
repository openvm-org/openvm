use afs_compiler::ir::{Array, Builder, Config, Ext, ExtConst, Felt, SymbolicExt};
use afs_compiler::prelude::*;
use afs_derive::DslVariable;
use afs_stark_backend::air_builders::symbolic::SymbolicConstraints;
use afs_stark_backend::prover::opener::AdjacentOpenedValues;
use afs_stark_backend::rap::Rap;
use itertools::Itertools;
use p3_commit::LagrangeSelectors;
use p3_field::{AbstractExtensionField, AbstractField};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;

use crate::commit::PolynomialSpaceVariable;
use crate::folder::RecursiveVerifierConstraintFolder;
use crate::fri::TwoAdicMultiplicativeCosetVariable;
use crate::types::{
    AdjacentOpenedValuesVariable, CommitmentsVariable, OpeningProofVariable,
    StarkVerificationAdvice, PROOF_MAX_NUM_PVS,
};

pub struct RapVerifier<'a, C: Config, R>
where
    R: Rap<RecursiveVerifierConstraintFolder<'a, C>> + Sync + ?Sized,
{
    builder: &'a mut Builder<C>,
    rap: &'a R,
    constants: &'a StarkVerificationAdvice<C>,
    zeta: Ext<C::F, C::EF>,
    alpha: Ext<C::F, C::EF>,
    qc_domains: &'a [TwoAdicMultiplicativeCosetVariable<C>],
}

#[derive(DslVariable, Clone)]
pub struct RapProofVariable<C: Config> {
    pub preprocessed_values: Array<C, AdjacentOpenedValuesVariable<C>>,
    pub partitioned_main_values: Array<C, AdjacentOpenedValuesVariable<C>>,
    pub quotient_chunks: Array<C, Array<C, Ext<C::F, C::EF>>>,
    pub public_values: Array<C, Felt<C::F>>,
    pub trace_domain: TwoAdicMultiplicativeCosetVariable<C>,
    pub zeta: Ext<C::F, C::EF>,
    pub alpha: Ext<C::F, C::EF>,
    pub after_challenge_values: AdjacentOpenedValuesVariable<C>,
    pub challenges: Array<C, Array<C, Ext<C::F, C::EF>>>,
    pub exposed_values_after_challenge: Array<C, Array<C, Ext<C::F, C::EF>>>,
}

impl<'a, C: Config, R> RapVerifier<'a, C, R>
where
    R: Rap<RecursiveVerifierConstraintFolder<'a, C>> + Sync + ?Sized,
{
    pub fn new(
        builder: &'a mut Builder<C>,
        rap: &'a R,
        constants: &'a StarkVerificationAdvice<C>,
        zeta: Ext<C::F, C::EF>,
        alpha: Ext<C::F, C::EF>,
        qc_domains: &'a [TwoAdicMultiplicativeCosetVariable<C>],
    ) -> Self {
        Self {
            builder,
            rap,
            constants,
            zeta,
            alpha,
            qc_domains,
        }
    }

    pub fn verify(&mut self, proof: RapProofVariable<C>) {
        verify_single_rap_constraints(
            self.builder,
            self.rap,
            self.constants,
            proof,
            self.qc_domains,
        );
    }
} 

/// Reference: [afs_stark_backend::verifier::constraints::verify_single_rap_constraints]
/// This function is shared by both static verifier and VM verifier. So this function cannot
/// use `for`/`if` and memory allocation/access.
pub fn verify_single_rap_constraints<R, C: Config>(
    builder: &mut Builder<C>,
    rap: &R,
    constants: &StarkVerificationAdvice<C>,
    rap_proof: RapProofVariable<C>,
    qc_domains: &[TwoAdicMultiplicativeCosetVariable<C>],
) where
    R: for<'b> Rap<RecursiveVerifierConstraintFolder<'b, C>> + Sync + ?Sized,
{
    let RapProofVariable::<C> {
        preprocessed_values,
        partitioned_main_values,
        quotient_chunks,
        public_values,
        trace_domain,
        zeta,
        alpha,
        after_challenge_values,
        challenges,
        exposed_values_after_challenge,
    } = rap_proof;
    let sels = trace_domain.selectors_at_point(builder, zeta);

    let mut preprocessed = AdjacentOpenedValues {
        local: vec![],
        next: vec![],
    };
    if let Some(width) = constants.width.preprocessed {
        preprocessed_values.assert_len(builder, 1);
        let mut preprocessed_values = builder.get(&preprocessed_values, 0);
        for i in 0..width {
            preprocessed
                .local
                .push(builder.get(&preprocessed_values.local, i));
            preprocessed
                .next
                .push(builder.get(&preprocessed_values.next, i));
        }
    } else {
        preprocessed_values.assert_len(builder, 0);
    }

    partitioned_main_values.assert_len(builder, constants.width.partitioned_main.len());
    let partitioned_main_values = constants
        .width
        .partitioned_main
        .iter()
        .enumerate()
        .map(|(i, &width)| {
            let main_values = builder.get(&partitioned_main_values, i);
            builder.assert_usize_eq(main_values.local.len(), width);
            builder.assert_usize_eq(main_values.next.len(), width);
            let mut main = AdjacentOpenedValues {
                local: vec![],
                next: vec![],
            };
            for i in 0..width {
                main.local.push(builder.get(&main_values.local, i));
                main.next.push(builder.get(&main_values.next, i));
            }
            main
        })
        .collect_vec();

    let mut after_challenge = AdjacentOpenedValues {
        local: vec![],
        next: vec![],
    };

    let after_challenge_width = if constants.width.after_challenge.is_empty() {
        0
    } else {
        C::EF::D * constants.width.after_challenge[0]
    };
    after_challenge_values
        .local
        .assert_len(builder, after_challenge_width);
    after_challenge_values
        .next
        .assert_len(builder, after_challenge_width);
    for i in 0..after_challenge_width {
        after_challenge
            .local
            .push(builder.get(&after_challenge_values.local, i));
        after_challenge
            .next
            .push(builder.get(&after_challenge_values.next, i));
    }

    public_values.assert_len(builder, constants.num_public_values);
    let public_values = (0..constants.num_public_values).map(|i| builder.get(&public_values, i)).collect_vec();

    challenges.assert_len(builder, constants.num_challenges_to_sample.len());
    let challenges = constants
        .num_challenges_to_sample
        .iter()
        .enumerate()
        .map(|(i, &num)| {
            let phase_challenges = builder.get(&challenges, i);
            // Assumption: phase_challenges.len() >= num
            (0..num)
                .map(|j| builder.get(&phase_challenges, j))
                .collect_vec()
        })
        .collect_vec();
    exposed_values_after_challenge
        .assert_len(builder, constants.num_exposed_values_after_challenge.len());
    let exposed_values_after_challenge = constants
        .num_exposed_values_after_challenge
        .iter()
        .enumerate()
        .map(|(i, &num)| {
            let phase_values = builder.get(&exposed_values_after_challenge, i);
            // Assumption: phase_values.len() >= num
            (0..num)
                .map(|j| builder.get(&phase_values, j))
                .collect_vec()
        })
        .collect_vec();

    let folded_constraints = eval_constraints(
        builder,
        rap,
        &constants.symbolic_constraints,
        preprocessed,
        &partitioned_main_values,
        &public_values,
        &sels,
        alpha,
        after_challenge,
        &challenges,
        &exposed_values_after_challenge,
    );

    let num_quotient_chunks = 1 << constants.log_quotient_degree();
    let mut quotient = vec![];
    // Assert that the length of the quotient chunk arrays match the expected length.
    quotient_chunks.assert_len(builder, num_quotient_chunks);
    // Collect the quotient values into vectors.
    for i in 0..num_quotient_chunks {
        let chunk = builder.get(&quotient_chunks, i);
        // Assert that the chunk length matches the expected length.
        chunk.assert_len(builder, C::EF::D);
        // Collect the quotient values into vectors.
        let mut quotient_vals = vec![];
        for j in 0..C::EF::D {
            let value = builder.get(&chunk, j);
            quotient_vals.push(value);
        }
        quotient.push(quotient_vals);
    }

    let quotient: Ext<_, _> = recompute_quotient(builder, &quotient, qc_domains, zeta);

    // Assert that the quotient times the zerofier is equal to the folded constraints.
    builder.assert_ext_eq(folded_constraints * sels.inv_zeroifier, quotient);
}

#[allow(clippy::too_many_arguments)]
fn eval_constraints<R, C: Config>(
    builder: &mut Builder<C>,
    rap: &R,
    symbolic_constraints: &SymbolicConstraints<C::F>,
    preprocessed_values: AdjacentOpenedValues<Ext<C::F, C::EF>>,
    partitioned_main_values: &[AdjacentOpenedValues<Ext<C::F, C::EF>>],
    public_values: &[Felt<C::F>],
    selectors: &LagrangeSelectors<Ext<C::F, C::EF>>,
    alpha: Ext<C::F, C::EF>,
    after_challenge: AdjacentOpenedValues<Ext<C::F, C::EF>>,
    challenges: &[Vec<Ext<C::F, C::EF>>],
    exposed_values_after_challenge: &[Vec<Ext<C::F, C::EF>>],
) -> Ext<C::F, C::EF>
where
    R: for<'b> Rap<RecursiveVerifierConstraintFolder<'b, C>> + Sync + ?Sized,
{
    let mut unflatten = |v: &[Ext<C::F, C::EF>]| {
        v.chunks_exact(C::EF::D)
            .map(|chunk| {
                builder.eval(
                    chunk
                        .iter()
                        .enumerate()
                        .map(|(e_i, &x)| x * C::EF::monomial(e_i).cons())
                        .sum::<SymbolicExt<_, _>>(),
                )
            })
            .collect::<Vec<Ext<_, _>>>()
    };

    let after_challenge_values = AdjacentOpenedValues {
        local: unflatten(&after_challenge.local),
        next: unflatten(&after_challenge.next),
    };

    let mut folder = RecursiveVerifierConstraintFolder::<C> {
        preprocessed: VerticalPair::new(
            RowMajorMatrixView::new_row(&preprocessed_values.local),
            RowMajorMatrixView::new_row(&preprocessed_values.next),
        ),
        partitioned_main: partitioned_main_values
            .iter()
            .map(|main_values| {
                VerticalPair::new(
                    RowMajorMatrixView::new_row(&main_values.local),
                    RowMajorMatrixView::new_row(&main_values.next),
                )
            })
            .collect(),
        after_challenge: vec![VerticalPair::new(
            RowMajorMatrixView::new_row(&after_challenge_values.local),
            RowMajorMatrixView::new_row(&after_challenge_values.next),
        )],
        challenges,
        is_first_row: selectors.is_first_row,
        is_last_row: selectors.is_last_row,
        is_transition: selectors.is_transition,
        alpha,
        accumulator: SymbolicExt::zero(),
        public_values,
        exposed_values_after_challenge, // FIXME

        symbolic_interactions: &symbolic_constraints.interactions,
        interactions: vec![],
    };

    rap.eval(&mut folder);
    builder.eval(folder.accumulator)
}

fn recompute_quotient<C: Config>(
    builder: &mut Builder<C>,
    quotient_chunks: &[Vec<Ext<C::F, C::EF>>],
    qc_domains: &[TwoAdicMultiplicativeCosetVariable<C>],
    zeta: Ext<C::F, C::EF>,
) -> Ext<C::F, C::EF> {
    let zps = qc_domains
        .iter()
        .enumerate()
        .map(|(i, domain)| {
            qc_domains
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other_domain)| {
                    let first_point: Ext<_, _> = builder.eval(domain.first_point());
                    other_domain.zp_at_point(builder, zeta)
                        * other_domain.zp_at_point(builder, first_point).inverse()
                })
                .product::<SymbolicExt<_, _>>()
        })
        .collect::<Vec<SymbolicExt<_, _>>>()
        .into_iter()
        .map(|x| builder.eval(x))
        .collect::<Vec<Ext<_, _>>>();

    builder.eval(
        quotient_chunks
            .iter()
            .enumerate()
            .map(|(ch_i, ch)| {
                assert_eq!(ch.len(), C::EF::D);
                ch.iter()
                    .enumerate()
                    .map(|(e_i, &c)| zps[ch_i] * C::EF::monomial(e_i) * c)
                    .sum::<SymbolicExt<_, _>>()
            })
            .sum::<SymbolicExt<_, _>>(),
    )
}
