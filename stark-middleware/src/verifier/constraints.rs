use itertools::Itertools;
use p3_commit::PolynomialSpace;
use p3_field::{AbstractExtensionField, AbstractField, Field};
use p3_matrix::dense::RowMajorMatrixView;
use p3_matrix::stack::VerticalPair;
use p3_uni_stark::Domain;
use p3_uni_stark::StarkGenericConfig;
use p3_uni_stark::Val;
use tracing::instrument;

use crate::air_builders::verifier::VerifierConstraintFolder;
use crate::prover::opener::AdjacentOpenedValues;
use crate::rap::Rap;

use super::error::VerificationError;

#[allow(clippy::too_many_arguments)]
#[instrument(skip_all)]
pub fn verify_single_rap_constraints<SC, R>(
    rap: &R,
    main_values: &AdjacentOpenedValues<SC::Challenge>,
    perm_values: Option<&AdjacentOpenedValues<SC::Challenge>>,
    quotient_chunks: &[Vec<SC::Challenge>],
    main_domain: Domain<SC>,
    qc_domains: &[Domain<SC>],
    zeta: SC::Challenge,
    alpha: SC::Challenge,
    perm_challenges: &[SC::Challenge],
    public_values: &[Val<SC>],
    perm_exposed_values: &[SC::Challenge],
) -> Result<(), VerificationError>
where
    SC: StarkGenericConfig,
    R: for<'b> Rap<VerifierConstraintFolder<'b, SC>> + ?Sized,
{
    let zps = qc_domains
        .iter()
        .enumerate()
        .map(|(i, domain)| {
            qc_domains
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other_domain)| {
                    other_domain.zp_at_point(zeta)
                        * other_domain.zp_at_point(domain.first_point()).inverse()
                })
                .product::<SC::Challenge>()
        })
        .collect_vec();

    let quotient = quotient_chunks
        .iter()
        .enumerate()
        .map(|(ch_i, ch)| {
            ch.iter()
                .enumerate()
                .map(|(e_i, &c)| zps[ch_i] * SC::Challenge::monomial(e_i) * c)
                .sum::<SC::Challenge>()
        })
        .sum::<SC::Challenge>();

    let unflatten = |v: &[SC::Challenge]| {
        v.chunks_exact(SC::Challenge::D)
            .map(|chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .map(|(e_i, &c)| SC::Challenge::monomial(e_i) * c)
                    .sum()
            })
            .collect::<Vec<SC::Challenge>>()
    };

    let sels = main_domain.selectors_at_point(zeta);

    let main = VerticalPair::new(
        RowMajorMatrixView::new_row(&main_values.local),
        RowMajorMatrixView::new_row(&main_values.next),
    );
    let (perm_local, perm_next) = perm_values
        .as_ref()
        .map(|values| (unflatten(&values.local), unflatten(&values.next)))
        .unwrap_or((vec![], vec![]));
    let perm = VerticalPair::new(
        RowMajorMatrixView::new_row(&perm_local),
        RowMajorMatrixView::new_row(&perm_next),
    );

    let mut folder: VerifierConstraintFolder<'_, SC> = VerifierConstraintFolder {
        preprocessed: VerticalPair::new(
            RowMajorMatrixView::new(&[], 0),
            RowMajorMatrixView::new(&[], 0),
        ),
        main,
        perm,
        is_first_row: sels.is_first_row,
        is_last_row: sels.is_last_row,
        is_transition: sels.is_transition,
        alpha,
        accumulator: SC::Challenge::zero(),
        perm_challenges,
        public_values,
        perm_exposed_values,
    };
    rap.eval(&mut folder);

    let folded_constraints = folder.accumulator;
    // Finally, check that
    //     folded_constraints(zeta) / Z_H(zeta) = quotient(zeta)
    if folded_constraints * sels.inv_zeroifier != quotient {
        return Err(VerificationError::OodEvaluationMismatch);
    }

    Ok(())
}
