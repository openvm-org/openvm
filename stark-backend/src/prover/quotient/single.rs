use std::cmp::min;

use itertools::Itertools;
use p3_commit::PolynomialSpace;
use p3_field::{AbstractExtensionField, AbstractField, PackedValue};
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, PackedChallenge, PackedVal, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    air_builders::{prover::ProverConstraintFolder, symbolic::SymbolicConstraints},
    rap::{PartitionedBaseAir, Rap},
};

// Starting reference: p3_uni_stark::prover::quotient_values
// TODO: make this into a trait that is auto-implemented so we can dynamic dispatch the trait
/// Computes evaluation of DEEP quotient polynomial on the quotient domain for a single RAP (single trace matrix).
///
/// Designed to be general enough to support RAP with multiple rounds of challenges.
#[allow(clippy::too_many_arguments)]
#[instrument(name = "compute single RAP quotient polynomial", skip_all)]
pub fn compute_single_rap_quotient_values<'a, SC, R, Mat>(
    rap: &'a R,
    symbolic_constraints: &SymbolicConstraints<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_trace_on_quotient_domain: Mat,
    partitioned_main_lde_on_quotient_domain: Vec<Mat>,
    after_challenge_lde_on_quotient_domain: Vec<Mat>,
    // For each challenge round, the challenges drawn
    challenges: &[Vec<PackedChallenge<SC>>],
    alpha: SC::Challenge,
    public_values: &'a [Val<SC>],
    // Values exposed to verifier after challenge round i
    exposed_values_after_challenge: &'a [&'a [PackedChallenge<SC>]],
    interaction_chunk_size: usize,
) -> Vec<SC::Challenge>
where
    // TODO: avoid ?Sized to prevent dynamic dispatching because `eval` is called many many times
    R: for<'b> Rap<ProverConstraintFolder<'b, SC>> + PartitionedBaseAir<Val<SC>> + Sync + ?Sized,
    SC: StarkGenericConfig,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let preprocessed_width = preprocessed_trace_on_quotient_domain.width();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_size) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    let ext_degree = SC::Challenge::D;

    let mut alpha_powers = alpha
        .powers()
        .take(symbolic_constraints.constraints.len())
        .collect_vec();
    alpha_powers.reverse();

    // assert!(quotient_size >= PackedVal::<SC>::WIDTH);
    // We take PackedVal::<SC>::WIDTH worth of values at a time from a quotient_size slice, so we need to
    // pad with default values in the case where quotient_size is smaller than PackedVal::<SC>::WIDTH.
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_zeroifier.push(Val::<SC>::default());
    }

    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let wrap = |i| i % quotient_size;
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_zeroifier = *PackedVal::<SC>::from_slice(&sels.inv_zeroifier[i_range.clone()]);

            let [preprocessed_local, preprocessed_next] = [0, 1].map(|step_idx| {
                (0..preprocessed_width)
                    .map(|col| {
                        PackedVal::<SC>::from_fn(|offset| {
                            preprocessed_trace_on_quotient_domain
                                .get(wrap(i_start + offset + step_idx * next_step), col)
                        })
                    })
                    .collect_vec()
            });

            let partitioned_main_pairs = partitioned_main_lde_on_quotient_domain
                .iter()
                .map(|lde| {
                    let width = lde.width();
                    [0, 1].map(|step_idx| {
                        (0..width)
                            .map(|col| {
                                PackedVal::<SC>::from_fn(|offset| {
                                    lde.get(wrap(i_start + offset + step_idx * next_step), col)
                                })
                            })
                            .collect_vec()
                    })
                })
                .collect_vec();

            let after_challenge_pairs = after_challenge_lde_on_quotient_domain
                .iter()
                .map(|lde| {
                    // Width in base field with extension field elements flattened
                    let base_width = lde.width();
                    [0, 1].map(|step_idx| {
                        (0..base_width)
                            .step_by(ext_degree)
                            .map(|col| {
                                PackedChallenge::<SC>::from_base_fn(|i| {
                                    PackedVal::<SC>::from_fn(|offset| {
                                        lde.get(
                                            wrap(i_start + offset + step_idx * next_step),
                                            col + i,
                                        )
                                    })
                                })
                            })
                            .collect_vec()
                    })
                })
                .collect_vec();

            let accumulator = PackedChallenge::<SC>::ZERO;
            let mut folder = ProverConstraintFolder {
                preprocessed: VerticalPair::new(
                    RowMajorMatrixView::new_row(&preprocessed_local),
                    RowMajorMatrixView::new_row(&preprocessed_next),
                ),
                partitioned_main: partitioned_main_pairs
                    .iter()
                    .map(|[local, next]| {
                        VerticalPair::new(
                            RowMajorMatrixView::new_row(local),
                            RowMajorMatrixView::new_row(next),
                        )
                    })
                    .collect(),
                after_challenge: after_challenge_pairs
                    .iter()
                    .map(|[local, next]| {
                        VerticalPair::new(
                            RowMajorMatrixView::new_row(local),
                            RowMajorMatrixView::new_row(next),
                        )
                    })
                    .collect(),
                challenges,
                is_first_row,
                is_last_row,
                is_transition,
                alpha_powers: &alpha_powers,
                accumulator,
                public_values,
                exposed_values_after_challenge,
                interactions: vec![],
                interaction_chunk_size,
                has_common_main: rap.common_main_width() > 0,
                constraint_index: 0,
            };
            rap.eval(&mut folder);

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.accumulator * inv_zeroifier;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            let width = min(PackedVal::<SC>::WIDTH, quotient_size);
            (0..width).map(move |idx_in_packing| {
                let quotient_value = (0..<SC::Challenge as AbstractExtensionField<Val<SC>>>::D)
                    .map(|coeff_idx| quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing])
                    .collect::<Vec<_>>();
                SC::Challenge::from_base_slice(&quotient_value)
            })
        })
        .collect()
}
