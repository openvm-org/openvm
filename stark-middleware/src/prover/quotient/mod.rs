use std::cmp::min;

use itertools::Itertools;
use p3_air::Air;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, AbstractField, PackedValue};
use p3_matrix::{
    dense::{RowMajorMatrix, RowMajorMatrixView},
    stack::VerticalPair,
    Matrix,
};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, PackedChallenge, PackedVal, StarkGenericConfig, Val};
use p3_util::log2_strict_usize;
use tracing::instrument;

use crate::{
    air_builders::prover::ProverConstraintFolder,
    config::PcsProverData,
    prover::types::{ProvenMultiMatrixAirTrace, ProverQuotientData},
};

pub struct QuotientCommitter<'pcs, SC: StarkGenericConfig> {
    pcs: &'pcs SC::Pcs,
    alpha: SC::Challenge,
}

impl<'pcs, SC: StarkGenericConfig> QuotientCommitter<'pcs, SC> {
    pub fn new(pcs: &'pcs SC::Pcs, alpha: SC::Challenge) -> Self {
        Self { pcs, alpha }
    }

    /// Constructs quotient domains and computes the evaluation of the quotient polynomials
    /// on the quotient domains of each AIR.
    ///
    /// ## Assumptions
    /// - `quotient_degrees` is in the same order as AIRs in `proven.airs`.
    ///   It is the factor to **multiply** the trace degree by to get the degree
    ///   of the quotient polynomial. This should be determined from the constraint degree
    ///   of the AIR.
    #[instrument(name = "compute quotient values", skip_all)]
    pub fn compute_quotient_values<'a>(
        &self,
        proven: ProvenMultiMatrixAirTrace<'a, SC>,
        quotient_degrees: Vec<usize>,
        public_values: &'a [Val<SC>],
    ) -> MultiMatrixAirQuotientData<SC>
    where
        Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        PcsProverData<SC>: Sync,
    {
        let traces_with_domains = &proven.trace_data.traces_with_domains;
        let prover_data = &proven.trace_data.data;

        let inner = traces_with_domains
            .par_iter()
            .zip_eq(proven.airs.into_par_iter())
            .zip_eq(quotient_degrees.into_par_iter())
            .enumerate()
            .map(|(i, (((trace_domain, _), air), quotient_degree))| {
                let quotient_domain =
                    trace_domain.create_disjoint_domain(trace_domain.size() * quotient_degree);
                let main_trace_on_quotient_domain = self
                    .pcs
                    .get_evaluations_on_domain(prover_data, i, quotient_domain)
                    .to_row_major_matrix();
                let trace_domain = traces_with_domains[i].0;
                let quotient_values = compute_single_air_quotient_values(
                    air,
                    trace_domain,
                    quotient_domain,
                    main_trace_on_quotient_domain,
                    self.alpha,
                    public_values,
                );
                SingleAirQuotientData {
                    quotient_degree,
                    quotient_domain,
                    quotient_values,
                }
            })
            .collect();
        MultiMatrixAirQuotientData { inner }
    }

    // TODO: not sure this is the right function signature
    #[instrument(name = "commit to quotient poly chunks", skip_all)]
    pub fn commit(&self, data: MultiMatrixAirQuotientData<SC>) -> ProverQuotientData<SC> {
        let quotient_degrees = data.inner.iter().map(|d| d.quotient_degree).collect();
        let quotient_domains_and_chunks = data
            .split()
            .into_iter()
            .map(|q| (q.domain, q.chunk))
            .collect();
        let (commit, data) = self.pcs.commit(quotient_domains_and_chunks);
        ProverQuotientData {
            quotient_degrees,
            commit,
            data,
        }
    }
}

/// The quotient polynomials from multiple AIRs, kept in the same order as the
/// proven trace data.
pub struct MultiMatrixAirQuotientData<SC: StarkGenericConfig> {
    inner: Vec<SingleAirQuotientData<SC>>,
}

impl<SC: StarkGenericConfig> MultiMatrixAirQuotientData<SC> {
    /// Splits the quotient polynomials from multiple AIRs into chunks of size equal to the trace domain size.
    pub fn split(self) -> impl IntoIterator<Item = QuotientChunk<SC>> {
        self.inner.into_iter().flat_map(|data| data.split())
    }
}

/// The quotient polynomial from a single matrix AIR, evaluated on the quotient domain.
pub struct SingleAirQuotientData<SC: StarkGenericConfig> {
    /// The factor by which the trace degree was multiplied to get the
    /// quotient domain size.
    quotient_degree: usize,
    /// Quotient domain
    quotient_domain: Domain<SC>,
    /// Evaluations of the quotient polynomial on the quotient domain
    quotient_values: Vec<SC::Challenge>,
}

impl<SC: StarkGenericConfig> SingleAirQuotientData<SC> {
    /// The vector of evaluations of the quotient polynomial on the quotient domain,
    /// first flattened from vector of extension field elements to matrix of base field elements,
    /// and then split into chunks of size equal to the trace domain size (quotient domain size
    /// divided by `quotient_degree`).
    pub fn split(self) -> impl IntoIterator<Item = QuotientChunk<SC>> {
        let quotient_degree = self.quotient_degree;
        let quotient_domain = self.quotient_domain;
        // Flatten from extension field elements to base field elements
        let quotient_flat = RowMajorMatrix::new_col(self.quotient_values).flatten_to_base();
        let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);
        let qc_domains = quotient_domain.split_domains(quotient_degree);
        qc_domains
            .into_iter()
            .zip_eq(quotient_chunks)
            .map(|(domain, chunk)| QuotientChunk { domain, chunk })
    }
}

/// The vector of evaluations of the quotient polynomial on the quotient domain,
/// split into chunks of size equal to the trace domain size (quotient domain size
/// divided by `quotient_degree`).
///
/// This represents a single chunk, where the vector of extension field elements is
/// further flattened to a matrix of base field elements.
pub struct QuotientChunk<SC: StarkGenericConfig> {
    /// Chunk of quotient domain, which is a coset of the trace domain
    pub domain: Domain<SC>,
    /// Matrix with number of rows equal to trace domain size,
    /// and number of columns equal to extension field degree.
    pub chunk: RowMajorMatrix<Val<SC>>,
}

// Starting reference: p3_uni_stark::prover::quotient_values
/// Computes evaluation of DEEP quotient polynomial on the quotient domain for a single AIR (single trace matrix).
#[instrument(name = "compute single AIR quotient polynomial", skip_all)]
fn compute_single_air_quotient_values<'a, SC, A, Mat>(
    air: &'a A,
    // cumulative_sum: SC::Challenge,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    // preprocessed_trace_on_quotient_domain: Mat, // TODO: add back in
    main_trace_on_quotient_domain: Mat,
    // perm_trace_on_quotient_domain: Mat, // TODO: add back in
    // perm_challenges: &[PackedChallenge<SC>], // TODO: add back in
    alpha: SC::Challenge,
    public_values: &'a [Val<SC>],
) -> Vec<SC::Challenge>
where
    A: for<'b> Air<ProverConstraintFolder<'b, SC>> + ?Sized,
    SC: StarkGenericConfig,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    // let preprocessed_width = preprocessed_trace_on_quotient_domain.width();
    let main_width = main_trace_on_quotient_domain.width();
    // let perm_width = perm_trace_on_quotient_domain.width();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_size) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

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

            // let prep_local: Vec<_> = (0..prep_width)
            //     .map(|col| {
            //         PackedVal::<SC>::from_fn(|offset| {
            //             preprocessed_trace_on_quotient_domain.get(wrap(i_start + offset), col)
            //         })
            //     })
            //     .collect();
            // let prep_next: Vec<_> = (0..prep_width)
            //     .map(|col| {
            //         PackedVal::<SC>::from_fn(|offset| {
            //             preprocessed_trace_on_quotient_domain
            //                 .get(wrap(i_start + next_step + offset), col)
            //         })
            //     })
            //     .collect();

            let local: Vec<_> = (0..main_width)
                .map(|col| {
                    PackedVal::<SC>::from_fn(|offset| {
                        main_trace_on_quotient_domain.get(wrap(i_start + offset), col)
                    })
                })
                .collect();
            let next: Vec<_> = (0..main_width)
                .map(|col| {
                    PackedVal::<SC>::from_fn(|offset| {
                        main_trace_on_quotient_domain.get(wrap(i_start + next_step + offset), col)
                    })
                })
                .collect();

            // let perm_local: Vec<_> = (0..perm_width)
            //     .step_by(ext_degree)
            //     .map(|col| {
            //         PackedChallenge::<SC>::from_base_fn(|i| {
            //             PackedVal::<SC>::from_fn(|offset| {
            //                 permutation_trace_on_quotient_domain
            //                     .get(wrap(i_start + offset), col + i)
            //             })
            //         })
            //     })
            //     .collect();

            // let perm_next: Vec<_> = (0..perm_width)
            //     .step_by(ext_degree)
            //     .map(|col| {
            //         PackedChallenge::<SC>::from_base_fn(|i| {
            //             PackedVal::<SC>::from_fn(|offset| {
            //                 permutation_trace_on_quotient_domain
            //                     .get(wrap(i_start + next_step + offset), col + i)
            //             })
            //         })
            //     })
            //     .collect();

            let accumulator = PackedChallenge::<SC>::zero();
            let mut folder = ProverConstraintFolder {
                // preprocessed: VerticalPair::new(
                //     RowMajorMatrixView::new_row(&prep_local),
                //     RowMajorMatrixView::new_row(&prep_next),
                // ),
                main: VerticalPair::new(
                    RowMajorMatrixView::new_row(&local),
                    RowMajorMatrixView::new_row(&next),
                ),
                // perm: VerticalPair::new(
                //     RowMajorMatrixView::new_row(&perm_local),
                //     RowMajorMatrixView::new_row(&perm_next),
                // ),
                // perm_challenges,
                // cumulative_sum,
                is_first_row,
                is_last_row,
                is_transition,
                alpha,
                accumulator,
                public_values,
            };
            air.eval(&mut folder);

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
