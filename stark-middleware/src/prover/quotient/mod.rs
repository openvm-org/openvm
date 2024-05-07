use std::cmp::min;

use itertools::Itertools;
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
    air_builders::prover::ProverConstraintFolder, config::PcsProverData,
    prover::types::ProverQuotientData, rap::Rap,
};

use super::{trace::ProvenSingleRapTraceView, types::ProverRap};

pub struct QuotientCommitter<'pcs, SC: StarkGenericConfig> {
    pcs: &'pcs SC::Pcs,
    perm_challenges: Vec<PackedChallenge<SC>>,
    alpha: SC::Challenge,
}

impl<'pcs, SC: StarkGenericConfig> QuotientCommitter<'pcs, SC> {
    pub fn new(
        pcs: &'pcs SC::Pcs,
        perm_challenges: &[SC::Challenge],
        alpha: SC::Challenge,
    ) -> Self {
        let packed_perm_challenges = perm_challenges
            .iter()
            .map(|c| PackedChallenge::<SC>::from_f(*c))
            .collect();
        Self {
            pcs,
            perm_challenges: packed_perm_challenges,
            alpha,
        }
    }

    /// Constructs quotient domains and computes the evaluation of the quotient polynomials
    /// on the quotient domains of each RAP.
    ///
    /// ## Assumptions
    /// - `raps`, `traces`, `quotient_degrees` are all the same length and in the same order.
    /// - `quotient_degrees` is the factor to **multiply** the trace degree by to get the degree
    ///   of the quotient polynomial. This should be determined from the constraint degree
    ///   of the RAP.
    #[instrument(name = "compute quotient values", skip_all)]
    pub fn quotient_values<'a>(
        &self,
        raps: Vec<&'a dyn ProverRap<SC>>,
        traces: Vec<ProvenSingleRapTraceView<'a, SC>>,
        quotient_degrees: &'a [usize],
        public_values: &'a [Val<SC>],
    ) -> QuotientData<SC>
    where
        Domain<SC>: Send + Sync,
        SC::Pcs: Sync,
        PcsProverData<SC>: Sync,
    {
        let inner = raps
            .into_par_iter()
            .zip_eq(traces.into_par_iter())
            .zip_eq(quotient_degrees.par_iter())
            .map(|((rap, trace), &quotient_degree)| {
                self.single_rap_quotient_values(rap, trace, quotient_degree, public_values)
            })
            .collect();
        QuotientData { inner }
    }

    pub fn single_rap_quotient_values<'a, R>(
        &self,
        rap: &'a R,
        trace: ProvenSingleRapTraceView<'a, SC>,
        quotient_degree: usize,
        public_values: &'a [Val<SC>],
    ) -> SingleQuotientData<SC>
    where
        R: for<'b> Rap<ProverConstraintFolder<'b, SC>> + ?Sized,
    {
        let trace_domain = trace.main.domain;
        let quotient_domain =
            trace_domain.create_disjoint_domain(trace_domain.size() * quotient_degree);
        // Empty matrix if no preprocessed trace
        let preprocessed_lde_on_quotient_domain = if let Some(view) = trace.preprocessed {
            self.pcs
                .get_evaluations_on_domain(view.data, view.index, quotient_domain)
                .to_row_major_matrix()
        } else {
            RowMajorMatrix::new(vec![], 0)
        };
        let main_lde_on_quotient_domain = self
            .pcs
            .get_evaluations_on_domain(trace.main.data, trace.main.index, quotient_domain)
            .to_row_major_matrix();
        // Empty matrix if no permutation
        let perm_lde_on_quotient_domain = if let Some(view) = trace.permutation {
            self.pcs
                .get_evaluations_on_domain(view.data, view.index, quotient_domain)
                .to_row_major_matrix()
        } else {
            RowMajorMatrix::new(vec![], 0)
        };
        let quotient_values = compute_single_rap_quotient_values(
            rap,
            trace_domain,
            quotient_domain,
            preprocessed_lde_on_quotient_domain,
            main_lde_on_quotient_domain,
            perm_lde_on_quotient_domain,
            &self.perm_challenges,
            self.alpha,
            public_values,
            &trace.permutation_exposed_values,
        );
        SingleQuotientData {
            quotient_degree,
            quotient_domain,
            quotient_values,
        }
    }

    // TODO: not sure this is the right function signature
    #[instrument(name = "commit to quotient poly chunks", skip_all)]
    pub fn commit(&self, data: QuotientData<SC>) -> ProverQuotientData<SC> {
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

/// The quotient polynomials from multiple RAP matrices.
pub struct QuotientData<SC: StarkGenericConfig> {
    inner: Vec<SingleQuotientData<SC>>,
}

impl<SC: StarkGenericConfig> QuotientData<SC> {
    /// Splits the quotient polynomials from multiple AIRs into chunks of size equal to the trace domain size.
    pub fn split(self) -> impl IntoIterator<Item = QuotientChunk<SC>> {
        self.inner.into_iter().flat_map(|data| data.split())
    }
}

/// The quotient polynomial from a single matrix RAP, evaluated on the quotient domain.
pub struct SingleQuotientData<SC: StarkGenericConfig> {
    /// The factor by which the trace degree was multiplied to get the
    /// quotient domain size.
    quotient_degree: usize,
    /// Quotient domain
    quotient_domain: Domain<SC>,
    /// Evaluations of the quotient polynomial on the quotient domain
    quotient_values: Vec<SC::Challenge>,
}

impl<SC: StarkGenericConfig> SingleQuotientData<SC> {
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
// TODO: make this into a trait that is auto-implemented so we can dynamic dispatch the trait
/// Computes evaluation of DEEP quotient polynomial on the quotient domain for a single AIR (single trace matrix).
#[allow(clippy::too_many_arguments)]
#[instrument(name = "compute single RAP quotient polynomial", skip_all)]
pub fn compute_single_rap_quotient_values<'a, SC, R, Mat>(
    rap: &'a R,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    preprocessed_trace_on_quotient_domain: Mat,
    main_lde_on_quotient_domain: Mat,
    perm_lde_on_quotient_domain: Mat,
    perm_challenges: &[PackedChallenge<SC>],
    alpha: SC::Challenge,
    public_values: &'a [Val<SC>],
    perm_exposed_values: &'a [SC::Challenge],
) -> Vec<SC::Challenge>
where
    // TODO: avoid ?Sized to prevent dynamic dispatching because `eval` is called many many times
    R: for<'b> Rap<ProverConstraintFolder<'b, SC>> + ?Sized,
    SC: StarkGenericConfig,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let preprocessed_width = preprocessed_trace_on_quotient_domain.width();
    let main_width = main_lde_on_quotient_domain.width();
    let perm_width = perm_lde_on_quotient_domain.width(); // Width with extension field elements flattened
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_size) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    let ext_degree = SC::Challenge::D;

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

            let preprocessed_local: Vec<_> = (0..preprocessed_width)
                .map(|col| {
                    PackedVal::<SC>::from_fn(|offset| {
                        preprocessed_trace_on_quotient_domain.get(wrap(i_start + offset), col)
                    })
                })
                .collect();
            let preprocessed_next: Vec<_> = (0..preprocessed_width)
                .map(|col| {
                    PackedVal::<SC>::from_fn(|offset| {
                        preprocessed_trace_on_quotient_domain
                            .get(wrap(i_start + next_step + offset), col)
                    })
                })
                .collect();

            let local: Vec<_> = (0..main_width)
                .map(|col| {
                    PackedVal::<SC>::from_fn(|offset| {
                        main_lde_on_quotient_domain.get(wrap(i_start + offset), col)
                    })
                })
                .collect();
            let next: Vec<_> = (0..main_width)
                .map(|col| {
                    PackedVal::<SC>::from_fn(|offset| {
                        main_lde_on_quotient_domain.get(wrap(i_start + next_step + offset), col)
                    })
                })
                .collect();

            let perm_local: Vec<_> = (0..perm_width)
                .step_by(ext_degree)
                .map(|col| {
                    PackedChallenge::<SC>::from_base_fn(|i| {
                        PackedVal::<SC>::from_fn(|offset| {
                            perm_lde_on_quotient_domain.get(wrap(i_start + offset), col + i)
                        })
                    })
                })
                .collect();

            let perm_next: Vec<_> = (0..perm_width)
                .step_by(ext_degree)
                .map(|col| {
                    PackedChallenge::<SC>::from_base_fn(|i| {
                        PackedVal::<SC>::from_fn(|offset| {
                            perm_lde_on_quotient_domain
                                .get(wrap(i_start + next_step + offset), col + i)
                        })
                    })
                })
                .collect();

            let accumulator = PackedChallenge::<SC>::zero();
            let mut folder = ProverConstraintFolder {
                preprocessed: VerticalPair::new(
                    RowMajorMatrixView::new_row(&preprocessed_local),
                    RowMajorMatrixView::new_row(&preprocessed_next),
                ),
                main: VerticalPair::new(
                    RowMajorMatrixView::new_row(&local),
                    RowMajorMatrixView::new_row(&next),
                ),
                perm: VerticalPair::new(
                    RowMajorMatrixView::new_row(&perm_local),
                    RowMajorMatrixView::new_row(&perm_next),
                ),
                perm_challenges,
                is_first_row,
                is_last_row,
                is_transition,
                alpha,
                accumulator,
                public_values,
                perm_exposed_values,
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
