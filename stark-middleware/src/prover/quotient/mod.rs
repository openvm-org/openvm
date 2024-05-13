use itertools::Itertools;
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use p3_uni_stark::{Domain, PackedChallenge, StarkGenericConfig, Val};
use tracing::instrument;

use crate::{
    air_builders::prover::ProverConstraintFolder,
    config::{Com, PcsProverData},
    rap::Rap,
};

use super::{trace::SingleRapCommittedTraceView, types::ProverRap};

use self::single::compute_single_rap_quotient_values;

pub mod single;

pub struct QuotientCommitter<'pcs, SC: StarkGenericConfig> {
    pcs: &'pcs SC::Pcs,
    /// For each challenge round, the challenges drawn
    challenges: Vec<Vec<PackedChallenge<SC>>>,
    alpha: SC::Challenge,
}

impl<'pcs, SC: StarkGenericConfig> QuotientCommitter<'pcs, SC> {
    pub fn new(
        pcs: &'pcs SC::Pcs,
        challenges: &[Vec<SC::Challenge>],
        alpha: SC::Challenge,
    ) -> Self {
        let packed_challenges = challenges
            .iter()
            .map(|challenges| {
                challenges
                    .iter()
                    .map(|c| PackedChallenge::<SC>::from_f(*c))
                    .collect_vec()
            })
            .collect_vec();
        Self {
            pcs,
            challenges: packed_challenges,
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
        traces: Vec<SingleRapCommittedTraceView<'a, SC>>,
        quotient_degrees: &'a [usize],
        public_values: &'a [Vec<Val<SC>>],
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
            .zip_eq(public_values.par_iter())
            .map(|(((rap, trace), &quotient_degree), pis)| {
                self.single_rap_quotient_values(rap, trace, quotient_degree, pis)
            })
            .collect();
        QuotientData { inner }
    }

    pub fn single_rap_quotient_values<'a, R>(
        &self,
        rap: &'a R,
        trace: SingleRapCommittedTraceView<'a, SC>,
        quotient_degree: usize,
        public_values: &'a [Val<SC>],
    ) -> SingleQuotientData<SC>
    where
        R: for<'b> Rap<ProverConstraintFolder<'b, SC>> + ?Sized,
    {
        let trace_domain = trace.domain;
        let quotient_domain =
            trace_domain.create_disjoint_domain(trace_domain.size() * quotient_degree);
        // Empty matrix if no preprocessed trace
        let preprocessed_lde_on_quotient_domain = if let Some(view) = trace.preprocessed {
            self.pcs
                .get_evaluations_on_domain(view.data, view.matrix_index, quotient_domain)
                .to_row_major_matrix()
        } else {
            RowMajorMatrix::new(vec![], 0)
        };
        let partitioned_main_lde_on_quotient_domain: Vec<_> = trace
            .partitioned_main
            .into_iter()
            .map(|view| {
                self.pcs
                    .get_evaluations_on_domain(view.data, view.matrix_index, quotient_domain)
                    .to_row_major_matrix()
            })
            .collect();

        let (after_challenge_lde_on_quotient_domain, exposed_values_after_challenge): (
            Vec<_>,
            Vec<_>,
        ) = trace
            .after_challenge
            .iter()
            .map(|(view, exposed_values)| {
                (
                    self.pcs
                        .get_evaluations_on_domain(view.data, view.matrix_index, quotient_domain)
                        .to_row_major_matrix(),
                    exposed_values.as_slice(),
                )
            })
            .unzip();

        let quotient_values = compute_single_rap_quotient_values(
            rap,
            trace_domain,
            quotient_domain,
            preprocessed_lde_on_quotient_domain,
            partitioned_main_lde_on_quotient_domain,
            after_challenge_lde_on_quotient_domain,
            &self.challenges,
            self.alpha,
            public_values,
            &exposed_values_after_challenge,
        );
        SingleQuotientData {
            quotient_degree,
            quotient_domain,
            quotient_values,
        }
    }

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

/// Prover data for multi-matrix quotient polynomial commitment.
/// Quotient polynomials for multiple RAP matrices are committed together into a single commitment.
/// The quotient polynomials can be committed together even if the corresponding trace matrices
/// are committed separately.
pub struct ProverQuotientData<SC: StarkGenericConfig> {
    /// For each AIR, the number of quotient chunks that were committed.
    pub quotient_degrees: Vec<usize>,
    /// Quotient commitment
    pub commit: Com<SC>,
    /// Prover data for the quotient commitment
    pub data: PcsProverData<SC>,
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
