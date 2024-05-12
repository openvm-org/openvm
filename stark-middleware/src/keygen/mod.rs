use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::instrument;

pub mod types;

use crate::{
    air_builders::symbolic::get_log_quotient_degree,
    commit::{MatrixCommitmentGraph, SingleMatrixCommitPtr},
    prover::trace::TraceCommitter,
};

use self::types::{
    MultiStarkProvingKey, ProverOnlySinglePreprocessedData, StarkProvingKey, StarkVerifyingKey,
    SymbolicRap, TraceWidth, VerifierSinglePreprocessedData,
};

/// Stateful builder to create multi-stark proving and verifying keys
/// for system of multiple RAPs with multiple multi-matrix commitments
pub struct MultiStarkKeygenBuilder<'a, SC: StarkGenericConfig> {
    pub config: &'a SC,
    pub pk: MultiStarkProvingKey<SC>,
}

impl<'a, SC: StarkGenericConfig> MultiStarkKeygenBuilder<'a, SC> {
    pub fn new(config: &'a SC) -> Self {
        Self {
            config,
            pk: MultiStarkProvingKey::empty(),
        }
    }

    /// Generates proving key, reseting the state of the builder.
    /// The verifying key can be obtained from the proving key.
    pub fn generate_pk(&mut self) -> MultiStarkProvingKey<SC> {
        std::mem::take(&mut self.pk)
    }

    /// Default way to add a single Interactive AIR.
    /// DO NOT use this if the main trace needs to be partitioned.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - Adds main trace to the default shared main trace commitment.
    #[instrument(level = "debug", skip_all)]
    pub fn add_air(&mut self, air: &dyn SymbolicRap<SC>, degree: usize, num_public_values: usize) {
        let (prep_prover_data, prep_verifier_data): (Option<_>, Option<_>) =
            self.get_single_preprocessed_data(air, degree).unzip();
        let preprocessed_width = prep_prover_data.as_ref().map(|d| d.trace.width());
        let main_width = air.width();
        let perm_width = air.permutation_width();
        let width = TraceWidth {
            preprocessed: preprocessed_width,
            partitioned_main: vec![main_width],
            after_challenge: perm_width.to_vec(),
        };
        let log_quotient_degree =
            get_log_quotient_degree(air, preprocessed_width, num_public_values);
        let quotient_degree = 1 << log_quotient_degree;
        let vk = StarkVerifyingKey {
            degree,
            preprocessed_data: prep_verifier_data,
            width,
            main_graph: MatrixCommitmentGraph::new(vec![SingleMatrixCommitPtr::new(0, 0)]),
            quotient_degree,
        };
        let pk = StarkProvingKey {
            vk,
            preprocessed_data: prep_prover_data,
        };

        self.pk.per_air.push(pk);
    }

    /// Default way to add a single Interactive AIR.
    /// DO NOT use this if the main trace needs to be partitioned.
    /// - `degree` is height of trace matrix
    /// - Generates preprocessed trace and creates a dedicated commitment for it.
    /// - Adds main trace to the default shared main trace commitment.
    #[instrument(level = "debug", skip_all)]
    pub fn get_single_preprocessed_data(
        &mut self,
        air: &dyn SymbolicRap<SC>,
        degree: usize,
    ) -> Option<(
        ProverOnlySinglePreprocessedData<SC>,
        VerifierSinglePreprocessedData<SC>,
    )> {
        let pcs = self.config.pcs();
        let preprocessed_trace = air.preprocessed_trace();
        let preprocessed_width = preprocessed_trace.as_ref().map(|t| t.width());
        preprocessed_trace.map(|trace| {
            let trace_committer = TraceCommitter::new(pcs);
            let mut data = trace_committer.commit(vec![trace]);
            let (domain, trace) = data
                .traces_with_domains
                .pop()
                .expect("Expected a single preprocessed trace");

            let vdata = VerifierSinglePreprocessedData {
                commit: data.commit,
            };
            let pdata = ProverOnlySinglePreprocessedData {
                trace,
                data: data.data,
            };
            (pdata, vdata)
        })
    }
}
