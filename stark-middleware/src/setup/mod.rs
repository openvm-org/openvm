use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::instrument;

pub mod types;

use crate::prover::{trace::TraceCommitter, types::ProverTraceData};

use self::types::{ProverPreprocessedData, ProvingKey, VerifierPreprocessedData, VerifyingKey};

/// Calculates the Proving and Verifying keys for a partition of multi-matrix AIRs.
pub struct PartitionSetup<'a, SC: StarkGenericConfig> {
    pub config: &'a SC,
}

impl<'a, SC: StarkGenericConfig> PartitionSetup<'a, SC> {
    pub fn new(config: &'a SC) -> Self {
        Self { config }
    }

    #[instrument(name = "PartitionSetup::setup", level = "debug", skip_all)]
    pub fn setup(
        &self,
        traces: Vec<Option<RowMajorMatrix<Val<SC>>>>,
    ) -> (ProvingKey<SC>, VerifyingKey<SC>) {
        let pcs = self.config.pcs();

        let (prover_data, verifier_data): (Vec<_>, Vec<_>) = traces
            .into_iter()
            .map(|mt| {
                mt.map(|trace| {
                    let trace_committer = TraceCommitter::new(pcs);
                    let degree = trace.height();
                    let mut proven_trace: ProverTraceData<SC> = trace_committer.commit(vec![trace]);
                    let (domain, trace) = proven_trace
                        .traces_with_domains
                        .pop()
                        .expect("Expected a single preprocessed trace");

                    let vdata = VerifierPreprocessedData {
                        commit: proven_trace.commit.clone(),
                        degree,
                    };

                    let pdata = ProverPreprocessedData {
                        domain,
                        trace,
                        commit: proven_trace.commit,
                        data: proven_trace.data,
                    };

                    (pdata, vdata)
                })
                .unzip()
            })
            .unzip();

        let pk = ProvingKey {
            preprocessed_data: prover_data,
        };
        let vk = VerifyingKey {
            preprocessed_data: verifier_data,
        };

        (pk, vk)
    }
}
