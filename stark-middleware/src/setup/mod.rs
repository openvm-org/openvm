use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::instrument;

pub mod types;

use self::types::{PreprocessedTraceCommitter, ProvingKey, VerifyingKey};

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
        maybe_traces: Vec<Option<RowMajorMatrix<Val<SC>>>>,
    ) -> (ProvingKey<SC>, VerifyingKey<SC>) {
        let pcs = self.config.pcs();

        let heights: Vec<_> = maybe_traces
            .iter()
            .flat_map(|mt| mt.as_ref().map(|t| t.height()))
            .collect();
        let indices_lookup = maybe_traces
            .iter()
            .enumerate()
            .flat_map(|(i, mt)| mt.as_ref().map(|_| i))
            .collect();

        let trace_committer = PreprocessedTraceCommitter::new(pcs);
        let proven_trace = trace_committer.commit(maybe_traces);
        let vk = VerifyingKey {
            commit: proven_trace.commit.clone(),
            heights,
            indices_lookup,
        };
        let pk = ProvingKey {
            trace_data: proven_trace,
        };

        (pk, vk)
    }
}
