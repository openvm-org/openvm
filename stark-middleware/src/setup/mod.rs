use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};
use tracing::instrument;

pub mod types;

use crate::prover::{trace::TraceCommitter, types::ProverTraceData};

use self::types::{ProvingKey, VerifyingKey};

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

        let (indices, traces): (Vec<_>, Vec<_>) = maybe_traces
            .into_iter()
            .enumerate()
            .flat_map(|(i, mt)| mt.map(|t| (i, t)))
            .unzip();
        let heights: Vec<_> = traces.iter().map(|t| t.height()).collect();

        let (proven_trace, commit) = if !traces.is_empty() {
            let trace_committer = TraceCommitter::new(pcs);
            let proven_trace: ProverTraceData<SC> = trace_committer.commit(traces);
            let commit = proven_trace.commit.clone();
            (Some(proven_trace), Some(commit))
        } else {
            (None, None)
        };
        let vk = VerifyingKey {
            commit,
            heights,
            indices: indices.clone(),
        };
        let pk = ProvingKey {
            trace_data: proven_trace,
            indices,
        };

        (pk, vk)
    }
}
