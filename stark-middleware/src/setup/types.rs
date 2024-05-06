use p3_uni_stark::StarkGenericConfig;
use serde::{Deserialize, Serialize};

use crate::{config::Com, prover::types::ProverTraceData};

/// Common proving key for multiple AIRs
pub struct ProvingKey<SC: StarkGenericConfig> {
    /// Prover data for multi-matrix preprocessed trace commitments
    pub trace_data: Option<ProverTraceData<SC>>,
    /// Mapping
    // TODO: Add doc
    pub indices: Vec<usize>,
}

/// Common verifying key for multiple AIRs
#[derive(Serialize, Deserialize)]
pub struct VerifyingKey<SC: StarkGenericConfig> {
    /// PCS to preprocessed traces
    pub commit: Option<Com<SC>>,
    /// Preprocessed trace heights
    pub heights: Vec<usize>,
    /// Mapping
    // TODO: Add doc
    pub indices: Vec<usize>,
}
