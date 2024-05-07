use p3_uni_stark::StarkGenericConfig;
use serde::{Deserialize, Serialize};

use crate::{config::Com, prover::types::ProverTraceData};

/// Common proving key for multiple AIRs.
///
/// This struct contains the necessary data for the prover to generate proofs for multiple AIRs
/// using a single proving key. It includes the prover data for multi-matrix commitments to
/// preprocessed traces and a mapping from trace indices to AIR indices.
pub struct ProvingKey<SC: StarkGenericConfig> {
    /// Prover data for multi-matrix commitments to preprocessed traces.
    ///
    /// This field contains the data required by the prover to commit to the preprocessed traces
    /// of the AIRs. It is an `Option` type, and it will be `None` if none of the AIRs contain
    /// a preprocessed trace.
    pub trace_data: Option<ProverTraceData<SC>>,
    /// Mapping from `ProverTraceData::traces_with_domains` vector index to AIR index.
    ///
    /// This vector provides a mapping between the indices of the traces in the
    /// `traces_with_domains` vector of `ProverTraceData` and the corresponding AIR indices.
    /// It allows the prover to associate a preprocessed trace with its respective AIR.
    pub indices: Vec<usize>,
}

/// Common verifying key for multiple AIRs.
///
/// This struct contains the necessary data for the verifier to verify proofs generated for
/// multiple AIRs using a single verifying key. It includes the combined PCS commitment to all
/// preprocessed traces, the heights of the preprocessed traces, and a mapping from height
/// vector indices to AIR indices.
#[derive(Serialize, Deserialize)]
pub struct VerifyingKey<SC: StarkGenericConfig> {
    /// Combined PCS commitment to all preprocessed traces.
    ///
    /// This field represents the combined PCS commitment to all the preprocessed traces of the
    /// AIRs. It is an `Option` type, and it will be `None` if none of the AIRs contain a
    /// preprocessed trace.
    pub commit: Option<Com<SC>>,
    /// Heights of the preprocessed traces.
    ///
    /// This vector contains the heights of the preprocessed traces of the AIRs. The heights
    /// are packed and only include non-zero values to optimize storage.
    pub heights: Vec<usize>,
    /// Mapping from height vector index to AIR index.
    ///
    /// This vector provides a mapping between the indices of the heights in the `heights`
    /// vector and the corresponding AIR indices. It allows the verifier to associate a
    /// preprocessed trace with its respective AIR.
    pub indices: Vec<usize>,
}
