use p3_uni_stark::StarkGenericConfig;
use serde::{Deserialize, Serialize};

use crate::{config::Com, prover::types::ProverTraceData};

pub struct ProverPreprocessedData<SC: StarkGenericConfig> {
    /// Prover data for PCS commitment to the preprocessed trace.
    // TODO: Replace with struct that contains only single trace
    pub data: ProverTraceData<SC>,
}

#[derive(Serialize, Deserialize)]
pub struct VerifierPreprocessedData<SC: StarkGenericConfig> {
    /// PCS commitment to the preprocessed trace.
    pub commit: Com<SC>,
    /// Height of trace matrix
    pub degree: usize,
}

/// Common proving key for multiple AIRs.
///
/// This struct contains the necessary data for the prover to generate proofs for multiple AIRs
/// using a single proving key.
pub struct ProvingKey<SC: StarkGenericConfig> {
    /// Prover data for the preprocessed trace for each AIR.
    /// None if AIR doesn't have a preprocessed trace.
    pub preprocessed_data: Vec<Option<ProverPreprocessedData<SC>>>,
}

/// Common verifying key for multiple AIRs.
///
/// This struct contains the necessary data for the verifier to verify proofs generated for
/// multiple AIRs using a single verifying key.
#[derive(Serialize, Deserialize)]
pub struct VerifyingKey<SC: StarkGenericConfig> {
    /// Verifier data for the preprocessed trace for each AIR.
    /// None if AIR doesn't have a preprocessed trace.
    pub preprocessed_data: Vec<Option<VerifierPreprocessedData<SC>>>,
}
