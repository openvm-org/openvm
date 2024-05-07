use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{Deserialize, Serialize};

use crate::config::{Com, PcsProverData};

pub struct ProverPreprocessedData<SC: StarkGenericConfig> {
    /// Domain the trace was commited with respect to.
    pub domain: Domain<SC>,
    /// Preprocessed trace matrix.
    pub trace: RowMajorMatrix<Val<SC>>,
    /// Commitment to the preprocessed trace.
    pub commit: Com<SC>,
    /// Prover data, such as a Merkle tree, for the trace commitment.
    pub data: PcsProverData<SC>,
}

#[derive(Serialize, Deserialize)]
pub struct VerifierPreprocessedData<SC: StarkGenericConfig> {
    /// Height of trace matrix.
    pub degree: usize,
    /// Commitment to the preprocessed trace.
    pub commit: Com<SC>,
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
