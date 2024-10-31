use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum VerificationError {
    #[error("invalid proof shape")]
    InvalidProofShape,
    /// An error occurred while verifying the claimed openings.
    #[error("invalid opening argument: {0}")]
    InvalidOpeningArgument(String),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    #[error("out-of-domain evaluation mismatch")]
    OodEvaluationMismatch,
    #[error("non-zero cumulative sum")]
    NonZeroCumulativeSum,
}
