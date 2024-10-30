#[derive(Debug, PartialEq, Eq)]
pub enum VerificationError {
    InvalidProofShape,
    /// An error occurred while verifying the claimed openings.
    InvalidOpeningArgument(String),
    /// Out-of-domain evaluation mismatch, i.e. `constraints(zeta)` did not match
    /// `quotient(zeta) Z_H(zeta)`.
    OodEvaluationMismatch,
    NonZeroCumulativeSum,
    InitialPcMismatch(String),
    InitialMemoryRootMismatch,
    IsTerminateMismatch(String),
    ExitCodeMismatch(String),
    UnexpectedPvs(String),
    NumPublicValuesMismatch(String),
}
