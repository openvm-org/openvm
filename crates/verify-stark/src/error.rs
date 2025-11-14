use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProofError;
use stark_backend_v2::{Digest, F, verifier::VerifierError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VerifyStarkError {
    #[error("Stark verifier failed with error: {0}")]
    StarkVerificationFailure(#[from] VerifierError),
    #[error("User public value proof verification failed with error: {0}")]
    UserPvsVerificationFailure(#[from] UserPublicValuesProofError),
    #[error("Invalid user pv commit: expected {expected:?}, actual {actual:?}")]
    UserPvCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid app exe commit: expected {expected:?}, actual {actual:?}")]
    AppExeCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid leaf commit: expected {expected:?}, actual {actual:?}")]
    LeafCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal for leaf commit: expected {expected:?}, actual {actual:?}")]
    InternalForLeafCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal recursive commit: expected {expected:?}, actual {actual:?}")]
    InternalRecursiveMismatch { expected: Digest, actual: Digest },
    #[error("Internal recursive commit should not be defined for internal flag 1")]
    InternalRecursiveDefined,
    #[error("Program execution did not terminate successfully, exit_code: {0}")]
    ExecutionUnsuccessful(F),
    #[error("Invalid internal flag {0}, should be either 1 or 2")]
    InvalidInternalFlag(F),
    #[error("Other error: {0}")]
    Other(eyre::Error),
}
