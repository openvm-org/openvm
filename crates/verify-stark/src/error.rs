use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProofError;
use stark_backend_v2::{Digest, F, verifier::VerifierError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VerifyStarkError {
    #[error("Stark verifier failed with error: {0}")]
    StarkVerificationFailure(#[from] VerifierError),
    #[error("User public value proof verification failed with error: {0}")]
    UserPvsVerificationFailure(#[from] UserPublicValuesProofError),
    #[error("Invalid app exe commit: expected {expected:?}, actual {actual:?}")]
    AppExeCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid app DAG commit: expected {expected:?}, actual {actual:?}")]
    AppDagCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid leaf DAG commit: expected {expected:?}, actual {actual:?}")]
    LeafDagCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal for leaf DAG commit: expected {expected:?}, actual {actual:?}")]
    InternalForLeafDagCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal recursive DAG commit: expected {expected:?}, actual {actual:?}")]
    InternalRecursiveDagCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid compression commit: expected {expected:?}, actual {actual:?}")]
    CompressionCommitMismatch { expected: Vec<F>, actual: Vec<F> },
    #[error("Compression commit should not be defined if not enabled, actual {actual:?}")]
    CompressionCommitDefined { actual: Vec<F> },
    #[error("Program execution did not terminate successfully, exit_code: {0}")]
    ExecutionUnsuccessful(F),
    #[error("Invalid internal flag {0}, should be 2")]
    InvalidInternalFlag(F),
    #[error("Invalid recursion flag {0}, should be 2")]
    InvalidRecursionFlag(F),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Other error: {0}")]
    Other(#[from] eyre::Error),
}
