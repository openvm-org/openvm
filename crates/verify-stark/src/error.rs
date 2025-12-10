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
    #[error("Invalid app vk commit: expected {expected:?}, actual {actual:?}")]
    AppVkCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid leaf vk commit: expected {expected:?}, actual {actual:?}")]
    LeafVkCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal for leaf vk commit: expected {expected:?}, actual {actual:?}")]
    InternalForLeafVkCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal recursive vk commit: expected {expected:?}, actual {actual:?}")]
    InternalRecursiveVkCommitMismatch { expected: Digest, actual: Digest },
    #[error(
        "Internal recursive vk commit (actual {actual:?}) shouldn't be defined if recursion flag is 1"
    )]
    InternalRecursiveVkCommitDefined { actual: Digest },
    #[error("Program execution did not terminate successfully, exit_code: {0}")]
    ExecutionUnsuccessful(F),
    #[error("Invalid internal flag {0}, should be 2")]
    InvalidInternalFlag(F),
    #[error("Invalid recursion flag {0}, should be either 1 or 2")]
    InvalidRecursionFlag(F),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Other error: {0}")]
    Other(#[from] eyre::Error),
}
