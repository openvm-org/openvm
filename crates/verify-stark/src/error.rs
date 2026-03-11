use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProofError;
use openvm_stark_backend::verifier::VerifierError;
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VerifyStarkError {
    #[error("Stark verifier failed with error: {0}")]
    StarkVerificationFailure(#[from] VerifierError<EF>),
    #[error("User public value proof verification failed with error: {0}")]
    UserPvsVerificationFailure(#[from] UserPublicValuesProofError),
    #[error("Invalid app exe commit: expected {expected:?}, actual {actual:?}")]
    AppExeCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid app cached commit: expected {expected:?}, actual {actual:?}")]
    AppDagCachedCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid app vk pre-hash: expected {expected:?}, actual {actual:?}")]
    AppDagPreHashMismatch { expected: Digest, actual: Digest },
    #[error("Invalid leaf cached commit: expected {expected:?}, actual {actual:?}")]
    LeafDagCachedCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid leaf vk pre-hash: expected {expected:?}, actual {actual:?}")]
    LeafDagPreHashMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal for leaf cached commit: expected {expected:?}, actual {actual:?}")]
    InternalForLeafDagCachedCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal for leaf vk pre-hash: expected {expected:?}, actual {actual:?}")]
    InternalForLeafDagPreHashMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal recursive cached commit: expected {expected:?}, actual {actual:?}")]
    InternalRecursiveDagCachedCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid internal recursive vk pre-hash: expected {expected:?}, actual {actual:?}")]
    InternalRecursiveDagPreHashMismatch { expected: Digest, actual: Digest },
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
    #[error("Deferrals are not enabled in verify-stark yet")]
    DeferralNotEnabled,
}
