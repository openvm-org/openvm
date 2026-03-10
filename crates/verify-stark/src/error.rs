use openvm_circuit::system::memory::merkle::public_values::UserPublicValuesProofError;
use openvm_stark_backend::verifier::VerifierError;
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use thiserror::Error;

use crate::DagCommit;

#[derive(Error, Debug)]
pub enum VerifyStarkError {
    #[error("Stark verifier failed with error: {0}")]
    StarkVerificationFailure(#[from] VerifierError<EF>),
    #[error("User public value proof verification failed with error: {0}")]
    UserPvsVerificationFailure(#[from] UserPublicValuesProofError),
    #[error("Invalid app exe commit: expected {expected:?}, actual {actual:?}")]
    AppExeCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid app DAG commit: expected {expected:?}, actual {actual:?}")]
    AppDagCommitMismatch {
        expected: DagCommit,
        actual: DagCommit,
    },
    #[error("Invalid leaf DAG commit: expected {expected:?}, actual {actual:?}")]
    LeafDagCommitMismatch {
        expected: DagCommit,
        actual: DagCommit,
    },
    #[error("Invalid internal for leaf DAG commit: expected {expected:?}, actual {actual:?}")]
    InternalForLeafDagCommitMismatch {
        expected: DagCommit,
        actual: DagCommit,
    },
    #[error("Invalid internal recursive DAG commit: expected {expected:?}, actual {actual:?}")]
    InternalRecursiveDagCommitMismatch {
        expected: DagCommit,
        actual: DagCommit,
    },
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
