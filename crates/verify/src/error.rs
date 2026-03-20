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
    #[error("Internal recursive cached commit should be unset, actual {actual:?}")]
    InternalRecursiveDagCachedCommitSet { actual: Digest },
    #[error("Internal recursive vk pre-hash should be unset, actual {actual:?}")]
    InternalRecursiveDagPreHashSet { actual: Digest },
    #[error("Invalid proof cached commit: expected {expected:?}, actual {actual:?}")]
    ProofCachedCommitMismatch { expected: Digest, actual: Digest },
    #[error("Missing constraint-eval trace vdata, expected at AIR idx {air_idx}")]
    MissingConstraintEvalTraceVdata { air_idx: usize },
    #[error("Missing constraint-eval cached trace at AIR idx {air_idx} cached idx {cached_idx}")]
    MissingConstraintEvalCachedTrace { air_idx: usize, cached_idx: usize },
    #[error("Program execution did not terminate successfully, exit_code: {0}")]
    ExecutionUnsuccessful(F),
    #[error("Invalid internal flag {0}, should be 2")]
    InvalidInternalFlag(F),
    #[error("Invalid recursion flag {0}, should be 1 or 2")]
    InvalidRecursionFlag(F),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Other error: {0}")]
    Other(#[from] eyre::Error),
    #[error("Deferral Merkle proof length mismatch: expected {expected}, actual {actual}")]
    DeferralMerkleProofLengthMismatch { expected: usize, actual: usize },
    #[error("Deferral initial root mismatch: expected {expected:?}, actual {actual:?}")]
    DeferralInitialRootMismatch { expected: Digest, actual: Digest },
    #[error("Deferral final root mismatch: expected {expected:?}, actual {actual:?}")]
    DeferralFinalRootMismatch { expected: Digest, actual: Digest },
    #[error("Invalid deferral flag {0}, should be 0 or 2")]
    InvalidDeferralFlag(F),
    #[error("Deferral hook VK commit mismatch: expected {expected:?}, actual {actual:?}")]
    DefHookVkCommitMismatch { expected: Digest, actual: Digest },
    #[error("Proof has deferrals but baseline has no expected_def_hook_vk_commit")]
    UnexpectedDeferralDisabled,
    #[error("Baseline expects deferrals but proof has no deferral Merkle proofs")]
    MissingDeferralMerkleProofs,
    #[error("Proof has deferral_flag=0 but def_hook_vk_commit is set, actual {actual:?}")]
    DefHookVkCommitSet { actual: Digest },
    #[error("Proof has deferral_flag=0 but initial_acc_hash is set, actual {actual:?}")]
    DefInitialAccHashCommitSet { actual: Digest },
    #[error("Proof has deferral_flag=0 but final_acc_hash is set, actual {actual:?}")]
    DefFinalAccHashCommitSet { actual: Digest },
    #[error("Proof has deferral_flag=0 but depth is set, actual {actual:?}")]
    DefDepthSet { actual: F },
}
