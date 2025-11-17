use openvm_circuit::arch::{VirtualMachineError, VmVerificationError};
use openvm_continuations::F;
use openvm_native_compiler::ir::DIGEST_SIZE;
use thiserror::Error;

type Digest = [F; DIGEST_SIZE];

#[derive(Error, Debug)]
pub enum VerifyStarkError {
    #[error("Invalid user pv commit: expected {expected:?}, actual {actual:?}")]
    UserPvCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid app exe commit: expected {expected:?}, actual {actual:?}")]
    AppExeCommitMismatch { expected: Digest, actual: Digest },
    #[error("Invalid app VM commit: expected {expected:?}, actual {actual:?}")]
    AppVmCommitMismatch { expected: Digest, actual: Digest },
    #[error("VM error: {0}")]
    Vm(#[from] VirtualMachineError),
    #[error("Other error: {0}")]
    Other(eyre::Error),
}

impl From<VmVerificationError> for VerifyStarkError {
    fn from(error: VmVerificationError) -> Self {
        VerifyStarkError::Vm(error.into())
    }
}
