use openvm_circuit::arch::{VirtualMachineError, VmVerificationError};
use openvm_continuations::F;
use openvm_native_compiler::ir::DIGEST_SIZE;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VerifyStarkError {
    #[error("Invalid user pv commit: expected {expected:?}, actual {actual:?}")]
    UserPvCommitMismatch {
        expected: [F; DIGEST_SIZE],
        actual: [F; DIGEST_SIZE],
    },
    #[error("Invalid app exe commit: expected {expected:?}, actual {actual:?}")]
    AppExeCommitMismatch {
        expected: [F; DIGEST_SIZE],
        actual: [F; DIGEST_SIZE],
    },
    #[error("Invalid app VM commit: expected {expected:?}, actual {actual:?}")]
    AppVmCommitMismatch {
        expected: [F; DIGEST_SIZE],
        actual: [F; DIGEST_SIZE],
    },
    #[error("VM error: {0}")]
    Vm(#[from] VirtualMachineError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Other error: {0}")]
    Other(eyre::Error),
}

impl From<VmVerificationError> for VerifyStarkError {
    fn from(error: VmVerificationError) -> Self {
        VerifyStarkError::Vm(error.into())
    }
}
