use openvm_circuit::arch::{VirtualMachineError, VmVerificationError};
use openvm_transpiler::transpiler::TranspilerError;
use thiserror::Error;

use crate::commit::CommitBytes;

#[derive(Error, Debug)]
pub enum SdkError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to build guest: code = {0}")]
    BuildFailedWithCode(i32),
    #[error("Failed to build guest (OPENVM_SKIP_BUILD is set)")]
    BuildFailed,
    #[error("SDK must set a transpiler")]
    TranspilerNotAvailable,
    #[error("Transpiler error: {0}")]
    Transpiler(#[from] TranspilerError),
    #[error("VM error: {0}")]
    Vm(#[from] VirtualMachineError),
    #[error("Invalid app exe commit: expected {expected}, actual {actual}")]
    InvalidAppExeCommit {
        expected: CommitBytes,
        actual: CommitBytes,
    },
    #[error("Invalid app vm commit: expected {expected}, actual {actual}")]
    InvalidAppVmCommit {
        expected: CommitBytes,
        actual: CommitBytes,
    },
    #[error("Other error: {0}")]
    Other(eyre::Error),
}

impl From<VmVerificationError> for SdkError {
    fn from(error: VmVerificationError) -> Self {
        SdkError::Vm(error.into())
    }
}
