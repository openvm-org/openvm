use openvm_circuit::arch::{VirtualMachineError, VmVerificationError};
use openvm_transpiler::transpiler::TranspilerError;
use thiserror::Error;
use verify_stark::error::VerifyStarkError;

use crate::SC;

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
    #[error("STARK verification failed with error: {0}")]
    VerifyStark(#[from] VerifyStarkError),
    #[error("Other error: {0}")]
    Other(#[from] eyre::Error),
}

impl From<VmVerificationError<SC>> for SdkError {
    fn from(error: VmVerificationError<SC>) -> Self {
        SdkError::Other(error.into())
    }
}
