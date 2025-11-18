use std::{
    fs::{create_dir_all, read, write},
    path::Path,
};

use eyre::{Report, Result};
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use serde::{Deserialize, Serialize};
use stark_backend_v2::{Digest, keygen::types::MultiStarkVerifyingKeyV2};

/// Verifying key and artifacts used to verify a STARK proof for a fixed VM and executable
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NonRootStarkVerifyingKey {
    pub mvk: MultiStarkVerifyingKeyV2,
    pub baseline: VerificationBaseline,
}

/// Baseline artifacts for a specific VM and fixed executable that are used to verify a final
/// (i.e. internal-recursive) VM STARK proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationBaseline {
    /// Commit to the app exe (i.e. hash of the program commit, initial memory merkle root,
    /// and initial program counter)
    pub app_exe_commit: Digest,
    /// VM memory metadata used to verify the user public values merkle proof
    pub memory_dimensions: MemoryDimensions,
    /// Cached trace commit of the leaf verifier circuit's SymbolicExpresionAir, which is
    /// derived from the app_vk
    pub leaf_commit: Digest,
    /// Cached trace commit of the internal-for-leaf verifier circuit's SymbolicExpresionAir,
    /// which derived from the leaf_vk
    pub internal_for_leaf_commit: Digest,
    /// Cached trace commit of the internal-recursive verifier circuit's SymbolicExpresionAir,
    /// which derived from the internal_for_leaf_vk
    pub internal_recursive_commit: Digest,
}

pub fn read_vk_from_file<P: AsRef<Path>>(path: P) -> Result<NonRootStarkVerifyingKey> {
    let ret = read(&path)
        .map_err(|e| read_error(&path, e.into()))
        .and_then(|data| {
            bitcode::deserialize(&data).map_err(|e: bitcode::Error| read_error(&path, e.into()))
        })?;
    Ok(ret)
}

pub fn write_vk_to_file<P: AsRef<Path>>(path: P, vk: &NonRootStarkVerifyingKey) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        create_dir_all(parent).map_err(|e| write_error(&path, e.into()))?;
    }
    bitcode::serialize(vk)
        .map_err(|e| write_error(&path, e.into()))
        .and_then(|bytes| write(&path, bytes).map_err(|e| write_error(&path, e.into())))?;
    Ok(())
}

fn read_error<P: AsRef<Path>>(path: P, error: Report) -> Report {
    eyre::eyre!(
        "reading from {} failed with the following error:\n    {}",
        path.as_ref().display(),
        error,
    )
}

fn write_error<P: AsRef<Path>>(path: P, error: Report) -> Report {
    eyre::eyre!(
        "writing to {} failed with the following error:\n    {}",
        path.as_ref().display(),
        error,
    )
}
