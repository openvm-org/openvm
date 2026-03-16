use std::{
    fs::{create_dir_all, read, write},
    path::Path,
};

use eyre::{Report, Result};
use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_stark_backend::keygen::types::MultiStarkVerifyingKey;
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest};
use serde::{Deserialize, Serialize};

use crate::DagCommit;

/// Verifying key and artifacts used to verify a STARK proof for a fixed VM and executable
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NonRootStarkVerifyingKey {
    pub mvk: MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
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
    /// Commit to the app_vk's DAG and its pre-hash, first exposed by the leaf verifier.
    pub app_dag_commit: DagCommit,
    /// Commit to the leaf_vk's DAG and its pre-hash, first exposed by the internal-for-leaf
    /// verifier.
    pub leaf_dag_commit: DagCommit,
    /// Commit to the internal_for_leaf_vk's DAG and its pre-hash, first exposed by the first
    /// (i.e. index 0) internal-recursive layer verifier.
    pub internal_for_leaf_dag_commit: DagCommit,
    /// Commit to the internal_recursive_vk's DAG and its pre-hash, exposed by subsequent (i.e.
    /// index > 0) internal-recursive layer verifiers.
    pub internal_recursive_dag_commit: DagCommit,
    /// Expected deferral VK commit (hash of the deferral aggregation prover's DAG commits).
    /// When `Some`, the proof must have `deferral_flag == 2` with a matching
    /// `def_hook_vk_commit` and valid deferral Merkle proofs. When `None`, the proof must
    /// have no deferral public values.
    pub expected_def_vk_commit: Option<Digest>,
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
