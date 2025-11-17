use std::{
    fs::{create_dir_all, read, write},
    path::Path,
};

use eyre::{Report, Result};
use openvm_circuit::openvm_stark_sdk::{
    config::FriParameters,
    openvm_stark_backend::{config::Com, keygen::types::MultiStarkVerifyingKey},
};
use openvm_continuations::SC;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct VmStarkVerifyingKey {
    pub leaf_fri_params: FriParameters,
    pub leaf_vk: MultiStarkVerifyingKey<SC>,

    pub internal_fri_params: FriParameters,
    pub internal_vk: MultiStarkVerifyingKey<SC>,
    pub internal_verifier_program_commit: Com<SC>,

    pub expected_app_exe_commit: Com<SC>,
    pub expected_app_vm_commit: Com<SC>,
}

pub fn read_vk_from_file<P: AsRef<Path>>(path: P) -> Result<VmStarkVerifyingKey> {
    let ret = read(&path)
        .map_err(|e| read_error(&path, e.into()))
        .and_then(|data| {
            bitcode::deserialize(&data).map_err(|e: bitcode::Error| read_error(&path, e.into()))
        })?;
    Ok(ret)
}

pub fn write_vk_to_file<P: AsRef<Path>>(path: P, vk: &VmStarkVerifyingKey) -> Result<()> {
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
