use std::{
    fs::{create_dir_all, read, write},
    path::Path,
};

use eyre::Result;
use openvm_circuit::arch::{instructions::exe::VmExe, VmConfig};
use openvm_native_recursion::halo2::{wrapper::EvmVerifier, EvmProof};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    keygen::{AggProvingKey, AppProvingKey, AppVerifyingKey},
    prover::vm::ContinuationVmProof,
    F, SC,
};

// TODO: remove these type specific functions
pub fn read_exe_from_file<P: AsRef<Path>>(path: P) -> Result<VmExe<F>> {
    read_from_file_bitcode(path)
}

pub fn write_exe_to_file<P: AsRef<Path>>(exe: VmExe<F>, path: P) -> Result<()> {
    write_to_file_bitcode(path, exe)
}

pub fn read_app_pk_from_file<VC: VmConfig<F>, P: AsRef<Path>>(
    path: P,
) -> Result<AppProvingKey<VC>> {
    read_from_file_bitcode(path)
}

pub fn write_app_pk_to_file<VC: VmConfig<F>, P: AsRef<Path>>(
    app_pk: AppProvingKey<VC>,
    path: P,
) -> Result<()> {
    write_to_file_bitcode(path, app_pk)
}

pub fn read_app_vk_from_file<P: AsRef<Path>>(path: P) -> Result<AppVerifyingKey> {
    read_from_file_bitcode(path)
}

pub fn write_app_vk_to_file<P: AsRef<Path>>(app_vk: AppVerifyingKey, path: P) -> Result<()> {
    write_to_file_bitcode(path, app_vk)
}

pub fn read_app_proof_from_file<P: AsRef<Path>>(path: P) -> Result<ContinuationVmProof<SC>> {
    read_from_file_bitcode(path)
}

pub fn write_app_proof_to_file<P: AsRef<Path>>(
    proof: ContinuationVmProof<SC>,
    path: P,
) -> Result<()> {
    write_to_file_bitcode(path, proof)
}

pub fn read_agg_pk_from_file<P: AsRef<Path>>(path: P) -> Result<AggProvingKey> {
    read_from_file_bitcode(path)
}

pub fn write_agg_pk_to_file<P: AsRef<Path>>(agg_pk: AggProvingKey, path: P) -> Result<()> {
    write_to_file_bitcode(path, agg_pk)
}

pub fn read_evm_proof_from_file<P: AsRef<Path>>(path: P) -> Result<EvmProof> {
    read_from_file_bitcode(path)
}

pub fn write_evm_proof_to_file<P: AsRef<Path>>(proof: EvmProof, path: P) -> Result<()> {
    write_to_file_bitcode(path, proof)
}

pub fn read_evm_verifier_from_file<P: AsRef<Path>>(path: P) -> Result<EvmVerifier> {
    read_from_file_bytes(path)
}

pub fn write_evm_verifier_to_file<P: AsRef<Path>>(verifier: EvmVerifier, path: P) -> Result<()> {
    write_to_file_bytes(path, verifier)
}

pub fn read_from_file_bitcode<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let data = std::fs::read(path)?;
    let ret = bitcode::deserialize(&data)?;
    Ok(ret)
}

pub fn write_to_file_bitcode<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    let bytes = bitcode::serialize(&data)?;
    if let Some(parent) = path.as_ref().parent() {
        create_dir_all(parent)?;
    }
    write(path, bytes)?;
    Ok(())
}

pub fn read_from_file_bytes<T: From<Vec<u8>>, P: AsRef<Path>>(path: P) -> Result<T> {
    let bytes = read(path)?;
    Ok(T::from(bytes))
}

pub fn write_to_file_bytes<T: Into<Vec<u8>>, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        create_dir_all(parent)?;
    }
    write(path, data.into())?;
    Ok(())
}
