use std::{
    fs::{create_dir_all, write, File},
    io::Write,
    path::Path,
};

use axvm_circuit::arch::{instructions::exe::AxVmExe, VmConfig};
use axvm_native_recursion::halo2::{wrapper::EvmVerifier, EvmProof};
use eyre::Result;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    keygen::{AggProvingKey, AppProvingKey, AppVerifyingKey},
    prover::vm::ContinuationVmProof,
    F, SC,
};

pub fn read_exe_from_file<P: AsRef<Path>>(path: P) -> Result<AxVmExe<F>> {
    let data = std::fs::read(path)?;
    let exe = bincode::serde::decode_from_slice(&data, bincode::config::standard())?.0;
    Ok(exe)
}

pub fn write_exe_to_file<P: AsRef<Path>>(exe: AxVmExe<F>, path: P) -> Result<()> {
    let data = bincode::serde::encode_to_vec(&exe, bincode::config::standard())?;
    File::create(path)?.write_all(&data)?;
    Ok(())
}

pub fn read_app_pk_from_file<VC: VmConfig<F>, P: AsRef<Path>>(
    path: P,
) -> Result<AppProvingKey<VC>> {
    read_from_file_bson(path)
}

pub fn write_app_pk_to_file<VC: VmConfig<F>, P: AsRef<Path>>(
    app_pk: AppProvingKey<VC>,
    path: P,
) -> Result<()> {
    write_to_file_bson(path, app_pk)
}

pub fn read_app_vk_from_file<P: AsRef<Path>>(path: P) -> Result<AppVerifyingKey> {
    read_from_file_bson(path)
}

pub fn write_app_vk_to_file<P: AsRef<Path>>(app_vk: AppVerifyingKey, path: P) -> Result<()> {
    write_to_file_bson(path, app_vk)
}

pub fn read_app_proof_from_file<P: AsRef<Path>>(path: P) -> Result<ContinuationVmProof<SC>> {
    read_from_file_bson(path)
}

pub fn write_app_proof_to_file<P: AsRef<Path>>(
    proof: ContinuationVmProof<SC>,
    path: P,
) -> Result<()> {
    write_to_file_bson(path, proof)
}

pub fn read_agg_pk_from_file<P: AsRef<Path>>(path: P) -> Result<AggProvingKey> {
    read_from_file_bson(path)
}

pub fn write_agg_pk_to_file<P: AsRef<Path>>(agg_pk: AggProvingKey, path: P) -> Result<()> {
    write_to_file_bson(path, agg_pk)
}

pub fn read_evm_proof_from_file<P: AsRef<Path>>(path: P) -> Result<EvmProof> {
    read_from_file_bson(path)
}

pub fn write_evm_proof_to_file<P: AsRef<Path>>(proof: EvmProof, path: P) -> Result<()> {
    write_to_file_bson(path, proof)
}

pub fn read_evm_verifier_from_file<P: AsRef<Path>>(path: P) -> Result<EvmVerifier> {
    read_from_file_bson(path)
}

pub fn write_evm_verifier_to_file<P: AsRef<Path>>(verifier: EvmVerifier, path: P) -> Result<()> {
    write_to_file_bson(path, verifier)
}

pub(crate) fn read_from_file_bson<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let ret = bson::from_reader(File::open(path)?)?;
    Ok(ret)
}

pub(crate) fn write_to_file_bson<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    let bytes = bson::to_vec(&data)?;
    if let Some(parent) = path.as_ref().parent() {
        create_dir_all(parent)?;
    }
    write(path, bytes)?;
    Ok(())
}
