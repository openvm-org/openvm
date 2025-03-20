use std::{
    fs::{create_dir_all, read, write, File},
    path::Path,
};

use eyre::Result;
use openvm_circuit::arch::{instructions::exe::VmExe, ContinuationVmProof, VmConfig};
use openvm_native_recursion::halo2::wrapper::{EvmVerifier, EvmVerifierByteCode};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    keygen::{AggProvingKey, AppProvingKey, AppVerifyingKey},
    types::EvmProof,
    F, SC,
};

pub const EVM_VERIFIER_SOL_FILENAME: &str = "verifier.sol";
pub const EVM_VERIFIER_ARTIFACT_FILENAME: &str = "verifier.bytecode.json";

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
    let proof: EvmProof = serde_json::from_reader(File::open(path)?)?;
    Ok(proof)
}

pub fn write_evm_proof_to_file<P: AsRef<Path>>(proof: EvmProof, path: P) -> Result<()> {
    serde_json::to_writer(File::create(path)?, &proof)?;
    Ok(())
}

pub fn read_evm_verifier_from_folder<P: AsRef<Path>>(folder: P) -> Result<EvmVerifier> {
    let sol_code_path = folder.as_ref().join(EVM_VERIFIER_SOL_FILENAME);
    let sol_code = std::fs::read_to_string(sol_code_path)?;
    let artifact_path = folder.as_ref().join(EVM_VERIFIER_ARTIFACT_FILENAME);
    let artifact: EvmVerifierByteCode = serde_json::from_reader(File::open(artifact_path)?)?;
    Ok(EvmVerifier { sol_code, artifact })
}

pub fn write_evm_verifier_to_folder<P: AsRef<Path>>(
    verifier: EvmVerifier,
    folder: P,
) -> Result<()> {
    let sol_code_path = folder.as_ref().join(EVM_VERIFIER_SOL_FILENAME);
    std::fs::write(sol_code_path, verifier.sol_code)?;
    let artifact_path = folder.as_ref().join(EVM_VERIFIER_ARTIFACT_FILENAME);
    serde_json::to_writer(File::create(artifact_path)?, &verifier.artifact)?;
    Ok(())
}

pub fn read_object_from_file<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    read_from_file_bitcode(path)
}

pub fn write_object_to_file<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    write_to_file_bitcode(path, data)
}

pub(crate) fn read_from_file_bitcode<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let data = std::fs::read(path)?;
    let ret = bitcode::deserialize(&data)?;
    Ok(ret)
}

pub(crate) fn write_to_file_bitcode<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
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
