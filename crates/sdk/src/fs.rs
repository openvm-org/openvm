use std::{
    fs::{create_dir_all, read, write, File},
    path::Path,
};

use eyre::{Report, Result};
use openvm_circuit::arch::{instructions::exe::VmExe, ContinuationVmProof, VmConfig};
use openvm_continuations::verifier::root::types::RootVmVerifierInput;
#[cfg(feature = "evm-prove")]
use openvm_native_recursion::halo2::wrapper::EvmVerifierByteCode;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    codec::{Decode, Encode},
    keygen::{AggStarkProvingKey, AppProvingKey, AppVerifyingKey},
    F, SC,
};
#[cfg(feature = "evm-prove")]
use crate::{
    keygen::Halo2ProvingKey,
    types::{EvmHalo2Verifier, EvmProof},
    OPENVM_VERSION,
};

pub const EVM_HALO2_VERIFIER_INTERFACE_NAME: &str = "IOpenVmHalo2Verifier.sol";
pub const EVM_HALO2_VERIFIER_PARENT_NAME: &str = "Halo2Verifier.sol";
pub const EVM_HALO2_VERIFIER_BASE_NAME: &str = "OpenVmHalo2Verifier.sol";
pub const EVM_VERIFIER_ARTIFACT_FILENAME: &str = "verifier.bytecode.json";

pub fn read_exe_from_file<P: AsRef<Path>>(path: P) -> Result<VmExe<F>> {
    read_from_file_bitcode(&path)
}

pub fn write_exe_to_file<P: AsRef<Path>>(exe: VmExe<F>, path: P) -> Result<()> {
    write_to_file_bitcode(&path, exe)
}

pub fn read_app_pk_from_file<VC: VmConfig<F>, P: AsRef<Path>>(
    path: P,
) -> Result<AppProvingKey<VC>> {
    read_from_file_bitcode(&path)
}

pub fn write_app_pk_to_file<VC: VmConfig<F>, P: AsRef<Path>>(
    app_pk: AppProvingKey<VC>,
    path: P,
) -> Result<()> {
    write_to_file_bitcode(&path, app_pk)
}

pub fn read_app_vk_from_file<P: AsRef<Path>>(path: P) -> Result<AppVerifyingKey> {
    read_from_file_bitcode(&path)
}

pub fn write_app_vk_to_file<P: AsRef<Path>>(app_vk: AppVerifyingKey, path: P) -> Result<()> {
    write_to_file_bitcode(&path, app_vk)
}

pub fn read_app_proof_from_file<P: AsRef<Path>>(path: P) -> Result<ContinuationVmProof<SC>> {
    decode_from_file(&path)
}

pub fn write_app_proof_to_file<P: AsRef<Path>>(
    proof: ContinuationVmProof<SC>,
    path: P,
) -> Result<()> {
    encode_to_file(&path, proof)
}

pub fn read_root_verifier_input_from_file<P: AsRef<Path>>(
    path: P,
) -> Result<RootVmVerifierInput<SC>> {
    decode_from_file(&path)
}

pub fn write_root_verifier_input_to_file<P: AsRef<Path>>(
    input: RootVmVerifierInput<SC>,
    path: P,
) -> Result<()> {
    encode_to_file(&path, input)
}

pub fn read_agg_stark_pk_from_file<P: AsRef<Path>>(path: P) -> Result<AggStarkProvingKey> {
    read_from_file_bitcode(&path)
}

pub fn write_agg_stark_pk_to_file<P: AsRef<Path>>(pk: &AggStarkProvingKey, path: P) -> Result<()> {
    write_to_file_bitcode(&path, pk)
}

#[cfg(feature = "evm-prove")]
pub fn read_agg_halo2_pk_from_file<P: AsRef<Path>>(path: P) -> Result<Halo2ProvingKey> {
    read_from_file_bitcode(&path)
}

#[cfg(feature = "evm-prove")]
pub fn write_agg_halo2_pk_to_file<P: AsRef<Path>>(pk: &Halo2ProvingKey, path: P) -> Result<()> {
    write_to_file_bitcode(&path, pk)
}

#[cfg(feature = "evm-prove")]
pub fn read_evm_proof_from_file<P: AsRef<Path>>(path: P) -> Result<EvmProof> {
    read_from_file_json(&path)
}

#[cfg(feature = "evm-prove")]
pub fn write_evm_proof_to_file<P: AsRef<Path>>(proof: EvmProof, path: P) -> Result<()> {
    write_to_file_json(&path, proof)
}

#[cfg(feature = "evm-prove")]
pub fn read_evm_halo2_verifier_from_folder<P: AsRef<Path>>(folder: P) -> Result<EvmHalo2Verifier> {
    use std::fs::read_to_string;

    let folder = folder
        .as_ref()
        .join("src")
        .join(format!("v{}", OPENVM_VERSION));
    let halo2_verifier_code_path = folder.join(EVM_HALO2_VERIFIER_PARENT_NAME);
    let openvm_verifier_code_path = folder.join(EVM_HALO2_VERIFIER_BASE_NAME);
    let interface_path = folder
        .join("interfaces")
        .join(EVM_HALO2_VERIFIER_INTERFACE_NAME);
    let halo2_verifier_code = read_to_string(halo2_verifier_code_path)?;
    let openvm_verifier_code = read_to_string(openvm_verifier_code_path)?;
    let interface = read_to_string(interface_path)?;

    let artifact_path = folder.join(EVM_VERIFIER_ARTIFACT_FILENAME);
    let artifact: EvmVerifierByteCode = serde_json::from_reader(File::open(artifact_path)?)?;

    Ok(EvmHalo2Verifier {
        halo2_verifier_code,
        openvm_verifier_code,
        openvm_verifier_interface: interface,
        artifact,
    })
}

/// Writes three Solidity contracts into the following folder structure:
///
/// ```text
/// halo2/
/// └── src/
///     └── v[OPENVM_VERSION]/
///         ├── interfaces/
///         │   └── IOpenVmHalo2Verifier.sol
///         ├── OpenVmHalo2Verifier.sol
///         └── Halo2Verifier.sol
/// ```
///
/// If the relevant directories do not exist, they will be created.
#[cfg(feature = "evm-prove")]
pub fn write_evm_halo2_verifier_to_folder<P: AsRef<Path>>(
    verifier: EvmHalo2Verifier,
    folder: P,
) -> Result<()> {
    let folder = folder
        .as_ref()
        .join("src")
        .join(format!("v{}", OPENVM_VERSION));
    if !folder.exists() {
        create_dir_all(&folder)?; // Make sure directories exist
    }

    let halo2_verifier_code_path = folder.join(EVM_HALO2_VERIFIER_PARENT_NAME);
    let openvm_verifier_code_path = folder.join(EVM_HALO2_VERIFIER_BASE_NAME);
    let interface_path = folder
        .join("interfaces")
        .join(EVM_HALO2_VERIFIER_INTERFACE_NAME);

    if let Some(parent) = interface_path.parent() {
        create_dir_all(parent)?;
    }

    write(halo2_verifier_code_path, verifier.halo2_verifier_code)
        .expect("Failed to write halo2 verifier code");
    write(openvm_verifier_code_path, verifier.openvm_verifier_code)
        .expect("Failed to write openvm halo2 verifier code");
    write(interface_path, verifier.openvm_verifier_interface)
        .expect("Failed to write openvm halo2 verifier interface");

    let artifact_path = folder.join(EVM_VERIFIER_ARTIFACT_FILENAME);
    serde_json::to_writer(File::create(artifact_path)?, &verifier.artifact)?;

    Ok(())
}

pub fn read_object_from_file<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    read_from_file_bitcode(path)
}

pub fn write_object_to_file<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    write_to_file_bitcode(path, data)
}

pub fn read_from_file_bitcode<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let ret = read(&path)
        .map_err(|e| read_error(&path, e.into()))
        .and_then(|data| {
            bitcode::deserialize(&data).map_err(|e: bitcode::Error| read_error(&path, e.into()))
        })?;
    Ok(ret)
}

pub fn write_to_file_bitcode<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        create_dir_all(parent).map_err(|e| write_error(&path, e.into()))?;
    }
    bitcode::serialize(&data)
        .map_err(|e| write_error(&path, e.into()))
        .and_then(|bytes| write(&path, bytes).map_err(|e| write_error(&path, e.into())))?;
    Ok(())
}

pub fn read_from_file_json<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let ret: T = File::open(&path)
        .and_then(|file| serde_json::from_reader(file).map_err(|e| e.into()))
        .map_err(|e| read_error(&path, e.into()))?;
    Ok(ret)
}

pub fn write_to_file_json<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        create_dir_all(parent).map_err(|e| write_error(&path, e.into()))?;
    }
    File::create(&path)
        .and_then(|file| serde_json::to_writer_pretty(file, &data).map_err(|e| e.into()))
        .map_err(|e| write_error(&path, e.into()))?;
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

pub fn decode_from_file<T: Decode, P: AsRef<Path>>(path: P) -> Result<T> {
    let reader = &mut File::open(path)?;
    let ret = T::decode(reader)?;
    Ok(ret)
}

pub fn encode_to_file<T: Encode, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        create_dir_all(parent)?;
    }
    let writer = &mut File::create(path)?;
    data.encode(writer)?;
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
