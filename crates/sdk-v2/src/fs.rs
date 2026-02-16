use std::{
    fs::{create_dir_all, read, write, File},
    path::Path,
};

use eyre::{Report, Result};
use openvm_stark_backend::codec::{Decode, Encode};
use serde::{de::DeserializeOwned, Serialize};

pub fn read_object_from_file<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    read_from_file_bitcode(path)
}

pub fn write_object_to_file<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
    write_to_file_bitcode(path, data)
}

fn read_from_file_bitcode<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T> {
    let ret = read(&path)
        .map_err(|e| read_error(&path, e.into()))
        .and_then(|data| {
            bitcode::deserialize(&data).map_err(|e: bitcode::Error| read_error(&path, e.into()))
        })?;
    Ok(ret)
}

fn write_to_file_bitcode<T: Serialize, P: AsRef<Path>>(path: P, data: T) -> Result<()> {
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
