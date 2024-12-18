use std::{
    fmt::Display,
    fs::{read, read_to_string},
    path::{Path, PathBuf},
    str::FromStr,
};

use eyre::Result;
use openvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    StdIn,
};
use serde::de::DeserializeOwned;

use crate::default::default_app_config;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) enum Input {
    FilePath(PathBuf),
    HexBytes(Vec<u8>),
}

impl FromStr for Input {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if is_valid_hex_string(s) {
            // Remove 0x prefix if present
            let s = if s.starts_with("0x") {
                s.strip_prefix("0x").unwrap()
            } else {
                s
            };
            if s.is_empty() {
                return Ok(Input::HexBytes(Vec::new()));
            }
            if !s.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err("Invalid hex string.".to_string());
            }
            let bytes = hex::decode(s).map_err(|e| e.to_string())?;
            Ok(Input::HexBytes(bytes))
        } else if PathBuf::from(s).exists() {
            Ok(Input::FilePath(PathBuf::from(s)))
        } else {
            Err("Input must be a valid file path or hex string.".to_string())
        }
    }
}

pub(crate) fn is_valid_hex_string(s: &str) -> bool {
    if s.len() % 2 != 0 {
        return false;
    }
    // All hex digits with optional 0x prefix
    s.starts_with("0x") && s[2..].chars().all(|c| c.is_ascii_hexdigit())
        || s.chars().all(|c| c.is_ascii_hexdigit())
}

pub(crate) fn write_status(style: &dyn Display, status: &str, msg: &str) {
    println!("{style}{status:>12}{style:#} {msg}");
}

pub(crate) fn classical_exe_path(elf_path: &Path) -> PathBuf {
    elf_path.with_extension("vmexe")
}

pub(crate) fn read_to_struct_toml<T: DeserializeOwned>(path: &PathBuf) -> Result<T> {
    let toml = read_to_string(path.as_ref() as &Path)?;
    let ret = toml::from_str(&toml)?;
    Ok(ret)
}

pub(crate) fn read_to_stdin(input: &Option<Input>) -> Result<StdIn> {
    match input {
        Some(Input::FilePath(path)) => {
            let bytes = read(path)?;
            Ok(StdIn::from_bytes(&bytes))
        }
        Some(Input::HexBytes(bytes)) => Ok(StdIn::from_bytes(bytes)),
        None => Ok(StdIn::default()),
    }
}

pub(crate) fn read_config_toml_or_default(config: &PathBuf) -> Result<AppConfig<SdkVmConfig>> {
    if config.exists() {
        read_to_struct_toml(config)
    } else {
        println!(
            "{:?} not found, using default application configuration",
            config
        );
        Ok(default_app_config())
    }
}
