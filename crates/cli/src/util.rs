use std::{
    fmt::Display,
    fs::read_to_string,
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_sdk::config::{AppConfig, SdkVmConfig};
use serde::de::DeserializeOwned;

use crate::default::default_app_config;

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
