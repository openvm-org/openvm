use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_sdk::config::{AppConfig, SdkVmConfig};
use serde::de::DeserializeOwned;

use crate::default::default_app_config;

pub(crate) fn read_to_struct_toml<T: DeserializeOwned>(path: &PathBuf) -> Result<T> {
    let toml = read_to_string(path.as_ref() as &Path)?;
    let ret = toml::from_str(&toml)?;
    Ok(ret)
}

pub fn read_config_toml_or_default(config: &PathBuf) -> Result<AppConfig<SdkVmConfig>> {
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

pub fn find_manifest_dir(mut current_dir: PathBuf) -> Result<PathBuf> {
    current_dir = current_dir.canonicalize()?;
    while !current_dir.join("Cargo.toml").exists() {
        current_dir = current_dir
            .parent()
            .expect("Could not find Cargo.toml in current directory or any parent directory")
            .to_path_buf();
    }
    Ok(current_dir)
}
