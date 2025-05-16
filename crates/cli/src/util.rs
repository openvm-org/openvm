use std::{
    fs::read_to_string,
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    fs::{read_agg_halo2_pk_from_file, read_agg_stark_pk_from_file},
    keygen::AggProvingKey,
};
use serde::de::DeserializeOwned;

use crate::default::{default_agg_halo2_pk_path, default_agg_stark_pk_path, default_app_config};

pub(crate) fn read_to_struct_toml<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let toml = read_to_string(path)?;
    let ret = toml::from_str(&toml)?;
    Ok(ret)
}

pub fn read_config_toml_or_default(config: impl AsRef<Path>) -> Result<AppConfig<SdkVmConfig>> {
    if config.as_ref().exists() {
        read_to_struct_toml(config)
    } else {
        println!(
            "{:?} not found, using default application configuration",
            config.as_ref()
        );
        Ok(default_app_config())
    }
}

pub fn read_default_agg_pk() -> Result<AggProvingKey> {
    let agg_stark_pk = read_agg_stark_pk_from_file(default_agg_stark_pk_path())?;
    let halo2_pk = read_agg_halo2_pk_from_file(default_agg_halo2_pk_path())?;
    Ok(AggProvingKey {
        agg_stark_pk,
        halo2_pk,
    })
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
