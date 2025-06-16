use std::{
    fs::{read_dir, read_to_string},
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_build::{get_in_scope_packages, get_workspace_packages};
use openvm_sdk::config::{AppConfig, SdkVmConfig};
#[cfg(feature = "evm-prove")]
use openvm_sdk::{fs::read_agg_stark_pk_from_file, keygen::AggProvingKey};
use serde::de::DeserializeOwned;

use crate::{
    commands::RunCargoArgs,
    default::{default_app_config, DEFAULT_APP_PK_NAME, DEFAULT_APP_VK_NAME},
};

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

#[cfg(feature = "evm-prove")]
pub fn read_default_agg_pk() -> Result<AggProvingKey> {
    let agg_stark_pk = read_agg_stark_pk_from_file(crate::default::default_agg_stark_pk_path())?;
    let halo2_pk =
        openvm_sdk::fs::read_agg_halo2_pk_from_file(crate::default::default_agg_halo2_pk_path())?;
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

pub fn get_manifest_path_and_dir(manifest_path: &Option<PathBuf>) -> Result<(PathBuf, PathBuf)> {
    let manifest_dir = if let Some(manifest_path) = &manifest_path {
        if !manifest_path.ends_with("Cargo.toml") {
            return Err(eyre::eyre!(
                "manifest_path must be a path to a Cargo.toml file"
            ));
        }
        manifest_path.parent().unwrap().canonicalize()?
    } else {
        find_manifest_dir(PathBuf::from("."))?
    };
    let manifest_path = manifest_dir.join("Cargo.toml");
    Ok((manifest_path.clone(), manifest_dir))
}

pub fn get_target_dir(target_dir: &Option<PathBuf>, manifest_path: &PathBuf) -> PathBuf {
    target_dir
        .clone()
        .unwrap_or_else(|| openvm_build::get_target_dir(manifest_path))
}

pub fn get_target_output_dir(target_dir: &Path, profile: &str) -> PathBuf {
    target_dir.join("openvm").join(profile).to_path_buf()
}

pub fn get_app_pk_path(target_dir: &Path) -> PathBuf {
    target_dir.join("openvm").join(DEFAULT_APP_PK_NAME)
}

pub fn get_app_vk_path(target_dir: &Path) -> PathBuf {
    target_dir.join("openvm").join(DEFAULT_APP_VK_NAME)
}

// Given the arguments to a run command, this function isolates the executable to
// run. If a specific binary or example is specified it will return that, else it
// will search the workspace/package for binary targets. If there is a single
// binary that will be returned, else an error will be raised.
pub fn get_single_target_name(cargo_args: &RunCargoArgs) -> Result<String> {
    let num_targets = cargo_args.bin.len() + cargo_args.example.len();
    let single_target_name = if num_targets > 1 {
        return Err(eyre::eyre!(
            "`cargo openvm run` can run at most one executable, but multiple were specified"
        ));
    } else if num_targets == 0 {
        let (_, manifest_dir) = get_manifest_path_and_dir(&cargo_args.manifest_path)?;

        let packages = if cargo_args.package.is_some() {
            get_workspace_packages(&manifest_dir)
        } else {
            get_in_scope_packages(&manifest_dir)
        }
        .into_iter()
        .filter(|pkg| {
            if let Some(package) = &cargo_args.package {
                pkg.name == *package
            } else {
                true
            }
        })
        .collect::<Vec<_>>();

        let binaries = packages
            .iter()
            .flat_map(|pkg| pkg.targets.iter())
            .filter(|t| t.is_bin())
            .collect::<Vec<_>>();

        if binaries.len() > 1 {
            return Err(eyre::eyre!(
                "Could not determine which binary to run. Use the --bin flag to specify.\n\
                    Available targets: {:?}",
                binaries.iter().map(|t| t.name.clone()).collect::<Vec<_>>()
            ));
        } else if binaries.is_empty() {
            return Err(eyre::eyre!(
                "No binaries found. If you would like to run an example, use the --example flag.",
            ));
        } else {
            binaries[0].name.clone()
        }
    } else if cargo_args.bin.is_empty() {
        format!("examples/{}", cargo_args.example[0])
    } else {
        cargo_args.bin[0].clone()
    };
    Ok(single_target_name)
}

pub fn get_files_with_ext(dir: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    let dir = dir.canonicalize()?;
    let mut files = Vec::new();
    for entry in read_dir(dir)? {
        let path = entry?.path();
        if path.is_file()
            && path
                .to_str()
                .is_some_and(|path_str| path_str.ends_with(extension))
        {
            files.push(path);
        }
    }
    Ok(files)
}
