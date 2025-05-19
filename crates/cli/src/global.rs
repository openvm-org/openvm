use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use eyre::Result;
use openvm_build::get_workspace_packages;

use crate::{
    commands::RunCargoArgs,
    default::{DEFAULT_APP_PK_NAME, DEFAULT_APP_VK_NAME},
    util::find_manifest_dir,
};

static MANIFEST_PATH: OnceLock<PathBuf> = OnceLock::new();
static MANIFEST_DIR: OnceLock<PathBuf> = OnceLock::new();
static TARGET_DIR: OnceLock<PathBuf> = OnceLock::new();
static SINGLE_TARGET_NAME: OnceLock<String> = OnceLock::new();

pub(crate) fn get_manifest_path_and_dir(
    manifest_path: &Option<PathBuf>,
) -> Result<(PathBuf, PathBuf)> {
    let manifest_dir = if MANIFEST_DIR.get().is_some() {
        MANIFEST_DIR.get().unwrap().clone()
    } else if let Some(manifest_path) = &manifest_path {
        if !manifest_path.ends_with("Cargo.toml") {
            return Err(eyre::eyre!(
                "manifest_path must be a path to a Cargo.toml file"
            ));
        }
        let canonical_path = manifest_path.canonicalize()?;
        let _ = MANIFEST_DIR.set(canonical_path.parent().unwrap().to_path_buf());
        MANIFEST_DIR.get().unwrap().clone()
    } else {
        find_manifest_dir(PathBuf::from("."))?
    };
    let manifest_path = MANIFEST_PATH.get_or_init(|| manifest_dir.join("Cargo.toml"));
    Ok((manifest_path.clone(), manifest_dir))
}

pub(crate) fn get_target_dir(target_dir: &Option<PathBuf>, manifest_path: &PathBuf) -> PathBuf {
    TARGET_DIR
        .get_or_init(|| {
            target_dir
                .clone()
                .unwrap_or_else(|| openvm_build::get_target_dir(manifest_path))
        })
        .to_path_buf()
}

pub(crate) fn get_target_output_dir(target_dir: &Path, profile: &str) -> PathBuf {
    target_dir.join("openvm").join(profile).to_path_buf()
}

pub(crate) fn get_app_pk_path(target_dir: &Path) -> PathBuf {
    target_dir.join("openvm").join(DEFAULT_APP_PK_NAME)
}

pub(crate) fn get_app_vk_path(target_dir: &Path) -> PathBuf {
    target_dir.join("openvm").join(DEFAULT_APP_VK_NAME)
}

pub(crate) fn get_single_target_name(cargo_args: &RunCargoArgs) -> Result<String> {
    if SINGLE_TARGET_NAME.get().is_some() {
        Ok(SINGLE_TARGET_NAME.get().unwrap().clone())
    } else {
        let num_targets = cargo_args.bin.len() + cargo_args.example.len();
        let single_target_name = if num_targets > 1 {
            return Err(eyre::eyre!(
                "`cargo openvm run` can run at most one executable, but multiple were specified"
            ));
        } else if num_targets == 0 {
            let (_, manifest_dir) = get_manifest_path_and_dir(&cargo_args.manifest_path)?;

            let packages = get_workspace_packages(&manifest_dir)
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
        let _ = SINGLE_TARGET_NAME.set(single_target_name.clone());
        Ok(single_target_name)
    }
}
