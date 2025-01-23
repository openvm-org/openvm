use std::{
    fs::{read, read_to_string},
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use openvm_stark_sdk::config::FriParameters;
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use serde::de::DeserializeOwned;
use tempfile::tempdir;

use crate::config::{AppConfig, SdkVmConfig};

pub fn build_program(program_dir: impl AsRef<Path>, profile: impl ToString) -> Result<Elf> {
    let pkg = get_package(program_dir);
    let target_dir = tempdir()?;
    // Build guest with default features
    let guest_opts = GuestOptions::default()
        .with_target_dir(target_dir.path())
        .with_profile(profile.to_string());
    if let Err(Some(code)) = build_guest_package(&pkg, &guest_opts, None, &None) {
        std::process::exit(code);
    }
    // Assumes the package has a single target binary
    let elf_path = guest_methods(&pkg, &target_dir, &guest_opts.features, &guest_opts.profile)
        .pop()
        .unwrap();
    let data = read(elf_path)?;
    Elf::decode(&data, MEM_SIZE as u32)
}

pub fn read_to_struct_toml<T: DeserializeOwned>(path: &PathBuf) -> Result<T> {
    let toml = read_to_string(path.as_ref() as &Path)?;
    let ret = toml::from_str(&toml)?;
    Ok(ret)
}

pub fn default_app_config() -> AppConfig<SdkVmConfig> {
    AppConfig {
        app_fri_params: FriParameters::standard_with_100_bits_conjectured_security(2).into(),
        app_vm_config: SdkVmConfig::builder()
            .system(Default::default())
            .rv32i(Default::default())
            .rv32m(Default::default())
            .io(Default::default())
            .build(),
        leaf_fri_params: FriParameters::standard_with_100_bits_conjectured_security(2).into(),
        compiler_options: Default::default(),
    }
}
