use std::{
    fs::read,
    path::{Path, PathBuf},
};

use cargo_metadata::Package;
use eyre::Result;
use openvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use tempfile::tempdir;

pub fn get_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
}

pub fn get_programs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../guest")
}

pub fn build_elf(manifest_dir: &PathBuf, profile: impl ToString) -> Result<Elf> {
    let pkg = get_package(manifest_dir);
    build_elf_with_path(&pkg, profile, None)
}

pub fn build_elf_with_path(
    pkg: &Package,
    profile: impl ToString,
    elf_path: Option<&PathBuf>,
) -> Result<Elf> {
    // Use a temporary directory for the build
    let temp_dir = tempdir()?;
    let target_dir = temp_dir.path();

    // Build guest with default features
    let guest_opts = GuestOptions::default()
        .with_target_dir(target_dir)
        .with_profile(profile.to_string());

    if let Err(Some(code)) = build_guest_package(pkg, &guest_opts, None, &None) {
        std::process::exit(code);
    }

    // Assumes the package has a single target binary
    let temp_elf_path = guest_methods(pkg, target_dir, &guest_opts.features, &guest_opts.profile)
        .pop()
        .unwrap();

    // If an elf_path is provided, copy the built ELF to that location
    if let Some(dest_path) = elf_path {
        // Create parent directories if they don't exist
        if let Some(parent) = dest_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }

        // Copy the built ELF to the destination
        std::fs::copy(&temp_elf_path, dest_path)?;
    }

    read_elf_file(&temp_elf_path)
}

pub fn get_elf_path(manifest_dir: &PathBuf) -> PathBuf {
    let pkg = get_package(manifest_dir);
    get_elf_path_with_pkg(manifest_dir, &pkg)
}

pub fn get_elf_path_with_pkg(manifest_dir: &Path, pkg: &Package) -> PathBuf {
    let elf_file_name = format!("{}.elf", &pkg.name);
    manifest_dir.join("elf").join(elf_file_name)
}

pub fn read_elf_file(elf_path: &PathBuf) -> Result<Elf> {
    let data = read(elf_path)?;
    Elf::decode(&data, MEM_SIZE as u32)
}
