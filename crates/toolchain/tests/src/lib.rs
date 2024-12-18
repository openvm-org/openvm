use std::{
    fs::read,
    path::{Path, PathBuf},
};

use eyre::Result;
use openvm_build::{build_guest_package, get_package, is_debug, GuestOptions, TargetFilter};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};
use tempfile::tempdir;

#[macro_export]
macro_rules! get_programs_dir {
    () => {{
        let mut dir = ::std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        dir.push("programs");
        dir
    }};
    ($subdir:expr) => {{
        let mut dir = ::std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
        dir.push($subdir);
        dir
    }};
}

pub fn decode_elf(elf_path: impl AsRef<Path>) -> Result<Elf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    Elf::decode(&data, MEM_SIZE as u32)
}

pub fn build_example_program(example_name: &str) -> Result<Elf> {
    build_example_program_with_features::<&str>(example_name, [])
}

pub fn build_example_program_with_features<S: AsRef<str>>(
    example_name: &str,
    features: impl IntoIterator<Item = S>,
) -> Result<Elf> {
    let manifest_dir = get_programs_dir!();
    build_example_program_at_path_with_features(manifest_dir, example_name, features)
}

pub fn build_example_program_at_path(manifest_dir: PathBuf, example_name: &str) -> Result<Elf> {
    build_example_program_at_path_with_features::<&str>(manifest_dir, example_name, [])
}

pub fn build_example_program_at_path_with_features<S: AsRef<str>>(
    manifest_dir: PathBuf,
    example_name: &str,
    features: impl IntoIterator<Item = S>,
) -> Result<Elf> {
    let pkg = get_package(manifest_dir);
    let target_dir = tempdir()?;
    // Build guest with default features
    let guest_opts = GuestOptions::default()
        .with_features(features)
        .with_target_dir(target_dir.path());
    if let Err(Some(code)) = build_guest_package(
        &pkg,
        &guest_opts,
        None,
        &Some(TargetFilter {
            name: example_name.to_string(),
            kind: "example".to_string(),
        }),
    ) {
        std::process::exit(code);
    }
    // Assumes the package has a single target binary
    let profile = if is_debug() { "debug" } else { "release" };
    let elf_path = pkg
        .targets
        .iter()
        .find(|target| target.name == example_name)
        .map(|target| {
            target_dir
                .as_ref()
                .join("riscv32im-risc0-zkvm-elf")
                .join(profile)
                .join("examples")
                .join(&target.name)
                .to_path_buf()
        })
        .expect("Could not find target binary");
    let data = read(elf_path)?;
    Elf::decode(&data, MEM_SIZE as u32)
}
