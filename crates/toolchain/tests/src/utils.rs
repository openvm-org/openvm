use std::{
    fs::read,
    path::{Path, PathBuf},
};

use axvm_build::{build_guest_package, get_package, is_debug, GuestOptions};
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf};
use eyre::Result;
use tempfile::tempdir;

fn get_programs_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    dir.push("programs");
    dir
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
    let manifest_dir = get_programs_dir();
    let pkg = get_package(manifest_dir);
    let target_dir = tempdir()?;
    // Build guest with default features
    let guest_opts = GuestOptions::default()
        .with_options(["--example", example_name])
        .with_features(features)
        .into();
    build_guest_package(&pkg, &target_dir, &guest_opts, None);
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
