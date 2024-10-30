use std::{fs::read, path::PathBuf};

use axvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf};
use eyre::Result;
use tempfile::tempdir;

fn get_programs_dir() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    dir.push("programs");
    dir
}

pub fn build_bench_program(program_name: &str) -> Result<Elf> {
    let manifest_dir = get_programs_dir().join(program_name);
    let pkg = get_package(manifest_dir);
    let target_dir = tempdir()?;
    // Build guest with default features
    let guest_opts = GuestOptions::default().into();
    build_guest_package(&pkg, &target_dir, &guest_opts, None);
    // Assumes the package has a single target binary
    let elf_path = guest_methods(&pkg, &target_dir, &[]).pop().unwrap();
    let data = read(elf_path)?;
    Elf::decode(&data, MEM_SIZE as u32)
}
