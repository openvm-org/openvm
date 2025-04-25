use std::{
    fs::read,
    path::{Path, PathBuf},
};

use eyre::{Context, Result};
use openvm_build::{
    build_guest_package, get_dir_with_profile, get_package, GuestOptions, TargetFilter,
};
use openvm_circuit::arch::{InitFileGenerator, OPENVM_DEFAULT_INIT_FILE_BASENAME};
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

// Some tests need to manually override the init macro build script (e.g. to test invalid moduli),
// so we can use this struct to avoid generating an init file
pub struct NoInitFile;
impl InitFileGenerator for NoInitFile {}

pub fn build_example_program(
    example_name: &str,
    init_config: &impl InitFileGenerator,
) -> Result<Elf> {
    build_example_program_with_features::<&str>(example_name, [], init_config)
}

pub fn build_example_program_with_features<S: AsRef<str>>(
    example_name: &str,
    features: impl IntoIterator<Item = S> + Clone,
    init_config: &impl InitFileGenerator,
) -> Result<Elf> {
    let manifest_dir = get_programs_dir!();
    build_example_program_at_path_with_features(manifest_dir, example_name, features, init_config)
}

pub fn build_example_program_at_path(
    manifest_dir: PathBuf,
    example_name: &str,
    init_config: &impl InitFileGenerator,
) -> Result<Elf> {
    build_example_program_at_path_with_features::<&str>(manifest_dir, example_name, [], init_config)
}

pub fn build_example_program_at_path_with_features<S: AsRef<str>>(
    manifest_dir: PathBuf,
    example_name: &str,
    features: impl IntoIterator<Item = S> + Clone,
    init_config: &impl InitFileGenerator,
) -> Result<Elf> {
    let pkg = get_package(&manifest_dir);
    let target_dir = tempdir()?;
    // Build guest with default features
    let guest_opts = GuestOptions::default()
        .with_features(features.clone())
        .with_target_dir(target_dir.path());
    let features = features
        .into_iter()
        .map(|x| x.as_ref().to_string())
        .collect::<Vec<_>>();
    let features_str = if !features.is_empty() {
        format!("_{}", features.join("_"))
    } else {
        "".to_string()
    };
    init_config.write_to_init_file(
        &manifest_dir,
        Some(&format!(
            "{}_{}{}.rs",
            OPENVM_DEFAULT_INIT_FILE_BASENAME, example_name, features_str
        )),
    )?;
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
    let profile = "release";
    let elf_path = pkg
        .targets
        .iter()
        .find(|target| target.name == example_name)
        .map(|target| {
            get_dir_with_profile(&target_dir, profile, true)
                .join(&target.name)
                .to_path_buf()
        })
        .expect("Could not find target binary");
    let data = read(&elf_path).with_context(|| format!("Path not found: {:?}", elf_path))?;
    target_dir.close()?;
    Elf::decode(&data, MEM_SIZE as u32)
}
