use std::path::PathBuf;

use axvm_build::{build_guest_package, get_package, guest_methods, GuestOptions};
use clap::Parser;
use eyre::Result;

#[derive(Parser)]
#[command(name = "build", about = "Compile an axVM program")]
pub struct BuildCmd {
    #[clap(flatten)]
    build_args: BuildArgs,
}

impl BuildCmd {
    pub fn run(&self) -> Result<()> {
        build(&self.build_args)?;
        Ok(())
    }
}

#[derive(Parser)]
pub struct BuildArgs {
    /// Location of the directory containing the Cargo.toml for the guest code.
    ///
    /// This path is relative to the current directory.
    #[arg(long)]
    pub manifest_dir: Option<PathBuf>,

    /// Feature flags passed to cargo.
    #[arg(long, value_delimiter = ',')]
    pub features: Vec<String>,
}

// Returns elf_path for now
pub(crate) fn build(build_args: &BuildArgs) -> Result<Vec<PathBuf>> {
    let manifest_dir = build_args
        .manifest_dir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    let guest_options = GuestOptions {
        features: build_args.features.clone(),
        ..Default::default()
    };
    let target_dir = manifest_dir.join("target");
    let pkg = get_package(&manifest_dir);
    build_guest_package(&pkg, &target_dir, &guest_options.into(), None);
    // Assumes the package has a single target binary
    let elf_paths = guest_methods(&pkg, &target_dir, &[]);
    Ok(elf_paths)
}
