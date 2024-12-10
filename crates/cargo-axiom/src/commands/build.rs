use std::path::PathBuf;

use axvm_build::GuestOptions;
use axvm_sdk::{Sdk, TargetFilter};
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

    #[clap(flatten)]
    pub bin_type_filter: BinTypeFilter,

    /// Target name substring filter
    #[arg(long)]
    pub name: Option<String>,
}

#[derive(clap::Args)]
#[group(required = false, multiple = false)]
pub struct BinTypeFilter {
    /// Specify that the target should be a binary kind
    #[arg(long)]
    pub bin: bool,

    /// Specify that the target should be an example kind
    #[arg(long)]
    pub example: bool,
}

// Returns elf_path for now
pub(crate) fn build(build_args: &BuildArgs) -> Result<PathBuf> {
    let target_filter = TargetFilter {
        name_substr: build_args.name.clone(),
        kind: if build_args.bin_type_filter.bin {
            Some("bin".to_string())
        } else if build_args.bin_type_filter.example {
            Some("example".to_string())
        } else {
            None
        },
    };
    let sdk = Sdk;
    let pkg_dir = build_args
        .manifest_dir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    let guest_options = GuestOptions {
        features: build_args.features.clone(),
        ..Default::default()
    };
    let (_, elf_path) = sdk.build(guest_options, &pkg_dir, target_filter)?;
    Ok(elf_path)
}
