use std::{fs::read, path::PathBuf};

use clap::Parser;
use eyre::Result;
use openvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
use openvm_sdk::{fs::write_exe_to_file, Sdk};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};

use crate::{
    default::{DEFAULT_APP_CONFIG_PATH, DEFAULT_APP_EXE_PATH, DEFAULT_MANIFEST_DIR},
    util::read_config_toml_or_default,
};

#[derive(Parser)]
#[command(name = "build", about = "Compile an OpenVM program")]
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

#[derive(Clone, Parser)]
pub struct BuildArgs {
    #[arg(
        long,
        help = "Path to the directory containing the Cargo.toml file for the guest code (relative to the current directory)",
        default_value = DEFAULT_MANIFEST_DIR
    )]
    pub manifest_dir: PathBuf,

    #[arg(long, value_delimiter = ',', help = "Feature flags passed to cargo")]
    pub features: Vec<String>,

    #[clap(flatten, help = "Filter the target to build")]
    pub bin_type_filter: BinTypeFilter,

    #[arg(
        long,
        default_value = "false",
        help = "Skips transpilation into exe when set"
    )]
    pub no_transpile: bool,

    #[arg(
        long,
        default_value = DEFAULT_APP_CONFIG_PATH,
        help = "Path to the SDK config .toml file that specifies the transpiler extensions"
    )]
    pub config: PathBuf,

    #[arg(
        long,
        default_value = DEFAULT_APP_EXE_PATH,
        help = "Output path for the transpiled program"
    )]
    pub exe_output: PathBuf,

    #[arg(long, default_value = "release", help = "Build profile")]
    pub profile: String,
}

#[derive(Clone, clap::Args)]
#[group(required = false, multiple = false)]
pub struct BinTypeFilter {
    #[arg(long, help = "Specifies that the bin target to build")]
    pub bin: Option<String>,

    #[arg(long, help = "Specifies that the example target to build")]
    pub example: Option<String>,
}

// Returns the path to the ELF file if it is unique.
pub(crate) fn build(build_args: &BuildArgs) -> Result<Option<PathBuf>> {
    println!("[openvm] Building the package...");
    let target_filter = if let Some(bin) = &build_args.bin_type_filter.bin {
        Some(TargetFilter {
            name: bin.clone(),
            kind: "bin".to_string(),
        })
    } else {
        build_args
            .bin_type_filter
            .example
            .as_ref()
            .map(|example| TargetFilter {
                name: example.clone(),
                kind: "example".to_string(),
            })
    };
    let guest_options = GuestOptions {
        features: build_args.features.clone(),
        ..Default::default()
    };

    let pkg = get_package(&build_args.manifest_dir);
    // We support builds of libraries with 0 or >1 executables.
    let elf_path = match build_guest_package(&pkg, &guest_options, None, &target_filter) {
        Ok(target_dir) => {
            find_unique_executable(&build_args.manifest_dir, &target_dir, &target_filter)
        }
        Err(None) => {
            return Err(eyre::eyre!("Failed to build guest"));
        }
        Err(Some(code)) => {
            return Err(eyre::eyre!("Failed to build guest: code = {}", code));
        }
    };

    if !build_args.no_transpile {
        let elf_path = elf_path?;
        println!("[openvm] Transpiling the package...");
        let output_path = &build_args.exe_output;
        let app_config = read_config_toml_or_default(&build_args.config)?;
        let transpiler = app_config.app_vm_config.transpiler();

        let data = read(elf_path.clone())?;
        let elf = Elf::decode(&data, MEM_SIZE as u32)?;
        let exe = Sdk.transpile(elf, transpiler)?;
        write_exe_to_file(exe, output_path)?;

        println!(
            "[openvm] Successfully transpiled to {}",
            output_path.display()
        );
        Ok(Some(elf_path))
    } else if let Ok(elf_path) = elf_path {
        println!(
            "[openvm] Successfully built the package: {}",
            elf_path.display()
        );
        Ok(Some(elf_path))
    } else {
        println!("[openvm] Successfully built the package");
        Ok(None)
    }
}
