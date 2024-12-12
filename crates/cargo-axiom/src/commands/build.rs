use std::{
    fs::read,
    path::{Path, PathBuf},
};

use axvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
use axvm_rv32im_transpiler::{Rv32ITranspilerExtension, Rv32MTranspilerExtension};
use axvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    fs::write_exe_to_file,
    Sdk,
};
use axvm_transpiler::{axvm_platform::memory::MEM_SIZE, elf::Elf, transpiler::Transpiler};
use clap::Parser;
use eyre::Result;

use crate::util::read_to_struct_toml;

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

#[derive(Clone, Parser)]
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

    /// Transpile the program after building
    #[arg(long, default_value = "false")]
    pub transpile: bool,

    /// Path to the SDK config .toml file that specifies the transpiler extensions
    #[arg(long)]
    pub transpiler_config: Option<PathBuf>,

    /// Output path for the transpiled program (default: <ELF base path>.axvmexe)
    #[arg(long)]
    pub transpile_to: Option<PathBuf>,

    /// Build profile
    #[arg(long, default_value = "release")]
    pub profile: String,
}

impl BuildArgs {
    pub fn exe_path(&self, elf_path: &Path) -> PathBuf {
        self.transpile_to
            .clone()
            .unwrap_or_else(|| elf_path.with_extension("axvmexe"))
    }
}

#[derive(Clone, clap::Args)]
#[group(required = false, multiple = false)]
pub struct BinTypeFilter {
    /// Specify that the target should be a binary kind
    #[arg(long)]
    pub bin: bool,

    /// Specify that the target should be an example kind
    #[arg(long)]
    pub example: bool,
}

// Returns the path to the ELF file if it is unique.
pub(crate) fn build(build_args: &BuildArgs) -> Result<Option<PathBuf>> {
    println!("[axiom] Building the package...");
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
    let pkg_dir = build_args
        .manifest_dir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap());
    let guest_options = GuestOptions {
        features: build_args.features.clone(),
        ..Default::default()
    };

    let pkg = get_package(&pkg_dir);
    // We support builds of libraries with 0 or >1 executables.
    let elf_path = match build_guest_package(&pkg, &guest_options, None) {
        Ok(target_dir) => find_unique_executable(&pkg_dir, &target_dir, &target_filter),
        Err(None) => {
            return Err(eyre::eyre!("Failed to build guest"));
        }
        Err(Some(code)) => {
            return Err(eyre::eyre!("Failed to build guest: code = {}", code));
        }
    };

    if build_args.transpile {
        let elf_path = elf_path?;
        println!("[axiom] Transpiling the package...");
        let output_path = build_args.exe_path(&elf_path);
        let transpiler = if let Some(transpiler_config) = build_args.transpiler_config.clone() {
            let app_config: AppConfig<SdkVmConfig> = read_to_struct_toml(&transpiler_config)?;
            app_config.app_vm_config.transpiler()
        } else {
            Transpiler::default()
                .with_extension(Rv32ITranspilerExtension)
                .with_extension(Rv32MTranspilerExtension)
        };

        let data = read(elf_path.clone())?;
        let elf = Elf::decode(&data, MEM_SIZE as u32)?;
        let exe = Sdk.transpile(elf, transpiler)?;
        write_exe_to_file(exe, &output_path)?;

        println!(
            "[axiom] Successfully transpiled to {}",
            output_path.display()
        );
        Ok(Some(elf_path))
    } else if let Ok(elf_path) = elf_path {
        println!(
            "[axiom] Successfully built the package: {}",
            elf_path.display()
        );
        Ok(Some(elf_path))
    } else {
        println!("[axiom] Successfully built the package");
        Ok(None)
    }
}
