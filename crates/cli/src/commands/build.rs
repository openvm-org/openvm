use std::{
    fs::read,
    path::{Path, PathBuf},
};

use clap::Parser;
use eyre::Result;
use openvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
use openvm_rv32im_transpiler::{Rv32ITranspilerExtension, Rv32MTranspilerExtension};
use openvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    fs::write_exe_to_file,
    Sdk,
};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler};

use crate::{default::DEFAULT_MANIFEST_DIR, util::read_to_struct_toml};

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

    #[arg(long, help = "Target name substring filter")]
    pub name: Option<String>,

    #[arg(
        long,
        default_value = "false",
        help = "Transpiles the program after building when set"
    )]
    pub transpile: bool,

    #[arg(
        long,
        help = "Path to the SDK config .toml file that specifies the transpiler extensions"
    )]
    pub transpiler_config: Option<PathBuf>,

    #[arg(
        long,
        help = "Output path for the transpiled program (default: <ELF base path>.vmexe)"
    )]
    pub transpile_to: Option<PathBuf>,

    #[arg(long, default_value = "release", help = "Build profile")]
    pub profile: String,
}

impl BuildArgs {
    pub fn exe_path(&self, elf_path: &Path) -> PathBuf {
        self.transpile_to
            .clone()
            .unwrap_or_else(|| elf_path.with_extension("vmexe"))
    }
}

#[derive(Clone, clap::Args)]
#[group(required = false, multiple = false)]
pub struct BinTypeFilter {
    #[arg(
        long,
        help = "Specifies that the target should be a binary kind when set"
    )]
    pub bin: bool,

    #[arg(
        long,
        help = "Specifies that the target should be an example kind when set"
    )]
    pub example: bool,
}

// Returns the path to the ELF file if it is unique.
pub(crate) fn build(build_args: &BuildArgs) -> Result<Option<PathBuf>> {
    println!("[openvm] Building the package...");
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
    let guest_options = GuestOptions {
        features: build_args.features.clone(),
        ..Default::default()
    };

    let pkg = get_package(&build_args.manifest_dir);
    // We support builds of libraries with 0 or >1 executables.
    let elf_path = match build_guest_package(&pkg, &guest_options, None) {
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

    if build_args.transpile {
        let elf_path = elf_path?;
        println!("[openvm] Transpiling the package...");
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
