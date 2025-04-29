use std::{
    env::var,
    fs::{create_dir_all, read, write},
    path::PathBuf,
    sync::Arc,
};

use clap::Parser;
use eyre::Result;
use openvm_build::{
    build_guest_package, find_unique_executable, get_package, GuestOptions, TargetFilter,
};
use openvm_sdk::{
    commit::{commit_app_exe, committed_exe_as_bn254},
    fs::write_exe_to_file,
    Sdk,
};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};

use crate::{
    default::{
        DEFAULT_APP_CONFIG_PATH, DEFAULT_APP_EXE_PATH, DEFAULT_COMMITTED_APP_EXE_PATH,
        DEFAULT_EXE_COMMIT_PATH, DEFAULT_MANIFEST_DIR,
    },
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

    #[arg(long, help = "Path to the target directory")]
    pub target_dir: Option<PathBuf>,

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

    #[arg(
        long,
        default_value = DEFAULT_COMMITTED_APP_EXE_PATH,
        help = "Output path for the committed program"
    )]
    pub committed_exe_output: PathBuf,

    #[arg(
        long,
        default_value = DEFAULT_EXE_COMMIT_PATH,
        help = "Output path for the exe commit (bn254 commit of committed program)"
    )]
    pub exe_commit_output: PathBuf,

    #[arg(long, default_value = "release", help = "Build profile")]
    pub profile: String,

    #[arg(long, default_value = "false", help = "use --offline in cargo build")]
    pub offline: bool,
}

#[derive(Clone, Default, clap::Args)]
#[group(required = false, multiple = false)]
pub struct BinTypeFilter {
    #[arg(long, help = "Specifies the bin target to build")]
    pub bin: Option<String>,

    #[arg(long, help = "Specifies the example target to build")]
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
    let mut guest_options = GuestOptions::default()
        .with_features(build_args.features.clone())
        .with_profile(build_args.profile.clone())
        .with_rustc_flags(var("RUSTFLAGS").unwrap_or_default().split_whitespace());
    guest_options.target_dir = build_args.target_dir.clone();
    if build_args.offline {
        guest_options.options = vec!["--offline".to_string()];
    }

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
        let exe = Sdk::new().transpile(elf, transpiler)?;
        let committed_exe = commit_app_exe(app_config.app_fri_params.fri_params, exe.clone());
        write_exe_to_file(exe, output_path)?;

        if let Some(parent) = build_args.exe_commit_output.parent() {
            create_dir_all(parent)?;
        }
        write(
            &build_args.exe_commit_output,
            committed_exe_as_bn254(&committed_exe).value.to_bytes(),
        )?;
        if let Some(parent) = build_args.committed_exe_output.parent() {
            create_dir_all(parent)?;
        }
        let committed_exe = match Arc::try_unwrap(committed_exe) {
            Ok(exe) => exe,
            Err(_) => return Err(eyre::eyre!("Failed to unwrap committed_exe Arc")),
        };
        write(
            &build_args.committed_exe_output,
            bitcode::serialize(&committed_exe)?,
        )?;

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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use eyre::Result;
    use openvm_build::RUSTC_TARGET;

    use super::*;

    #[test]
    fn test_build_with_profile() -> Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let target_dir = temp_dir.path();
        let build_args = BuildArgs {
            manifest_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("example"),
            features: vec![],
            bin_type_filter: Default::default(),
            no_transpile: true,
            config: PathBuf::from(DEFAULT_APP_CONFIG_PATH),
            exe_output: PathBuf::from(DEFAULT_APP_EXE_PATH),
            committed_exe_output: PathBuf::from(DEFAULT_COMMITTED_APP_EXE_PATH),
            exe_commit_output: PathBuf::from(DEFAULT_EXE_COMMIT_PATH),
            profile: "dev".to_string(),
            target_dir: Some(target_dir.to_path_buf()),
            offline: false,
        };
        build(&build_args)?;
        assert!(
            target_dir.join(RUSTC_TARGET).join("debug").exists(),
            "did not build with dev profile"
        );
        temp_dir.close()?;
        Ok(())
    }
}
