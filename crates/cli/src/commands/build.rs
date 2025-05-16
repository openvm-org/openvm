use std::{
    env::var,
    fs::{create_dir_all, read, write},
    path::PathBuf,
    sync::Arc,
};

use clap::Parser;
use eyre::Result;
use openvm_build::{
    build_generic, get_package, get_target_dir, get_workspace_packages, get_workspace_root,
    GuestOptions,
};
use openvm_circuit::arch::{InitFileGenerator, OPENVM_DEFAULT_INIT_FILE_NAME};
use openvm_sdk::{commit::commit_app_exe, fs::write_exe_to_file, Sdk};
use openvm_transpiler::{elf::Elf, openvm_platform::memory::MEM_SIZE};

use crate::{
    default::{DEFAULT_APP_CONFIG_PATH, DEFAULT_APP_EXE_PATH, DEFAULT_COMMITTED_APP_EXE_PATH},
    util::{find_manifest_dir, read_config_toml_or_default},
};

#[derive(Parser)]
#[command(name = "build", about = "Compile an OpenVM program")]
pub struct BuildCmd {
    #[clap(flatten)]
    build_args: BuildArgs,

    #[clap(flatten)]
    cargo_args: BuildCargoArgs,
}

impl BuildCmd {
    pub fn run(&self) -> Result<()> {
        build(&self.build_args, &self.cargo_args)?;
        Ok(())
    }
}

#[derive(Clone, Parser)]
pub struct BuildArgs {
    #[arg(
        long,
        default_value = "false",
        help = "Skips transpilation into exe when set",
        help_heading = "OpenVM Options"
    )]
    pub no_transpile: bool,

    #[arg(
        long,
        default_value = DEFAULT_APP_CONFIG_PATH,
        help = "Path to the OpenVM config .toml file that specifies the VM extensions",
        help_heading = "OpenVM Options"
    )]
    pub config: PathBuf,

    #[arg(
        long,
        default_value = DEFAULT_APP_EXE_PATH,
        help = "Output path for the transpiled program",
        help_heading = "OpenVM Options"
    )]
    pub exe_output: PathBuf,

    #[arg(
        long,
        default_value = DEFAULT_COMMITTED_APP_EXE_PATH,
        help = "Output path for the committed program",
        help_heading = "OpenVM Options"
    )]
    pub committed_exe_output: PathBuf,

    #[arg(
        long,
        default_value = OPENVM_DEFAULT_INIT_FILE_NAME,
        help = "Name of the init file",
        help_heading = "OpenVM Options"
    )]
    pub init_file_name: String,
}

#[derive(Clone, Parser)]
pub struct BuildCargoArgs {
    #[arg(
        long,
        short = 'p',
        value_name = "PACKAGES",
        help = "Build only specified packages",
        help_heading = "Package Selection"
    )]
    pub package: Vec<String>,

    #[arg(
        long,
        alias = "all",
        help = "Build all members of the workspace",
        help_heading = "Package Selection"
    )]
    pub workspace: bool,

    #[arg(
        long,
        value_name = "PACKAGES",
        help = "Exclude specified packages",
        help_heading = "Package Selection"
    )]
    pub exclude: Vec<String>,

    #[arg(
        long,
        help = "Build the package library",
        help_heading = "Target Selection"
    )]
    pub lib: bool,

    #[arg(
        long,
        value_name = "BIN",
        help = "Build the specified binary",
        help_heading = "Target Selection"
    )]
    pub bin: Vec<String>,

    #[arg(
        long,
        help = "Build all binary targets",
        help_heading = "Target Selection"
    )]
    pub bins: bool,

    #[arg(
        long,
        value_name = "EXAMPLE",
        help = "Build the specified example",
        help_heading = "Target Selection"
    )]
    pub example: Vec<String>,

    #[arg(
        long,
        help = "Build all example targets",
        help_heading = "Target Selection"
    )]
    pub examples: bool,

    #[arg(
        long,
        help = "Build all package targets",
        help_heading = "Target Selection"
    )]
    pub all_targets: bool,

    #[arg(
        long,
        short = 'F',
        value_name = "FEATURES",
        value_delimiter = ',',
        help = "Space/comma separated list of features to activate",
        help_heading = "Feature Selection"
    )]
    pub features: Vec<String>,

    #[arg(
        long,
        help = "Activate all available features of all selected packages",
        help_heading = "Feature Selection"
    )]
    pub all_features: bool,

    #[arg(
        long,
        help = "Do not activate the `default` feature of the selected packages",
        help_heading = "Feature Selection"
    )]
    pub no_default_features: bool,

    #[arg(
        long,
        value_name = "NAME",
        default_value = "release",
        help = "Build with the given profile",
        help_heading = "Compilation Options"
    )]
    pub profile: String,

    #[arg(
        long,
        value_name = "DIR",
        help = "Directory for all generated artifacts and intermediate files",
        help_heading = "Output Options"
    )]
    pub target_dir: Option<PathBuf>,

    #[arg(
        long,
        short = 'v',
        help = "Use verbose output",
        help_heading = "Display Options"
    )]
    pub verbose: bool,

    #[arg(
        long,
        short = 'q',
        help = "Do not print cargo log messages",
        help_heading = "Display Options"
    )]
    pub quiet: bool,

    #[arg(
        long,
        value_name = "WHEN",
        default_value = "always",
        help = "Control when colored output is used",
        help_heading = "Display Options"
    )]
    pub color: String,

    #[arg(
        long,
        value_name = "PATH",
        help = "Path to the Cargo.toml file, by default searches for the file in the current or any parent directory",
        help_heading = "Manifest Options"
    )]
    pub manifest_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Ignore rust-version specification in packages",
        help_heading = "Manifest Options"
    )]
    pub ignore_rust_version: bool,

    #[arg(
        long,
        help = "Asserts same dependencies and versions are used as when the existing Cargo.lock file was originally generated",
        help_heading = "Manifest Options"
    )]
    pub locked: bool,

    #[arg(
        long,
        help = "Prevents Cargo from accessing the network for any reason",
        help_heading = "Manifest Options"
    )]
    pub offline: bool,

    #[arg(
        long,
        help = "Equivalent to specifying both --locked and --offline",
        help_heading = "Manifest Options"
    )]
    pub frozen: bool,
}

// Returns the paths to the ELF file for each built executable target.
pub fn build(build_args: &BuildArgs, cargo_args: &BuildCargoArgs) -> Result<Vec<PathBuf>> {
    println!("[openvm] Building the package...");

    // Find manifest directory using either manifest_path or find_manifest_dir
    let manifest_dir = if let Some(manifest_path) = &cargo_args.manifest_path {
        if !manifest_path.ends_with("Cargo.toml") {
            return Err(eyre::eyre!(
                "manifest_path must be a path to a Cargo.toml file"
            ));
        }
        manifest_path.parent().unwrap().to_path_buf()
    } else {
        find_manifest_dir(PathBuf::from("."))?
    };
    let manifest_path = manifest_dir.join("Cargo.toml");
    println!("[openvm] Manifest directory: {}", manifest_dir.display());

    // Get target path
    let target_path = if let Some(target_path) = &cargo_args.target_dir {
        target_path.to_path_buf()
    } else {
        get_target_dir(&manifest_path)
    };

    // Set guest options using build arguments; use found manifest directory for consistency
    let mut guest_options = GuestOptions::default()
        .with_features(cargo_args.features.clone())
        .with_profile(cargo_args.profile.clone())
        .with_rustc_flags(var("RUSTFLAGS").unwrap_or_default().split_whitespace());

    guest_options.target_dir = Some(target_path);
    guest_options
        .options
        .push(format!("--color={}", cargo_args.color));
    guest_options.options.push("--manifest-path".to_string());
    guest_options
        .options
        .push(manifest_path.to_string_lossy().to_string());

    for pkg in &cargo_args.package {
        guest_options.options.push("--package".to_string());
        guest_options.options.push(pkg.clone());
    }
    for pkg in &cargo_args.exclude {
        guest_options.options.push("--exclude".to_string());
        guest_options.options.push(pkg.clone());
    }
    for target in &cargo_args.bin {
        guest_options.options.push("--bin".to_string());
        guest_options.options.push(target.clone());
    }
    for example in &cargo_args.example {
        guest_options.options.push("--example".to_string());
        guest_options.options.push(example.clone());
    }

    let all_bins = cargo_args.bins || cargo_args.all_targets;
    let all_examples = cargo_args.examples || cargo_args.all_targets;

    let boolean_flags = [
        ("--workspace", cargo_args.workspace),
        ("--lib", cargo_args.lib || cargo_args.all_targets),
        ("--bins", all_bins),
        ("--examples", all_examples),
        ("--all-features", cargo_args.all_features),
        ("--no-default-features", cargo_args.no_default_features),
        ("--verbose", cargo_args.verbose),
        ("--quiet", cargo_args.quiet),
        ("--ignore-rust-version", cargo_args.ignore_rust_version),
        ("--locked", cargo_args.locked),
        ("--offline", cargo_args.offline),
        ("--frozen", cargo_args.frozen),
    ];
    for (flag, enabled) in boolean_flags {
        if enabled {
            guest_options.options.push(flag.to_string());
        }
    }

    // Build (allowing passed options to decide what gets built)
    let target_dir = match build_generic(&guest_options) {
        Ok(target_dir) => target_dir,
        Err(None) => {
            return Err(eyre::eyre!("Failed to build guest"));
        }
        Err(Some(code)) => {
            return Err(eyre::eyre!("Failed to build guest: code = {}", code));
        }
    };

    // Write to init file
    let app_config = read_config_toml_or_default(&build_args.config)?;
    app_config
        .app_vm_config
        .write_to_init_file(&manifest_dir, Some(&build_args.init_file_name))?;

    // Get all built packages
    let workspace_root = get_workspace_root(&manifest_path);
    let packages = if cargo_args.workspace || manifest_dir == workspace_root {
        get_workspace_packages(manifest_dir)
            .into_iter()
            .filter(|pkg| {
                (cargo_args.package.is_empty() || cargo_args.package.contains(&pkg.name))
                    && !cargo_args.exclude.contains(&pkg.name)
            })
            .collect()
    } else {
        vec![get_package(manifest_dir)]
    };

    // Find elf paths of all targets for all built packages
    let elf_paths: Vec<PathBuf> = packages
        .iter()
        .flat_map(|pkg| {
            pkg.targets
                .iter()
                .filter(move |target| {
                    // We only build bin and example targets (note they are mutually exclusive
                    // types). If no target selection flags are set, then all bin targets are
                    // built by default.
                    if target.is_example() {
                        return all_examples || cargo_args.example.contains(&target.name);
                    } else if target.is_bin() {
                        return all_bins
                            || cargo_args.bin.contains(&target.name)
                            || (!cargo_args.examples
                                && !cargo_args.lib
                                && cargo_args.bin.is_empty()
                                && cargo_args.example.is_empty());
                    }
                    false
                })
                .map(|target| {
                    if target.is_example() {
                        target_dir.join("examples")
                    } else {
                        target_dir.clone()
                    }
                    .join(&target.name)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Transpile and commit, storing in target_dir/openvm/${profile} by default
    // TODO[stephen]: actually implement target_dir change
    if !build_args.no_transpile {
        for elf_path in &elf_paths {
            println!("[openvm] Transpiling the package...");
            let output_path = &build_args.exe_output;
            let transpiler = app_config.app_vm_config.transpiler();

            let data = read(elf_path.clone())?;
            let elf = Elf::decode(&data, MEM_SIZE as u32)?;
            let exe = Sdk::new().transpile(elf, transpiler)?;
            let committed_exe = commit_app_exe(app_config.app_fri_params.fri_params, exe.clone());
            write_exe_to_file(exe, output_path)?;

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
        }
    }

    // Return elf paths of all targets for all built packages
    println!("[openvm] Successfully built the packages");
    Ok(elf_paths)
}
