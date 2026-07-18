#[cfg(feature = "rvr")]
use std::path::Path;
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use eyre::{eyre, Result};
use openvm_circuit::arch::instructions::exe::VmExe;
use openvm_sdk::{
    config::AggregationSystemParams, fs::read_object_from_file, keygen::AppProvingKey, Sdk, F,
};
use openvm_sdk_config::SdkVmConfig;

use super::{build, BuildArgs, BuildCargoArgs};
use crate::{
    args::{ManifestArgs, OpenVmConfigArgs},
    default::{OPENVM_CONFIG_FILENAME, VMEXE_EXT},
    input::{read_to_stdin, Input},
    util::{
        get_app_pk_path, get_manifest_path_and_dir, get_single_target_name, get_target_dir,
        read_config_toml_or_default,
    },
};

#[cfg(feature = "rvr")]
const DEFAULT_EXECUTION_PROFILE_HZ: u32 = 1_000;

#[derive(Clone, Debug, ValueEnum)]
pub enum ExecutionMode {
    /// Runs the program normally
    Pure,
    /// Runs the program and estimates the execution cost in terms of number of cells
    Meter,
    /// Runs the program and calculates the number of segments that the execution will be split
    /// into for proving
    Segment,
}

#[derive(Parser)]
#[command(name = "execute", about = "Run an OpenVM program")]
pub struct RunCmd {
    #[clap(flatten)]
    run_args: RunArgs,

    #[clap(flatten)]
    cargo_args: RunCargoArgs,
}

#[derive(Clone, Parser)]
pub struct RunArgs {
    #[arg(
        long,
        action,
        help = "Path to OpenVM executable, if specified build will be skipped",
        help_heading = "OpenVM Options"
    )]
    pub exe: Option<PathBuf>,

    #[clap(flatten)]
    pub openvm_config: OpenVmConfigArgs,

    #[arg(
        long,
        value_parser,
        help = "Input to OpenVM program",
        help_heading = "OpenVM Options"
    )]
    pub input: Option<Input>,

    #[arg(
        long,
        value_enum,
        default_value = "pure",
        help = "Execution mode",
        help_heading = "OpenVM Options"
    )]
    pub mode: ExecutionMode,

    #[arg(
        long,
        value_name = "PATH",
        num_args = 0..=1,
        require_equals = true,
        help = "Profile RVR guest execution and print a Firefox Profiler link; optionally save the .json.gz to PATH",
        help_heading = "Execution Profiling"
    )]
    pub execution_profile: Option<Option<PathBuf>>,

    #[arg(
        long,
        short = 'r',
        value_name = "HZ",
        value_parser = clap::value_parser!(u32).range(1..=1_000_000),
        requires = "execution_profile",
        help = "Sampling rate, in Hz (default: 1000)",
        help_heading = "Execution Profiling"
    )]
    pub rate: Option<u32>,
}

impl From<RunArgs> for BuildArgs {
    fn from(args: RunArgs) -> Self {
        BuildArgs {
            openvm_config: args.openvm_config,
            ..Default::default()
        }
    }
}

#[derive(Clone, Parser)]
pub struct RunCargoArgs {
    #[arg(
        long,
        short = 'p',
        value_name = "PACKAGES",
        help = "The package to run; by default is the package in the current workspace",
        help_heading = "Package Selection"
    )]
    pub package: Option<String>,

    #[arg(
        long,
        value_name = "BIN",
        help = "Run the specified binary",
        help_heading = "Target Selection"
    )]
    pub bin: Vec<String>,

    #[arg(
        long,
        value_name = "EXAMPLE",
        help = "Run the specified example",
        help_heading = "Target Selection"
    )]
    pub example: Vec<String>,

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
        help = "Run with the given profile",
        help_heading = "Compilation Options"
    )]
    pub profile: String,

    #[clap(flatten)]
    pub manifest: ManifestArgs,

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

impl From<RunCargoArgs> for BuildCargoArgs {
    fn from(args: RunCargoArgs) -> Self {
        BuildCargoArgs {
            package: args.package.into_iter().collect(),
            bin: args.bin,
            example: args.example,
            features: args.features,
            all_features: args.all_features,
            no_default_features: args.no_default_features,
            profile: args.profile,
            manifest: args.manifest,
            verbose: args.verbose,
            quiet: args.quiet,
            color: args.color,
            ignore_rust_version: args.ignore_rust_version,
            locked: args.locked,
            offline: args.offline,
            frozen: args.frozen,
            ..Default::default()
        }
    }
}

impl RunCmd {
    pub fn run(&self) -> Result<()> {
        let profile_enabled =
            self.run_args.execution_profile.is_some() || self.run_args.rate.is_some();
        if profile_enabled && self.run_args.exe.is_some() {
            return Err(eyre!(
                "--execution-profile requires building the guest so frame pointers and debug info can be enabled; --exe is not supported"
            ));
        }
        if profile_enabled && !cfg!(feature = "rvr") {
            return Err(eyre!(
                "--execution-profile requires cargo-openvm to be built with the `rvr` feature"
            ));
        }
        if profile_enabled && !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            return Err(eyre!("--execution-profile currently requires Linux x86_64"));
        }

        #[cfg(feature = "rvr")]
        let mut guest_elf_path = None;
        let exe_path = if let Some(exe) = &self.run_args.exe {
            exe.clone()
        } else {
            // Build and get the executable name
            let target_name = get_single_target_name(&self.cargo_args)?;
            let build_args: BuildArgs = self.run_args.clone().into();
            #[cfg(feature = "rvr")]
            let mut build_args = build_args;
            #[cfg(feature = "rvr")]
            if profile_enabled {
                build_args
                    .rustc_flags
                    .extend(["-Cforce-frame-pointers=yes".to_string()]);
                let cargo_profile = self
                    .cargo_args
                    .profile
                    .to_ascii_uppercase()
                    .replace('-', "_");
                build_args.cargo_env.extend([
                    (
                        format!("CARGO_PROFILE_{cargo_profile}_DEBUG"),
                        "2".to_string(),
                    ),
                    (
                        format!("CARGO_PROFILE_{cargo_profile}_STRIP"),
                        "none".to_string(),
                    ),
                ]);
                build_args.quiet_status = true;
            }
            let cargo_args = self.cargo_args.clone().into();
            let output_dir = build(&build_args, &cargo_args)?;
            #[cfg(feature = "rvr")]
            if profile_enabled {
                let (manifest_path, _) =
                    get_manifest_path_and_dir(&self.cargo_args.manifest.manifest_path)?;
                let target_dir =
                    get_target_dir(&self.cargo_args.manifest.target_dir, &manifest_path);
                let is_example = target_name.parent() == Some(Path::new("examples"));
                let file_name = target_name
                    .file_name()
                    .ok_or_else(|| eyre!("invalid guest target name"))?;
                guest_elf_path = Some(
                    openvm_build::get_dir_with_profile(
                        target_dir,
                        &self.cargo_args.profile,
                        is_example,
                    )
                    .join(file_name),
                );
            }
            output_dir.join(target_name.with_extension(VMEXE_EXT))
        };

        let (manifest_path, manifest_dir) =
            get_manifest_path_and_dir(&self.cargo_args.manifest.manifest_path)?;
        let exe: VmExe<F> = read_object_from_file(&exe_path)?;
        let inputs = read_to_stdin(&self.run_args.input)?;

        let sdk = if matches!(
            self.run_args.mode,
            ExecutionMode::Segment | ExecutionMode::Meter
        ) {
            let target_dir = get_target_dir(&self.cargo_args.manifest.target_dir, &manifest_path);
            let app_pk_path = get_app_pk_path(&target_dir);

            let app_pk: AppProvingKey<SdkVmConfig> =
                read_object_from_file(&app_pk_path).map_err(|e| {
                    eyre!(
                        "Failed to read app proving key from {}: {e}\nRun 'cargo openvm keygen --app-only' first to generate it",
                        app_pk_path.display()
                    )
                })?;
            Sdk::builder()
                .app_pk(app_pk)
                .agg_params(AggregationSystemParams::default())
                .build()?
        } else {
            let config_path = self
                .run_args
                .openvm_config
                .config
                .to_owned()
                .unwrap_or_else(|| manifest_dir.join(OPENVM_CONFIG_FILENAME));
            let app_config = read_config_toml_or_default(&config_path)?;
            Sdk::new(app_config, AggregationSystemParams::default())?
        };

        if profile_enabled {
            #[cfg(feature = "rvr")]
            {
                let guest_elf_path = guest_elf_path
                    .as_deref()
                    .ok_or_else(|| eyre!("guest ELF path is unavailable"))?;
                let sample_hz = self.run_args.rate.unwrap_or(DEFAULT_EXECUTION_PROFILE_HZ);
                let profile = match self.run_args.mode {
                    ExecutionMode::Pure => {
                        let (output, profile) = openvm_sdk::execution_profile::profile_execution(
                            guest_elf_path,
                            sample_hz,
                            || sdk.compile_and_execute(exe, inputs),
                        )?;
                        eprintln!("[openvm] Execution output: {output:?}");
                        profile
                    }
                    ExecutionMode::Meter => {
                        let ((output, (cost, instret)), profile) =
                            openvm_sdk::execution_profile::profile_execution(
                                guest_elf_path,
                                sample_hz,
                                || sdk.compile_and_execute_metered_cost(exe, inputs),
                            )?;
                        eprintln!("[openvm] Execution output: {output:?}");
                        eprintln!("[openvm] Number of instructions executed: {instret}");
                        eprintln!("[openvm] Total cost: {cost}");
                        profile
                    }
                    ExecutionMode::Segment => {
                        let ((output, segments), profile) =
                            openvm_sdk::execution_profile::profile_execution(
                                guest_elf_path,
                                sample_hz,
                                || sdk.compile_and_execute_metered(exe, inputs),
                            )?;
                        let total_instructions: u64 =
                            segments.iter().map(|segment| segment.num_insns).sum();
                        eprintln!("[openvm] Execution output: {output:?}");
                        eprintln!("[openvm] Number of instructions executed: {total_instructions}");
                        eprintln!("[openvm] Total segments: {}", segments.len());
                        profile
                    }
                };
                if let Some(output_path) = self
                    .run_args
                    .execution_profile
                    .as_ref()
                    .and_then(Option::as_deref)
                {
                    profile.save(output_path)?;
                    eprintln!(
                        "[openvm] Saved execution profile to {}",
                        output_path.display()
                    );
                }
                let url = profile.upload()?;
                println!("{url}");
                return Ok(());
            }
            #[cfg(not(feature = "rvr"))]
            unreachable!("profile feature check above must reject this configuration");
        }

        match self.run_args.mode {
            ExecutionMode::Pure => {
                let output = sdk.compile_and_execute(exe, inputs)?;
                println!("Execution output: {output:?}");
            }
            ExecutionMode::Meter => {
                let (output, (cost, instret)) =
                    sdk.compile_and_execute_metered_cost(exe, inputs)?;
                println!("Execution output: {output:?}");

                println!("Number of instructions executed: {instret}");
                println!("Total cost: {cost}");
            }
            ExecutionMode::Segment => {
                let (output, segments) = sdk.compile_and_execute_metered(exe, inputs)?;
                println!("Execution output: {output:?}");

                let total_instructions: u64 = segments.iter().map(|s| s.num_insns).sum();
                println!("Number of instructions executed: {total_instructions}");
                println!("Total segments: {}", segments.len());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use clap::{error::ErrorKind, CommandFactory, Parser};

    use super::RunCmd;

    #[test]
    fn execution_profile_help_has_only_the_compact_interface() {
        let help = RunCmd::command().render_long_help().to_string();
        let profiling_help = help
            .split_once("Execution Profiling:")
            .expect("execution profiling help section")
            .1
            .split_once("Package Selection:")
            .expect("package selection follows execution profiling")
            .0;

        assert!(profiling_help.contains("--execution-profile[=<PATH>]"));
        assert!(profiling_help.contains("-r, --rate <HZ>"));
        assert!(!help.contains("--profile-execution"));
        assert!(!help.contains("--profile-output"));
        assert!(!help.contains("--profile-hz"));
    }

    #[test]
    fn rate_requires_execution_profile() {
        let error = match RunCmd::try_parse_from(["execute", "--rate", "2000"]) {
            Ok(_) => panic!("--rate should require --execution-profile"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
        assert!(error.to_string().contains("--execution-profile"));
    }
}
