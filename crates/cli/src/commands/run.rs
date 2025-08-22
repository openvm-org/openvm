use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use eyre::Result;
use itertools::Itertools;
use openvm_circuit::{
    arch::{instructions::exe::VmExe, VirtualMachine, OPENVM_DEFAULT_INIT_FILE_NAME},
    system::memory::merkle::public_values::extract_public_values,
};
use openvm_sdk::{config::SdkVmCpuBuilder, fs::read_object_from_file, Sdk, F};
use openvm_stark_backend::prover::hal::MatrixDimensions;
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Engine;

use super::{build, BuildArgs, BuildCargoArgs};
use crate::{
    input::{read_to_stdin, Input},
    util::{get_manifest_path_and_dir, get_single_target_name, read_config_toml_or_default},
};

#[derive(Clone, Debug, ValueEnum)]
pub enum ExecutionMode {
    /// Pure execution (default)
    Pure,
    /// Execute with cost metering (execute_metered_cost)
    Meter,
    /// Execute with segmentation (execute_metered)
    Segment,
}

#[derive(Parser)]
#[command(name = "run", about = "Run an OpenVM program")]
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

    #[arg(
        long,
        help = "Path to the OpenVM config .toml file that specifies the VM extensions, by default will search for the file at ${manifest_dir}/openvm.toml",
        help_heading = "OpenVM Options"
    )]
    pub config: Option<PathBuf>,

    #[arg(
        long,
        help = "Output directory that OpenVM proving artifacts will be copied to",
        help_heading = "OpenVM Options"
    )]
    pub output_dir: Option<PathBuf>,

    #[arg(
        long,
        value_parser,
        help = "Input to OpenVM program",
        help_heading = "OpenVM Options"
    )]
    pub input: Option<Input>,

    #[arg(
        long,
        default_value = OPENVM_DEFAULT_INIT_FILE_NAME,
        help = "Name of the init file",
        help_heading = "OpenVM Options"
    )]
    pub init_file_name: String,

    #[arg(
        long,
        value_enum,
        default_value = "pure",
        help = "Execution mode",
        help_heading = "OpenVM Options"
    )]
    pub mode: ExecutionMode,
}

impl From<RunArgs> for BuildArgs {
    fn from(args: RunArgs) -> Self {
        BuildArgs {
            config: args.config,
            output_dir: args.output_dir,
            init_file_name: args.init_file_name,
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
            target_dir: args.target_dir,
            verbose: args.verbose,
            quiet: args.quiet,
            color: args.color,
            manifest_path: args.manifest_path,
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
        let exe_path = if let Some(exe) = &self.run_args.exe {
            exe
        } else {
            // Build and get the executable name
            let target_name = get_single_target_name(&self.cargo_args)?;
            let build_args = self.run_args.clone().into();
            let cargo_args = self.cargo_args.clone().into();
            let output_dir = build(&build_args, &cargo_args)?;
            &output_dir.join(target_name.with_extension("vmexe"))
        };

        let (_, manifest_dir) = get_manifest_path_and_dir(&self.cargo_args.manifest_path)?;
        let app_config = read_config_toml_or_default(
            self.run_args
                .config
                .to_owned()
                .unwrap_or_else(|| manifest_dir.join("openvm.toml")),
        )?;
        let exe: VmExe<F> = read_object_from_file(exe_path)?;

        let sdk = Sdk::new(app_config)?;
        let inputs = read_to_stdin(&self.run_args.input)?;

        match self.run_args.mode {
            ExecutionMode::Pure => {
                let output = sdk.execute(exe, inputs)?;
                println!("Execution output: {:?}", output);
            }
            ExecutionMode::Segment => {
                let exe = sdk.convert_to_exe(exe)?;
                let app_pk = sdk.app_pk();
                let executor_idx_to_air_idx = VirtualMachine::<
                    BabyBearPoseidon2Engine,
                    SdkVmCpuBuilder,
                >::executor_idx_to_air_idx_from_config::<
                    BabyBearPoseidon2Engine,
                    SdkVmCpuBuilder,
                >(
                    sdk.app_vm_builder(), &app_pk.app_vm_pk.vm_config
                )
                .map_err(|e| eyre::eyre!("Failed to get executor mapping: {}", e))?;

                // Extract data from the proving key to build metered context
                let (constant_trace_heights, air_names, widths, interactions): (
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                    Vec<_>,
                ) = app_pk
                    .app_vm_pk
                    .vm_pk
                    .per_air
                    .iter()
                    .map(|pk| {
                        let constant_trace_height =
                            pk.preprocessed_data.as_ref().map(|pd| pd.trace.height());
                        let air_names = pk.air_name.clone();
                        let width = pk.vk.params.width.total_width(4); // BabyBear extension degree
                        let num_interactions = pk.vk.symbolic_constraints.interactions.len();
                        (constant_trace_height, air_names, width, num_interactions)
                    })
                    .multiunzip();

                let metered_ctx = sdk.executor().build_metered_ctx(
                    &constant_trace_heights,
                    &air_names,
                    &widths,
                    &interactions,
                );
                let metered_interpreter = sdk
                    .executor()
                    .metered_instance(&exe, &executor_idx_to_air_idx)?;
                let (segments, final_state) =
                    metered_interpreter.execute_metered(inputs, metered_ctx)?;

                let output = extract_public_values(
                    sdk.executor().config.as_ref().num_public_values,
                    &final_state.memory.memory,
                );
                println!("Execution output: {:?}", output);

                let total_instructions: u64 = segments.iter().map(|s| s.num_insns).sum();
                println!("Total instructions: {}", total_instructions);
                println!("Number of segments: {}", segments.len());
            }
            ExecutionMode::Meter => {
                let exe = sdk.convert_to_exe(exe)?;
                let app_pk = sdk.app_pk();
                let executor_idx_to_air_idx = VirtualMachine::<
                    BabyBearPoseidon2Engine,
                    SdkVmCpuBuilder,
                >::executor_idx_to_air_idx_from_config::<
                    BabyBearPoseidon2Engine,
                    SdkVmCpuBuilder,
                >(
                    sdk.app_vm_builder(), &app_pk.app_vm_pk.vm_config
                )
                .map_err(|e| eyre::eyre!("Failed to get executor mapping: {}", e))?;

                // Extract widths from the proving key to build metered cost context
                let widths: Vec<_> = app_pk
                    .app_vm_pk
                    .vm_pk
                    .per_air
                    .iter()
                    .map(|pk| {
                        pk.vk.params.width.total_width(4) // BabyBear extension degree
                    })
                    .collect();

                let output = sdk.execute(exe.clone(), inputs.clone())?;
                println!("Execution output: {:?}", output);

                let cost_ctx = sdk.executor().build_metered_cost_ctx(&widths);
                let cost_interpreter = sdk
                    .executor()
                    .metered_cost_instance(&exe, &executor_idx_to_air_idx)?;
                let cost_output = cost_interpreter.execute_metered_cost(inputs, cost_ctx)?;

                println!("Total instructions: {}", cost_output.instret);
                println!("Total cost: {}", cost_output.cost);
            }
        }

        Ok(())
    }
}
