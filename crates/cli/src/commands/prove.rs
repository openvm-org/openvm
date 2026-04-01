use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::{eyre, Result};
use openvm_circuit::arch::{
    execution_mode::metered::segment_ctx::{
        SegmentationConfig, SegmentationLimits, DEFAULT_MAX_MEMORY, DEFAULT_MAX_TRACE_HEIGHT_BITS,
    },
    instructions::exe::VmExe,
};
use openvm_continuations::CommitBytes;
use openvm_sdk::{
    config::{AggregationSystemParams, AggregationTreeConfig, AppConfig},
    fs::{read_object_from_file, write_object_to_file, write_to_file_json},
    keygen::AppProvingKey,
    types::{AppExecutionCommit, VerificationBaselineJson, VersionedNonRootStarkProof},
    Sdk, F,
};
use openvm_sdk_config::SdkVmConfig;
use p3_bn254::Bn254;

use super::{RunArgs, RunCargoArgs};
use crate::{
    args::ProvingKeyArgs,
    commands::build,
    default::{APP_PROOF_EXT, EVM_PROOF_EXT, STARK_PROOF_EXT, VMEXE_EXT},
    input::read_to_stdin,
    util::{
        get_agg_pk_path, get_app_baseline_path, get_app_pk_path, get_manifest_path_and_dir,
        get_single_target_name, get_target_dir, get_target_output_dir,
    },
};

#[derive(Parser)]
#[command(name = "prove", about = "Generate a program proof")]
pub struct ProveCmd {
    #[command(subcommand)]
    command: ProveSubCommand,
}

#[derive(Parser)]
enum ProveSubCommand {
    App {
        #[arg(
            long,
            action,
            help = "Path to app proof output, by default will be ./${bin_name}.app.proof",
            help_heading = "Output"
        )]
        proof: Option<PathBuf>,

        #[arg(
            long,
            action,
            help = "Path to app proving key, by default will be ${openvm_dir}/app.pk",
            help_heading = "OpenVM Options"
        )]
        app_pk: Option<PathBuf>,

        #[command(flatten)]
        run_args: RunArgs,

        #[command(flatten)]
        cargo_args: RunCargoArgs,

        #[command(flatten)]
        segmentation_args: SegmentationArgs,
    },
    Stark {
        #[arg(
            long,
            action,
            help = "Path to STARK proof output, by default will be ./${bin_name}.stark.proof",
            help_heading = "Output"
        )]
        proof: Option<PathBuf>,

        #[command(flatten)]
        keys: ProvingKeyArgs,

        #[command(flatten)]
        run_args: RunArgs,

        #[command(flatten)]
        cargo_args: RunCargoArgs,

        #[command(flatten)]
        segmentation_args: SegmentationArgs,

        #[command(flatten)]
        agg_tree_config: AggregationTreeConfig,
    },
    #[cfg(feature = "evm-prove")]
    Evm {
        #[arg(
            long,
            action,
            help = "Path to EVM proof output, by default will be ./${bin_name}.evm.proof",
            help_heading = "Output"
        )]
        proof: Option<PathBuf>,

        #[command(flatten)]
        keys: ProvingKeyArgs,

        #[command(flatten)]
        run_args: RunArgs,

        #[command(flatten)]
        cargo_args: RunCargoArgs,

        #[command(flatten)]
        segmentation_args: SegmentationArgs,

        #[command(flatten)]
        agg_tree_config: AggregationTreeConfig,
    },
}

#[derive(Clone, Copy, Parser)]
pub struct SegmentationArgs {
    /// Trace height threshold, in bits, across all chips for triggering segmentation for
    /// continuations in the app proof. These thresholds are not exceeded except when they are too
    /// small.
    #[arg(
        long,
        default_value_t = DEFAULT_MAX_TRACE_HEIGHT_BITS,
        help_heading = "OpenVM Options"
    )]
    pub segment_max_height_bits: u8,
    /// Total memory in bytes used across all chips for triggering segmentation for continuations
    /// in the app proof. These thresholds are not exceeded except when they are too small.
    #[arg(
        long,
        default_value_t = DEFAULT_MAX_MEMORY,
        help_heading = "OpenVM Options"
    )]
    pub segment_max_memory: usize,
}

impl ProveCmd {
    pub fn run(&self) -> Result<()> {
        match &self.command {
            ProveSubCommand::App {
                app_pk,
                proof,
                run_args,
                cargo_args,
                segmentation_args,
            } => {
                let mut app_pk = load_app_pk(app_pk, cargo_args)?;
                let app_config = get_app_config(&mut app_pk, segmentation_args);
                let sdk =
                    Sdk::new(app_config, AggregationSystemParams::default())?.with_app_pk(app_pk);
                let (exe, target_name) = load_or_build_exe(run_args, cargo_args)?;

                let app_proof = sdk
                    .app_prover(exe)?
                    .prove(read_to_stdin(&run_args.input)?)?;

                let proof_path = if let Some(proof) = proof {
                    proof
                } else {
                    &PathBuf::from(target_name).with_extension(APP_PROOF_EXT)
                };
                println!(
                    "App proof completed! Writing App proof to {}",
                    proof_path.display()
                );
                write_object_to_file(proof_path, app_proof)?;
            }
            ProveSubCommand::Stark {
                keys,
                proof,
                run_args,
                cargo_args,
                segmentation_args,
                agg_tree_config,
            } => {
                let mut app_pk = load_app_pk(&keys.app_pk, cargo_args)?;
                let (exe, target_name) = load_or_build_exe(run_args, cargo_args)?;
                let app_config = get_app_config(&mut app_pk, segmentation_args);
                let sdk = with_required_agg_pk(
                    Sdk::new(app_config, AggregationSystemParams::default())?
                        .with_agg_tree_config(*agg_tree_config)
                        .with_app_pk(app_pk),
                    &keys.agg_pk,
                    cargo_args,
                )?;
                let mut prover = sdk.prover(exe)?;
                let baseline = prover.generate_baseline();
                let app_vk_commit = prover.app_vk_commit();

                let app_commit = AppExecutionCommit {
                    app_exe_commit: CommitBytes::from(baseline.app_exe_commit),
                    app_vk_commit: CommitBytes::from(app_vk_commit),
                };
                let exe_commit_bn254 = Bn254::from(app_commit.app_exe_commit);
                let vk_commit_bn254 = Bn254::from(app_commit.app_vk_commit);
                println!("exe commit: {:?}", exe_commit_bn254);
                println!("vk commit: {:?}", vk_commit_bn254);

                let (stark_proof, _metadata) =
                    prover.prove(read_to_stdin(&run_args.input)?, &[])?;
                let stark_proof_bytes = VersionedNonRootStarkProof::new(stark_proof)?;

                let target_dir = target_dir_from_cargo_args(cargo_args)?;
                let target_output_dir = get_target_output_dir(&target_dir, &cargo_args.profile);
                let target_name_path =
                    get_single_target_name(cargo_args).unwrap_or(PathBuf::from(&target_name));
                let baseline_path = get_app_baseline_path(&target_output_dir, target_name_path);
                println!("Writing baseline to {}", baseline_path.display());
                let baseline_json: VerificationBaselineJson = baseline.into();
                write_to_file_json(&baseline_path, &baseline_json)?;

                let proof_path = if let Some(proof) = proof {
                    proof
                } else {
                    &PathBuf::from(target_name).with_extension(STARK_PROOF_EXT)
                };
                println!(
                    "STARK proof completed! Writing STARK proof to {}",
                    proof_path.display()
                );
                write_to_file_json(proof_path, stark_proof_bytes)?;
            }
            #[cfg(feature = "evm-prove")]
            ProveSubCommand::Evm {
                keys,
                proof,
                run_args,
                cargo_args,
                segmentation_args,
                agg_tree_config,
            } => {
                let mut app_pk = load_app_pk(&keys.app_pk, cargo_args)?;
                let (exe, target_name) = load_or_build_exe(run_args, cargo_args)?;

                println!("Generating EVM proof, this may take a lot of compute and memory...");
                let app_config = get_app_config(&mut app_pk, segmentation_args);
                let sdk = with_required_agg_pk(
                    Sdk::new(app_config, AggregationSystemParams::default())?
                        .with_agg_tree_config(*agg_tree_config)
                        .with_app_pk(app_pk),
                    &keys.agg_pk,
                    cargo_args,
                )?;
                let sdk = with_required_root_pk(sdk)?;
                let mut prover = sdk.evm_prover(exe)?;
                let exe_commit = prover.stark_prover.app_prover.app_exe_commit();
                println!("exe commit: {:?}", exe_commit);
                let evm_proof = prover.prove_evm(read_to_stdin(&run_args.input)?, &[])?;

                let proof_path = if let Some(proof) = proof {
                    proof
                } else {
                    &PathBuf::from(target_name).with_extension(EVM_PROOF_EXT)
                };
                println!(
                    "EVM proof completed! Writing EVM proof to {}",
                    proof_path.display()
                );
                write_to_file_json(proof_path, evm_proof)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn load_app_pk(
    app_pk: &Option<PathBuf>,
    cargo_args: &RunCargoArgs,
) -> Result<AppProvingKey<SdkVmConfig>> {
    let app_pk_path = if let Some(app_pk) = app_pk {
        app_pk.to_path_buf()
    } else {
        let (manifest_path, _) = get_manifest_path_and_dir(&cargo_args.manifest.manifest_path)?;
        let target_dir = get_target_dir(&cargo_args.manifest.target_dir, &manifest_path);
        get_app_pk_path(&target_dir)
    };

    read_object_from_file(app_pk_path)
}

/// Returns `(exe, target_name.file_stem())` where target_name has no extension and only contains
/// the file stem (in particular it does not include `examples/` if the target was an example)
pub(crate) fn load_or_build_exe(
    run_args: &RunArgs,
    cargo_args: &RunCargoArgs,
) -> Result<(VmExe<F>, String)> {
    let exe_path = if let Some(exe) = &run_args.exe {
        exe
    } else {
        // Build and get the executable name
        let target_name = get_single_target_name(cargo_args)?;
        let build_args = run_args.clone().into();
        let cargo_args = cargo_args.clone().into();
        let output_dir = build(&build_args, &cargo_args)?;
        &output_dir.join(target_name.with_extension(VMEXE_EXT))
    };

    let app_exe = read_object_from_file(exe_path)?;
    Ok((
        app_exe,
        exe_path.file_stem().unwrap().to_string_lossy().into_owned(),
    ))
}

/// Should only be called when `app_pk` has only a single reference internally.
/// Mutates the `SystemConfig` within `app_pk` and then returns the updated `AppConfig`.
fn get_app_config(
    app_pk: &mut AppProvingKey<SdkVmConfig>,
    segmentation_args: &SegmentationArgs,
) -> AppConfig<SdkVmConfig> {
    Arc::get_mut(&mut app_pk.app_vm_pk)
        .unwrap()
        .vm_config
        .system
        .config
        .set_segmentation_config((*segmentation_args).into());
    app_pk.app_config()
}

fn target_dir_from_cargo_args(cargo_args: &RunCargoArgs) -> Result<PathBuf> {
    let (manifest_path, _) = get_manifest_path_and_dir(&cargo_args.manifest.manifest_path)?;
    Ok(get_target_dir(
        &cargo_args.manifest.target_dir,
        &manifest_path,
    ))
}

fn resolve_agg_pk_path(agg_pk: &Option<PathBuf>, cargo_args: &RunCargoArgs) -> Result<PathBuf> {
    if let Some(agg_pk) = agg_pk {
        Ok(agg_pk.to_path_buf())
    } else {
        let target_dir = target_dir_from_cargo_args(cargo_args)?;
        Ok(get_agg_pk_path(&target_dir))
    }
}

fn with_required_agg_pk(
    sdk: Sdk,
    agg_pk: &Option<PathBuf>,
    cargo_args: &RunCargoArgs,
) -> Result<Sdk> {
    let agg_pk_path = resolve_agg_pk_path(agg_pk, cargo_args)?;
    let agg_pk = read_object_from_file(&agg_pk_path).map_err(|e| {
        eyre!(
            "Failed to read aggregation proving key from {}: {e}\nRun 'cargo openvm keygen' first to generate it",
            agg_pk_path.display()
        )
    })?;
    Ok(sdk.with_agg_pk(agg_pk))
}

#[cfg(feature = "evm-prove")]
fn with_required_root_pk(sdk: Sdk) -> Result<Sdk> {
    let root_pk_path = PathBuf::from(crate::default::default_root_pk_path());
    let root_pk = read_object_from_file(&root_pk_path).map_err(|e| {
        eyre!(
            "Failed to read root proving key from {}: {e}\nRun 'cargo openvm setup' first to generate it",
            root_pk_path.display()
        )
    })?;
    Ok(sdk.with_root_pk(root_pk))
}

impl From<SegmentationArgs> for SegmentationConfig {
    fn from(args: SegmentationArgs) -> Self {
        SegmentationConfig {
            limits: SegmentationLimits::default()
                .with_max_trace_height(
                    1u32.checked_shl(args.segment_max_height_bits as u32)
                        .expect("segment_max_height_bits too large"),
                )
                .with_max_memory(args.segment_max_memory),
            ..Default::default()
        }
    }
}
