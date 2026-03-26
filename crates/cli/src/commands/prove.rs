use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::Result;
use openvm_circuit::arch::{
    execution_mode::metered::segment_ctx::{
        SegmentationConfig, SegmentationLimits, DEFAULT_MAX_MEMORY, DEFAULT_MAX_TRACE_HEIGHT_BITS,
    },
    instructions::exe::VmExe,
};
use openvm_sdk::{
    config::{AggregationSystemParams, AggregationTreeConfig, AppConfig},
    fs::{read_object_from_file, write_object_to_file, write_to_file_json},
    keygen::AppProvingKey,
    types::VersionedNonRootStarkProof,
    Sdk, F,
};
use openvm_sdk_config::SdkVmConfig;

use super::{RunArgs, RunCargoArgs};
use crate::{
    commands::build,
    input::read_to_stdin,
    util::{
        get_agg_pk_path, get_agg_vk_path, get_app_pk_path, get_manifest_path_and_dir,
        get_single_target_name, get_target_dir,
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
            help = "Path to app proving key, by default will be ${target_dir}/openvm/app.pk",
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

        #[arg(
            long,
            action,
            help = "Path to app proving key, by default will be ${target_dir}/openvm/app.pk",
            help_heading = "OpenVM Options"
        )]
        app_pk: Option<PathBuf>,

        #[arg(
            long,
            action,
            help = "Path to aggregation proving key, by default will be ${target_dir}/openvm/agg.pk",
            help_heading = "OpenVM Options"
        )]
        agg_pk: Option<PathBuf>,

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

        #[arg(
            long,
            action,
            help = "Path to app proving key, by default will be ${target_dir}/openvm/app.pk",
            help_heading = "OpenVM Options"
        )]
        app_pk: Option<PathBuf>,

        #[arg(
            long,
            action,
            help = "Path to aggregation proving key, by default will be ${target_dir}/openvm/agg.pk",
            help_heading = "OpenVM Options"
        )]
        agg_pk: Option<PathBuf>,

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
                    &PathBuf::from(target_name).with_extension("app.proof")
                };
                println!(
                    "App proof completed! Writing App proof to {}",
                    proof_path.display()
                );
                write_object_to_file(proof_path, app_proof)?;
            }
            ProveSubCommand::Stark {
                app_pk,
                agg_pk,
                proof,
                run_args,
                cargo_args,
                segmentation_args,
                agg_tree_config,
            } => {
                let mut app_pk = load_app_pk(app_pk, cargo_args)?;
                let (exe, target_name) = load_or_build_exe(run_args, cargo_args)?;
                let app_config = get_app_config(&mut app_pk, segmentation_args);
                let sdk = maybe_with_cached_keys(
                    Sdk::new(app_config, AggregationSystemParams::default())?
                        .with_agg_tree_config(*agg_tree_config)
                        .with_app_pk(app_pk),
                    agg_pk,
                    cargo_args,
                )?;
                let mut prover = sdk.prover(exe)?;
                let baseline = prover.generate_baseline();
                println!("exe commit: {:?}", baseline.app_exe_commit);

                let (stark_proof, _metadata) =
                    prover.prove(read_to_stdin(&run_args.input)?, &[])?;
                let stark_proof_bytes = VersionedNonRootStarkProof::new(stark_proof)?;

                save_keys(&sdk, cargo_args)?;

                let proof_path = if let Some(proof) = proof {
                    proof
                } else {
                    &PathBuf::from(target_name).with_extension("stark.proof")
                };
                println!(
                    "STARK proof completed! Writing STARK proof to {}",
                    proof_path.display()
                );
                write_to_file_json(proof_path, stark_proof_bytes)?;
            }
            #[cfg(feature = "evm-prove")]
            ProveSubCommand::Evm {
                app_pk,
                agg_pk,
                proof,
                run_args,
                cargo_args,
                segmentation_args,
                agg_tree_config,
            } => {
                let mut app_pk = load_app_pk(app_pk, cargo_args)?;
                let (exe, target_name) = load_or_build_exe(run_args, cargo_args)?;

                println!("Generating EVM proof, this may take a lot of compute and memory...");
                let app_config = get_app_config(&mut app_pk, segmentation_args);
                let sdk = maybe_with_cached_keys(
                    Sdk::new(app_config, AggregationSystemParams::default())?
                        .with_agg_tree_config(*agg_tree_config)
                        .with_app_pk(app_pk),
                    agg_pk,
                    cargo_args,
                )?;
                let mut prover = sdk.evm_prover(exe)?;
                let exe_commit = prover.stark_prover.app_prover.app_exe_commit();
                println!("exe commit: {:?}", exe_commit);
                let evm_proof = prover.prove_evm(read_to_stdin(&run_args.input)?, &[])?;

                save_keys(&sdk, cargo_args)?;

                let proof_path = if let Some(proof) = proof {
                    proof
                } else {
                    &PathBuf::from(target_name).with_extension("evm.proof")
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
        let (manifest_path, _) = get_manifest_path_and_dir(&cargo_args.manifest_path)?;
        let target_dir = get_target_dir(&cargo_args.target_dir, &manifest_path);
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
        &output_dir.join(target_name.with_extension("vmexe"))
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
    let (manifest_path, _) = get_manifest_path_and_dir(&cargo_args.manifest_path)?;
    Ok(get_target_dir(&cargo_args.target_dir, &manifest_path))
}

fn maybe_with_cached_keys(
    sdk: Sdk,
    agg_pk: &Option<PathBuf>,
    cargo_args: &RunCargoArgs,
) -> Result<Sdk> {
    let target_dir = target_dir_from_cargo_args(cargo_args)?;
    let mut sdk = if let Some(agg_pk) = agg_pk {
        sdk.with_agg_pk(read_object_from_file(agg_pk)?)
    } else {
        let agg_pk_path = get_agg_pk_path(&target_dir);
        if agg_pk_path.exists() {
            sdk.with_agg_pk(read_object_from_file(agg_pk_path)?)
        } else {
            sdk
        }
    };
    let root_pk_path = PathBuf::from(crate::default::default_root_pk_path());
    if root_pk_path.exists() {
        sdk = sdk.with_root_pk(read_object_from_file(&root_pk_path)?);
    }
    Ok(sdk)
}

/// Save generated keys for reuse.
fn save_keys(sdk: &Sdk, cargo_args: &RunCargoArgs) -> Result<()> {
    let target_dir = target_dir_from_cargo_args(cargo_args)?;
    let agg_pk_path = get_agg_pk_path(&target_dir);
    let agg_vk_path = get_agg_vk_path(&target_dir);
    if !agg_pk_path.exists() {
        println!(
            "Writing aggregation proving key to {}",
            agg_pk_path.display()
        );
        write_object_to_file(&agg_pk_path, sdk.agg_pk())?;
    }
    if !agg_vk_path.exists() {
        println!(
            "Writing aggregation verifying key to {}",
            agg_vk_path.display()
        );
        write_object_to_file(&agg_vk_path, sdk.agg_vk().as_ref().clone())?;
    }
    let root_pk_path = PathBuf::from(crate::default::default_root_pk_path());
    if !root_pk_path.exists() {
        println!("Writing root proving key to {}", root_pk_path.display());
        write_object_to_file(&root_pk_path, sdk.root_pk())?;
    }
    Ok(())
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
