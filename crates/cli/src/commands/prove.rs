use std::{path::PathBuf, sync::Arc};

use clap::Parser;
use eyre::Result;
#[cfg(feature = "evm-prove")]
use openvm_sdk::fs::write_evm_proof_to_file;
use openvm_sdk::{
    commit::AppExecutionCommit,
    config::{AggregationTreeConfig, SdkVmConfig},
    fs::{
        read_agg_stark_pk_from_file, read_app_pk_from_file, read_exe_from_file,
        write_app_proof_to_file, write_to_file_json,
    },
    keygen::AppProvingKey,
    types::VmStarkProofBytes,
    NonRootCommittedExe, Sdk,
};

use super::{RunArgs, RunCargoArgs};
use crate::{
    commands::build,
    default::default_agg_stark_pk_path,
    input::read_to_stdin,
    util::{get_app_pk_path, get_manifest_path_and_dir, get_single_target_name, get_target_dir},
};
#[cfg(feature = "evm-prove")]
use crate::{default::default_params_dir, util::read_default_agg_pk};

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

        #[command(flatten)]
        run_args: RunArgs,

        #[command(flatten)]
        cargo_args: RunCargoArgs,

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

        #[command(flatten)]
        run_args: RunArgs,

        #[command(flatten)]
        cargo_args: RunCargoArgs,

        #[command(flatten)]
        agg_tree_config: AggregationTreeConfig,
    },
}

impl ProveCmd {
    pub fn run(&self) -> Result<()> {
        match &self.command {
            ProveSubCommand::App {
                app_pk,
                proof,
                run_args,
                cargo_args,
            } => {
                let sdk = Sdk::new();
                let app_pk = load_app_pk(app_pk, cargo_args)?;
                let (committed_exe, target_name) =
                    load_or_build_and_commit_exe(&sdk, run_args, cargo_args, &app_pk)?;

                let app_proof =
                    sdk.generate_app_proof(app_pk, committed_exe, read_to_stdin(&run_args.input)?)?;

                let proof_path = if let Some(proof) = proof {
                    proof
                } else {
                    &PathBuf::from(format!("{}.app.proof", target_name))
                };
                write_app_proof_to_file(app_proof, proof_path)?;
            }
            ProveSubCommand::Stark {
                app_pk,
                proof,
                run_args,
                cargo_args,
                agg_tree_config,
            } => {
                let sdk = Sdk::new().with_agg_tree_config(*agg_tree_config);
                let app_pk = load_app_pk(app_pk, cargo_args)?;
                let (committed_exe, target_name) =
                    load_or_build_and_commit_exe(&sdk, run_args, cargo_args, &app_pk)?;

                let commits = AppExecutionCommit::compute(
                    &app_pk.app_vm_pk.vm_config,
                    &committed_exe,
                    &app_pk.leaf_committed_exe,
                );
                println!("exe commit: {:?}", commits.app_exe_commit.to_bn254());
                println!("vm commit: {:?}", commits.app_vm_commit.to_bn254());

                let agg_stark_pk = read_agg_stark_pk_from_file(default_agg_stark_pk_path()).map_err(|e| {
                    eyre::eyre!("Failed to read aggregation proving key: {}\nPlease run 'cargo openvm setup' first", e)
                })?;
                let stark_proof = sdk.generate_e2e_stark_proof(
                    app_pk,
                    committed_exe,
                    agg_stark_pk,
                    read_to_stdin(&run_args.input)?,
                )?;

                let stark_proof_bytes = VmStarkProofBytes::new(commits, stark_proof)?;

                let proof_path = if let Some(proof) = proof {
                    proof
                } else {
                    &PathBuf::from(format!("{}.stark.proof", target_name))
                };
                write_to_file_json(proof_path, stark_proof_bytes)?;
            }
            #[cfg(feature = "evm-prove")]
            ProveSubCommand::Evm {
                app_pk,
                proof,
                run_args,
                cargo_args,
                agg_tree_config,
            } => {
                use openvm_native_recursion::halo2::utils::CacheHalo2ParamsReader;

                let sdk = Sdk::new().with_agg_tree_config(*agg_tree_config);
                let app_pk = load_app_pk(app_pk, cargo_args)?;
                let (committed_exe, target_name) =
                    load_or_build_and_commit_exe(&sdk, run_args, cargo_args, &app_pk)?;

                let commits = AppExecutionCommit::compute(
                    &app_pk.app_vm_pk.vm_config,
                    &committed_exe,
                    &app_pk.leaf_committed_exe,
                );
                println!("exe commit: {:?}", commits.app_exe_commit.to_bn254());
                println!("vm commit: {:?}", commits.app_vm_commit.to_bn254());

                println!("Generating EVM proof, this may take a lot of compute and memory...");
                let agg_pk = read_default_agg_pk().map_err(|e| {
                    eyre::eyre!("Failed to read aggregation proving key: {}\nPlease run 'cargo openvm setup' first", e)
                })?;
                let params_reader = CacheHalo2ParamsReader::new(default_params_dir());
                let evm_proof = sdk.generate_evm_proof(
                    &params_reader,
                    app_pk,
                    committed_exe,
                    agg_pk,
                    read_to_stdin(&run_args.input)?,
                )?;

                let proof_path = if let Some(proof) = proof {
                    proof
                } else {
                    &PathBuf::from(format!("{}.evm.proof", target_name))
                };
                write_evm_proof_to_file(evm_proof, proof_path)?;
            }
        }
        Ok(())
    }
}

pub(crate) fn load_app_pk(
    app_pk: &Option<PathBuf>,
    cargo_args: &RunCargoArgs,
) -> Result<Arc<AppProvingKey<SdkVmConfig>>> {
    let (manifest_path, _) = get_manifest_path_and_dir(&cargo_args.manifest_path)?;
    let target_dir = get_target_dir(&cargo_args.target_dir, &manifest_path);

    let app_pk_path = if let Some(app_pk) = app_pk {
        app_pk.to_path_buf()
    } else {
        get_app_pk_path(&target_dir)
    };

    Ok(Arc::new(read_app_pk_from_file(app_pk_path)?))
}

// Returns (committed_exe, target_name) where target_name has no extension
pub(crate) fn load_or_build_and_commit_exe(
    sdk: &Sdk,
    run_args: &RunArgs,
    cargo_args: &RunCargoArgs,
    app_pk: &Arc<AppProvingKey<SdkVmConfig>>,
) -> Result<(Arc<NonRootCommittedExe>, String)> {
    let exe_path = if let Some(exe) = &run_args.exe {
        exe
    } else {
        // Build and get the executable name
        let target_name = get_single_target_name(cargo_args)?;
        let build_args = run_args.clone().into();
        let cargo_args = cargo_args.clone().into();
        let output_dir = build(&build_args, &cargo_args)?;
        &output_dir.join(format!("{}.vmexe", target_name))
    };

    let app_exe = read_exe_from_file(exe_path)?;
    let committed_exe = sdk.commit_app_exe(app_pk.app_fri_params(), app_exe)?;
    Ok((
        committed_exe,
        exe_path.file_stem().unwrap().to_string_lossy().into_owned(),
    ))
}
