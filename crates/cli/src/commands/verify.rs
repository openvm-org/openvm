use std::path::{Path, PathBuf};

use clap::Parser;
use eyre::Result;
use openvm_sdk::{
    fs::{decode_from_file, read_from_file_json, read_object_from_file},
    prover::verify_app_proof,
    types::VmStarkProofBytes,
    Sdk,
};

use super::KeygenCargoArgs;
#[cfg(feature = "evm-verify")]
use crate::default::default_evm_halo2_verifier_path;
use crate::{
    commands::RunCargoArgs,
    default::default_agg_stark_vk_path,
    util::{
        get_app_commit_path, get_app_vk_path, get_files_with_ext, get_manifest_path_and_dir,
        get_single_target_name, get_target_dir, get_target_output_dir,
    },
};

#[derive(Parser)]
#[command(name = "verify", about = "Verify a proof")]
pub struct VerifyCmd {
    #[command(subcommand)]
    command: VerifySubCommand,
}

#[derive(Parser)]
enum VerifySubCommand {
    App {
        #[arg(
            long,
            action,
            help = "Path to app verifying key, by default will search for it in ${target_dir}/openvm/app.vk",
            help_heading = "OpenVM Options"
        )]
        app_vk: Option<PathBuf>,

        #[arg(
            long,
            action,
            help = "Path to app proof, by default will search the working directory for a file with extension .app.proof",
            help_heading = "OpenVM Options"
        )]
        proof: Option<PathBuf>,

        #[command(flatten)]
        cargo_args: KeygenCargoArgs,
    },
    Stark {
        /// NOTE: if `openvm commit` was called with the `--exe` option, then `--app-commit` must
        /// be specified so the command knows where to find the app commit.
        #[arg(
            long,
            action,
            help = "Path to app commit, by default will search for it using the binary target name",
            help_heading = "OpenVM Options"
        )]
        app_commit: Option<PathBuf>,

        #[arg(
            long,
            action,
            help = "Path to STARK proof, by default will search the working directory for a file with extension .stark.proof",
            help_heading = "OpenVM Options"
        )]
        proof: Option<PathBuf>,

        #[command(flatten)]
        cargo_args: RunCargoArgs,
    },
    #[cfg(feature = "evm-verify")]
    Evm {
        #[arg(
            long,
            action,
            help = "Path to EVM proof, by default will search the working directory for a file with extension .evm.proof",
            help_heading = "OpenVM Options"
        )]
        proof: Option<PathBuf>,
    },
}

impl VerifyCmd {
    pub fn run(&self) -> Result<()> {
        match &self.command {
            VerifySubCommand::App {
                app_vk,
                proof,
                cargo_args,
            } => {
                let app_vk_path = if let Some(app_vk) = app_vk {
                    app_vk.to_path_buf()
                } else {
                    let (manifest_path, _) = get_manifest_path_and_dir(&cargo_args.manifest_path)?;
                    let target_dir = get_target_dir(&cargo_args.target_dir, &manifest_path);
                    get_app_vk_path(&target_dir)
                };
                let app_vk = read_object_from_file(app_vk_path)?;

                let proof_path = if let Some(proof) = proof {
                    proof.clone()
                } else {
                    let files = get_files_with_ext(Path::new("."), "app.proof")?;
                    if files.len() > 1 {
                        return Err(eyre::eyre!("multiple .app.proof files found, please specify the path using option --proof"));
                    } else if files.is_empty() {
                        return Err(eyre::eyre!("no .app.proof file found, please specify the path using option --proof"));
                    }
                    files[0].clone()
                };
                println!("Verifying application proof at {}", proof_path.display());
                let app_proof = decode_from_file(proof_path)?;
                verify_app_proof(&app_vk, &app_proof)?;
            }
            VerifySubCommand::Stark {
                app_commit,
                proof,
                cargo_args,
            } => {
                let agg_vk = read_object_from_file(default_agg_stark_vk_path())
                    .map_err(|e| {
                        eyre::eyre!(
                        "Failed to read aggregation STARK verifying key: {e}\nPlease run 'cargo openvm setup' first",
                    )
                    })?;
                let app_commit_path = if let Some(app_commit) = app_commit {
                    app_commit.to_path_buf()
                } else {
                    let (manifest_path, _) = get_manifest_path_and_dir(&cargo_args.manifest_path)?;
                    let target_dir = get_target_dir(&cargo_args.target_dir, &manifest_path);
                    let target_output_dir = get_target_output_dir(&target_dir, &cargo_args.profile);
                    let target_name = get_single_target_name(cargo_args)?;
                    get_app_commit_path(&target_output_dir, target_name)
                };
                let expected_app_commit = read_from_file_json(app_commit_path)?;

                let proof_path = if let Some(proof) = proof {
                    proof.clone()
                } else {
                    let files = get_files_with_ext(Path::new("."), "stark.proof")?;
                    if files.len() > 1 {
                        return Err(eyre::eyre!("multiple .stark.proof files found, please specify the path using option --proof"));
                    } else if files.is_empty() {
                        return Err(eyre::eyre!("no .stark.proof file found, please specify the path using option --proof"));
                    }
                    files[0].clone()
                };
                println!("Verifying STARK proof at {}", proof_path.display());
                let stark_proof_bytes: VmStarkProofBytes = read_from_file_json(proof_path)?;
                Sdk::verify_proof(&agg_vk, expected_app_commit, &stark_proof_bytes.try_into()?)?;
            }
            #[cfg(feature = "evm-verify")]
            VerifySubCommand::Evm { proof } => {
                use openvm_sdk::fs::read_evm_halo2_verifier_from_folder;

                let evm_verifier =
                    read_evm_halo2_verifier_from_folder(default_evm_halo2_verifier_path())
                        .map_err(|e| {
                            eyre::eyre!(
                        "Failed to read EVM verifier: {e}\nPlease run 'cargo openvm setup' first"
                    )
                        })?;

                let proof_path = if let Some(proof) = proof {
                    proof.clone()
                } else {
                    let files = get_files_with_ext(Path::new("."), "evm.proof")?;
                    if files.len() > 1 {
                        return Err(eyre::eyre!("multiple .evm.proof files found, please specify the path using option --proof"));
                    } else if files.is_empty() {
                        return Err(eyre::eyre!("no .evm.proof file found, please specify the path using option --proof"));
                    }
                    files[0].clone()
                };
                // The app config used here doesn't matter, it is ignored in verification
                println!("Verifying EVM proof at {}", proof_path.display());
                let evm_proof = read_from_file_json(proof_path)?;
                Sdk::verify_evm_halo2_proof(&evm_verifier, evm_proof)?;
            }
        }
        Ok(())
    }
}
