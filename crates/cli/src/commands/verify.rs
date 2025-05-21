use std::path::PathBuf;

use clap::Parser;
use eyre::Result;
use openvm_sdk::{
    fs::{read_app_proof_from_file, read_app_vk_from_file},
    Sdk,
};

use super::KeygenCargoArgs;
use crate::{
    default::*,
    util::{get_app_vk_path, get_manifest_path_and_dir, get_target_dir},
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
            default_value = DEFAULT_APP_PROOF_PATH,
            help = "Path to app proof",
            help_heading = "OpenVM Options"
        )]
        proof: PathBuf,

        #[command(flatten)]
        cargo_args: KeygenCargoArgs,
    },
    #[cfg(feature = "evm-verify")]
    Evm {
        #[arg(
            long,
            action,
            default_value = DEFAULT_EVM_PROOF_PATH,
            help = "Path to EVM proof",
            help_heading = "OpenVM Options"
        )]
        proof: PathBuf,
    },
}

impl VerifyCmd {
    pub fn run(&self) -> Result<()> {
        let sdk = Sdk::new();
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

                let app_vk = read_app_vk_from_file(app_vk_path)?;
                let app_proof = read_app_proof_from_file(proof)?;
                sdk.verify_app_proof(&app_vk, &app_proof)?;
            }
            #[cfg(feature = "evm-verify")]
            VerifySubCommand::Evm { proof } => {
                use openvm_sdk::fs::{
                    read_evm_halo2_verifier_from_folder, read_evm_proof_from_file,
                };

                let evm_verifier =
                    read_evm_halo2_verifier_from_folder(default_evm_halo2_verifier_path())
                        .map_err(|e| {
                            eyre::eyre!(
                        "Failed to read EVM verifier: {}\nPlease run 'cargo openvm setup' first",
                        e
                    )
                        })?;
                let evm_proof = read_evm_proof_from_file(proof)?;
                sdk.verify_evm_halo2_proof(&evm_verifier, evm_proof)?;
            }
        }
        Ok(())
    }
}
