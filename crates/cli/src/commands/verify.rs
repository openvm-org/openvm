use std::path::PathBuf;

use clap::Parser;
use eyre::{Context, Result};
use openvm_sdk::{
    fs::{read_from_file_json, read_object_from_file},
    prover::verify_app_proof,
    types::{VerificationBaselineJson, VersionedNonRootStarkProof},
    Sdk, OPENVM_VERSION, SC,
};
use openvm_stark_backend::keygen::types::MultiStarkProvingKey;

use super::KeygenCargoArgs;
use crate::{
    args::ManifestArgs,
    default::{APP_PROOF_EXT, STARK_PROOF_EXT},
    util::{
        get_app_baseline_path, get_app_vk_path, get_manifest_path_and_dir,
        get_single_target_name_raw, get_target_dir, get_target_output_dir, resolve_proof_path,
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
            help = "Path to app verifying key, by default will search for it in ${openvm_dir}/app.vk",
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
        /// NOTE: if `openvm commit` was called with the `--exe` option, then `--app-baseline` must
        /// be specified so the command knows where to find the baseline file.
        #[arg(
            long,
            action,
            help = "Path to app baseline (.baseline.json), by default will search for it using the binary target name",
            help_heading = "OpenVM Options"
        )]
        app_baseline: Option<PathBuf>,

        #[arg(
            long,
            action,
            help = "Path to STARK proof, by default will search the working directory for a file with extension .stark.proof",
            help_heading = "OpenVM Options"
        )]
        proof: Option<PathBuf>,

        #[command(flatten)]
        cargo_args: SingleTargetCargoArgs,
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

#[derive(Parser)]
pub struct SingleTargetCargoArgs {
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
        value_name = "NAME",
        default_value = "release",
        help = "Run with the given profile",
        help_heading = "Compilation Options"
    )]
    pub profile: String,

    #[clap(flatten)]
    pub manifest: ManifestArgs,
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
                    let (manifest_path, _) =
                        get_manifest_path_and_dir(&cargo_args.manifest.manifest_path)?;
                    let target_dir =
                        get_target_dir(&cargo_args.manifest.target_dir, &manifest_path);
                    get_app_vk_path(&target_dir)
                };
                let app_vk: openvm_sdk::keygen::AppVerifyingKey =
                    read_object_from_file(app_vk_path)?;

                let proof_path = resolve_proof_path(proof, APP_PROOF_EXT)?;
                println!("Verifying application proof at {}", proof_path.display());
                let app_proof = read_object_from_file(proof_path)?;
                let _exe_commit = verify_app_proof::<openvm_sdk::DefaultStarkEngine>(
                    &app_vk.vk,
                    app_vk.memory_dimensions,
                    &app_proof,
                )?;
            }
            VerifySubCommand::Stark {
                app_baseline,
                proof,
                cargo_args,
            } => {
                let (manifest_path, _) =
                    get_manifest_path_and_dir(&cargo_args.manifest.manifest_path)?;
                let target_dir = get_target_dir(&cargo_args.manifest.target_dir, &manifest_path);
                let internal_recursive_pk_path =
                    PathBuf::from(crate::default::default_internal_recursive_pk_path());
                let internal_recursive_pk: std::sync::Arc<MultiStarkProvingKey<SC>> =
                    read_object_from_file(&internal_recursive_pk_path).map_err(|e| {
                    eyre::eyre!(
                        "Failed to read internal-recursive proving key from {}: {e}\nRun 'cargo openvm setup' first to generate it",
                        internal_recursive_pk_path.display()
                    )
                })?;
                let agg_vk = internal_recursive_pk.get_vk();
                let baseline_path = if let Some(app_baseline) = app_baseline {
                    app_baseline.to_path_buf()
                } else {
                    let target_output_dir = get_target_output_dir(&target_dir, &cargo_args.profile);
                    let target_name = get_single_target_name_raw(
                        &cargo_args.bin,
                        &cargo_args.example,
                        &cargo_args.manifest.manifest_path,
                        &cargo_args.package,
                    )?;
                    get_app_baseline_path(&target_output_dir, target_name)
                };
                let baseline_json: VerificationBaselineJson = read_from_file_json(baseline_path)?;
                let expected_app_commit = baseline_json.into();

                let proof_path = resolve_proof_path(proof, STARK_PROOF_EXT)?;
                println!("Verifying STARK proof at {}", proof_path.display());
                let stark_proof: VersionedNonRootStarkProof = read_from_file_json(proof_path)
                    .with_context(|| {
                        format!("Proof needs to be compatible with openvm v{OPENVM_VERSION}",)
                    })?;
                if stark_proof.version != format!("v{OPENVM_VERSION}") {
                    eprintln!("Attempting to verify proof generated with openvm {}, but the verifier is on openvm v{OPENVM_VERSION}", stark_proof.version);
                }
                Sdk::verify_proof(agg_vk, expected_app_commit, &stark_proof.try_into()?)?;
            }
            #[cfg(feature = "evm-verify")]
            VerifySubCommand::Evm { proof } => {
                use openvm_sdk::{fs::read_evm_halo2_verifier_from_folder, types::EvmProof};

                let verifier_path =
                    PathBuf::from(crate::default::default_evm_halo2_verifier_path());
                let evm_verifier = read_evm_halo2_verifier_from_folder(&verifier_path).map_err(
                    |e| {
                        eyre::eyre!(
                            "Failed to read EVM verifier from {}: {e}\nRun 'cargo openvm setup' to generate it",
                            verifier_path.display()
                        )
                    },
                )?;

                let proof_path = resolve_proof_path(proof, crate::default::EVM_PROOF_EXT)?;
                // The app config used here doesn't matter, it is ignored in verification
                println!("Verifying EVM proof at {}", proof_path.display());
                let evm_proof: EvmProof = read_from_file_json(proof_path).with_context(|| {
                    format!("Proof needs to be compatible with openvm v{OPENVM_VERSION}",)
                })?;
                if evm_proof.version != format!("v{OPENVM_VERSION}") {
                    eprintln!("Attempting to verify proof generated with openvm {}, but the verifier is on openvm v{OPENVM_VERSION}", evm_proof.version);
                }
                Sdk::verify_evm_halo2_proof(&evm_verifier, evm_proof)?;
            }
        }
        println!("Proof verified successfully!");
        Ok(())
    }
}
