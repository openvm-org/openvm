use std::{
    fs::{copy, create_dir_all},
    path::PathBuf,
};

use clap::Parser;
use eyre::{Context, Result};
use openvm_continuations::CommitBytes;
use openvm_sdk::{
    fs::write_to_file_json,
    types::{AppExecutionCommit, VerificationBaselineJson},
    Sdk,
};
use p3_bn254::Bn254;

use super::{
    prove::load_required_agg_pk,
    RunArgs, RunCargoArgs
};
use crate::{
    args::{OpenVmConfigArgs, ProvingKeyArgs},
    commands::{load_app_pk, load_or_build_exe, ExecutionMode},
    util::{
        get_app_baseline_path, get_app_commit_path, get_manifest_path_and_dir,
        get_single_target_name, get_target_dir, get_target_output_dir,
    },
};

#[derive(Parser)]
#[command(
    name = "commit",
    about = "View the Bn254 commit of an OpenVM executable"
)]
pub struct CommitCmd {
    #[arg(
        long,
        action,
        help = "Path to OpenVM executable, if specified build will be skipped",
        help_heading = "OpenVM Options"
    )]
    pub exe: Option<PathBuf>,

    #[command(flatten)]
    pub keys: ProvingKeyArgs,

    #[clap(flatten)]
    pub openvm_config: OpenVmConfigArgs,

    #[command(flatten)]
    cargo_args: RunCargoArgs,
}

impl CommitCmd {
    pub fn run(&self) -> Result<()> {
        let app_pk = load_app_pk(&self.keys.app_pk, &self.cargo_args)?;

        let run_args = RunArgs {
            exe: self.exe.clone(),
            openvm_config: self.openvm_config.clone(),
            input: None,
            mode: ExecutionMode::Pure,
        };
        let (exe, target_name_stem) = load_or_build_exe(&run_args, &self.cargo_args)?;
        let (manifest_path, _) =
            get_manifest_path_and_dir(&self.cargo_args.manifest.manifest_path)?;
        let target_dir = get_target_dir(&self.cargo_args.manifest.target_dir, &manifest_path);

        let agg_pk = load_required_agg_pk(&self.keys.agg_prefix_pk, &self.cargo_args)?;
        let sdk = Sdk::builder()
            .app_pk(app_pk)
            .agg_pk(agg_pk)
            .build()?;

        let prover = sdk.prover(exe)?;
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

        let target_output_dir = get_target_output_dir(&target_dir, &self.cargo_args.profile);

        // target_name_stem does not contain "examples/" prefix
        let target_name =
            get_single_target_name(&self.cargo_args).unwrap_or(target_name_stem.into());

        // Write Bn254 commit values
        let commit_path = get_app_commit_path(&target_output_dir, target_name.clone());
        println!("Writing app commit to {}", commit_path.display());
        write_to_file_json(&commit_path, &app_commit)?;

        // Write verification baseline (used by `verify stark`)
        let baseline_path = get_app_baseline_path(&target_output_dir, target_name);
        println!("Writing baseline to {}", baseline_path.display());
        let baseline_json: VerificationBaselineJson = baseline.into();
        write_to_file_json(&baseline_path, &baseline_json)?;

        if let Some(output_dir) = &self.openvm_config.output_dir {
            create_dir_all(output_dir)
                .with_context(|| format!("failed to create directory {}", output_dir.display()))?;
            let commit_name = commit_path.file_name().unwrap();
            copy(&commit_path, output_dir.join(commit_name))
                .with_context(|| format!("failed to copy commit to {}", output_dir.display()))?;
            let baseline_name = baseline_path.file_name().unwrap();
            copy(&baseline_path, output_dir.join(baseline_name))
                .with_context(|| format!("failed to copy baseline to {}", output_dir.display()))?;
        }

        Ok(())
    }
}
