use std::{
    fs::{copy, create_dir_all},
    path::PathBuf,
};

use clap::Parser;
use eyre::{Context, Result};
use openvm_continuations::CommitBytes;
use openvm_sdk::{
    config::AggregationSystemParams,
    fs::{read_object_from_file, write_object_to_file, write_to_file_json},
    types::{AppExecutionCommit, VerificationBaselineJson},
    Sdk,
};
use p3_bn254::Bn254;

use super::{RunArgs, RunCargoArgs};
use crate::{
    args::OpenVmConfigArgs,
    commands::{load_app_pk, load_or_build_exe, ExecutionMode},
    util::{
        get_agg_pk_path, get_agg_vk_path, get_app_baseline_path, get_app_commit_path,
        get_manifest_path_and_dir, get_single_target_name, get_target_dir, get_target_output_dir,
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
        help = "Path to app proving key, by default will be ${openvm_dir}/app.pk",
        help_heading = "OpenVM Options"
    )]
    pub app_pk: Option<PathBuf>,

    #[arg(
        long,
        action,
        help = "Path to OpenVM executable, if specified build will be skipped",
        help_heading = "OpenVM Options"
    )]
    pub exe: Option<PathBuf>,

    #[clap(flatten)]
    pub openvm_config: OpenVmConfigArgs,

    #[command(flatten)]
    cargo_args: RunCargoArgs,
}

impl CommitCmd {
    pub fn run(&self) -> Result<()> {
        let app_pk = load_app_pk(&self.app_pk, &self.cargo_args)?;

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

        let mut builder = Sdk::builder().app_pk(app_pk);
        let agg_pk_path = get_agg_pk_path(&target_dir);
        if agg_pk_path.exists() {
            builder = builder.agg_pk(read_object_from_file(&agg_pk_path)?);
        } else {
            builder = builder.agg_params(AggregationSystemParams::default());
        }
        let root_pk_path = PathBuf::from(crate::default::default_root_pk_path());
        if root_pk_path.exists() {
            builder = builder.root_pk(read_object_from_file(&root_pk_path)?);
        }
        let sdk = builder.build()?;

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

        // Save keys for reuse
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
        if !root_pk_path.exists() {
            println!("Writing root proving key to {}", root_pk_path.display());
            write_object_to_file(&root_pk_path, sdk.root_pk())?;
        }

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
