use std::{
    fs::{copy, create_dir_all},
    path::PathBuf,
};

use clap::Parser;
use eyre::Result;
use openvm_circuit::arch::OPENVM_DEFAULT_INIT_FILE_NAME;
use openvm_continuations::CommitBytes;
use openvm_sdk::{
    config::AggregationSystemParams,
    fs::{read_object_from_file, write_object_to_file, write_to_file_json},
    types::AppExecutionCommit,
    Sdk,
};
use p3_bn254::Bn254;

use super::{RunArgs, RunCargoArgs};
use crate::{
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
        default_value = OPENVM_DEFAULT_INIT_FILE_NAME,
        help = "Name of the init file",
        help_heading = "OpenVM Options"
    )]
    pub init_file_name: String,

    #[command(flatten)]
    cargo_args: RunCargoArgs,
}

impl CommitCmd {
    pub fn run(&self) -> Result<()> {
        let app_pk = load_app_pk(&self.app_pk, &self.cargo_args)?;

        let run_args = RunArgs {
            exe: self.exe.clone(),
            config: self.config.clone(),
            output_dir: self.output_dir.clone(),
            init_file_name: self.init_file_name.clone(),
            input: None,
            mode: ExecutionMode::Pure,
        };
        let (exe, target_name_stem) = load_or_build_exe(&run_args, &self.cargo_args)?;
        let (manifest_path, _) = get_manifest_path_and_dir(&self.cargo_args.manifest_path)?;
        let target_dir = get_target_dir(&self.cargo_args.target_dir, &manifest_path);

        let mut sdk =
            Sdk::new(app_pk.app_config(), AggregationSystemParams::default())?.with_app_pk(app_pk);
        let agg_pk_path = get_agg_pk_path(&target_dir);
        if agg_pk_path.exists() {
            sdk = sdk.with_agg_pk(read_object_from_file(&agg_pk_path)?);
        }
        let root_pk_path = PathBuf::from(crate::default::default_root_pk_path());
        if root_pk_path.exists() {
            sdk = sdk.with_root_pk(read_object_from_file(&root_pk_path)?);
        }

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
        write_to_file_json(&baseline_path, &baseline)?;

        if let Some(output_dir) = &self.output_dir {
            create_dir_all(output_dir)?;
            let commit_name = commit_path.file_name().unwrap();
            copy(&commit_path, output_dir.join(commit_name))?;
            let baseline_name = baseline_path.file_name().unwrap();
            copy(&baseline_path, output_dir.join(baseline_name))?;
        }

        Ok(())
    }
}
