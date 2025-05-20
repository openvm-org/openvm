use std::{
    fs::{copy, create_dir_all},
    path::PathBuf,
};

use clap::Parser;

use eyre::Result;
use openvm_circuit::arch::OPENVM_DEFAULT_INIT_FILE_NAME;
use openvm_sdk::{
    commit::AppExecutionCommit,
    fs::{write_app_exe_bn254_commit_to_file, write_app_exe_commit_to_file},
    Sdk,
};

use crate::{
    commands::{load_app_pk, load_or_build_and_commit_exe},
    util::{
        get_manifest_path_and_dir, get_single_target_name, get_target_dir, get_target_output_dir,
    },
};

use super::{RunArgs, RunCargoArgs};

#[derive(Parser)]
#[command(name = "commit", about = "View the commit of an OpenVM executable")]
pub struct CommitCmd {
    #[arg(
        long,
        action,
        help = "Path to app proving key, by default will be ${target_dir}/openvm/app.pk",
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
        let sdk = Sdk::new();
        let app_pk = load_app_pk(&self.app_pk, &self.cargo_args)?;

        let run_args = RunArgs {
            exe: self.exe.clone(),
            config: self.config.clone(),
            output_dir: self.output_dir.clone(),
            init_file_name: self.init_file_name.clone(),
            input: None,
        };
        let committed_exe =
            load_or_build_and_commit_exe(&sdk, &run_args, &self.cargo_args, &app_pk)?;

        let commits = AppExecutionCommit::compute(
            &app_pk.app_vm_pk.vm_config,
            &committed_exe,
            &app_pk.leaf_committed_exe,
        );
        let bn254_commits = commits.to_bn254_commit();
        println!("exe commit ([u32]): {:?}", commits.exe_commit);
        println!("exe commit (Bn254): {:?}", bn254_commits.exe_commit);
        println!("vm commit ([u32]): {:?}", commits.vm_commit);
        println!("vm commit (Bn254): {:?}", bn254_commits.vm_commit);

        let (manifest_path, _) = get_manifest_path_and_dir(&self.cargo_args.manifest_path)?;
        let target_dir = get_target_dir(&self.cargo_args.target_dir, &manifest_path);
        let target_output_dir = get_target_output_dir(&target_dir, &self.cargo_args.profile);
        let target_name = get_single_target_name(&self.cargo_args)?;

        let commit_name = format!("{}.commit.json", &target_name);
        let bn254_name = format!("{}.commit.bn254.json", &target_name);
        let commit_path = target_output_dir.join(&commit_name);
        let bn254_path = target_output_dir.join(&bn254_name);

        write_app_exe_commit_to_file(commits, &commit_path)?;
        write_app_exe_bn254_commit_to_file(bn254_commits, &bn254_path)?;
        if let Some(output_dir) = &self.output_dir {
            create_dir_all(output_dir)?;
            copy(commit_path, output_dir.join(commit_name))?;
            copy(bn254_path, output_dir.join(bn254_name))?;
        }

        Ok(())
    }
}
