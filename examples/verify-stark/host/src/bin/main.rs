use std::path::PathBuf;

use clap::{Parser, Subcommand};
use eyre::Result;
use openvm_sdk::fs::{
    read_from_file_json, read_object_from_file, write_object_to_file, write_to_file_json,
};
use verify_stark_host_example::{keygen, prove};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Keygen(KeygenCmd),
    Prove(ProveCmd),
}

#[derive(Parser)]
struct KeygenCmd {
    #[arg(long, required = true)]
    child_agg_vk: PathBuf,

    #[arg(long, required = true)]
    sdk_pk: PathBuf,

    #[arg(long, required = true)]
    openvm_toml: PathBuf,

    #[arg(long, required = true)]
    agg_vk: PathBuf,
}

#[derive(Parser)]
struct ProveCmd {
    #[arg(long, required = true)]
    sdk_pk: PathBuf,

    #[arg(long, required = true)]
    child_agg_vk: PathBuf,

    #[arg(long, required = true)]
    child_baseline: PathBuf,

    #[arg(long, required = true)]
    inupt_proof: PathBuf,

    #[arg(long, required = true)]
    output_proof: PathBuf,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Keygen(cmd) => {
            let child_agg_vk = read_object_from_file(cmd.child_agg_vk)?;
            let (cached_pk, vm_config, agg_vk) = keygen(child_agg_vk)?;
            write_object_to_file(cmd.sdk_pk, cached_pk)?;
            if let Some(parent) = cmd
                .openvm_toml
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(cmd.openvm_toml, vm_config.to_toml()?)?;
            write_object_to_file(cmd.agg_vk, agg_vk)?;
            println!("verify-stark deferral proving keys written");
        }
        Command::Prove(cmd) => {
            let cached_pk = read_object_from_file(cmd.sdk_pk)?;
            let child_agg_vk = read_object_from_file(cmd.child_agg_vk)?;
            let child_baseline = read_object_from_file(cmd.child_baseline)?;
            let input_proof = read_from_file_json(cmd.inupt_proof)?;
            let output_proof = prove(cached_pk, child_agg_vk, child_baseline, input_proof)?;
            write_to_file_json(cmd.output_proof, output_proof)?;
            println!("verify-stark proof written");
        }
    }

    Ok(())
}
