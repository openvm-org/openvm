use std::{fs, path::PathBuf};

use clap::Parser;
use verify_stark::{verify_vm_stark_proof, vk::read_vk_from_file};

#[derive(Debug, Parser)]
pub struct CliArgs {
    #[clap(long)]
    vk_path: PathBuf,

    #[clap(long)]
    proof_path: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = CliArgs::parse();

    let vk = read_vk_from_file(args.vk_path)?;
    let encoded_proof_without_version = fs::read(args.proof_path)?;

    verify_vm_stark_proof(&vk, encoded_proof_without_version)?;
    println!("Proof verified successfully!");

    Ok(())
}
