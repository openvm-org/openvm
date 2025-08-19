use std::{fs, path::PathBuf};

use clap::Parser;
use openvm_sdk::{fs::read_object_from_file, keygen::Halo2ProvingKey};

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(long)]
    pub path: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Cli::parse();
    let pk: Halo2ProvingKey = read_object_from_file(&args.path)?;
    let vk1 = pk.verifier.pinning.pk.get_vk();
    fs::write(
        args.path.with_extension("verifier.vk.json"),
        format!("{:?}", vk1.pinned()),
    )?;
    let vk2 = pk.wrapper.pinning.pk.get_vk();
    fs::write(
        args.path.with_extension("wrapper.vk.json"),
        format!("{:?}", vk2.pinned()),
    )?;
    Ok(())
}
