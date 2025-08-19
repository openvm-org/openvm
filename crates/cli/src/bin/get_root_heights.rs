use std::path::PathBuf;

use clap::Parser;
use openvm_sdk::{
    fs::{read_object_from_file, write_to_file_json},
    keygen::AggProvingKey,
};

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(long)]
    pub path: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Cli::parse();
    let pk: AggProvingKey = read_object_from_file(&args.path)?;
    write_to_file_json(
        args.path.with_extension("root.vk.json"),
        pk.root_verifier_pk.vm_pk.vm_pk.get_vk(),
    )?;
    println!(
        "root verifier circuit fixed air heights: {:?}",
        pk.root_verifier_pk.air_heights
    );

    Ok(())
}
