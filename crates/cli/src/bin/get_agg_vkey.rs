use std::path::PathBuf;

use clap::Parser;
use openvm_sdk::{
    fs::{read_object_from_file, write_object_to_file, write_to_file_json},
    keygen::AggStarkProvingKey,
};

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(long)]
    pub path: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Cli::parse();
    let pk: AggStarkProvingKey = read_object_from_file(&args.path)?;
    let vk = pk.get_agg_vk();
    write_object_to_file(args.path.with_extension("vk"), &vk)?;
    write_to_file_json(args.path.with_extension("vk.json"), &vk)?;
    Ok(())
}
