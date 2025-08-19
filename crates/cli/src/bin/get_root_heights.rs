use std::path::PathBuf;

use clap::Parser;
use openvm_sdk::{fs::read_object_from_file, keygen::AggProvingKey};

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(long)]
    pub path: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Cli::parse();
    let pk: AggProvingKey = read_object_from_file(&args.path)?;
    println!(
        "root verifier circuit fixed air heights: {:?}",
        pk.root_verifier_pk.air_heights
    );

    Ok(())
}
