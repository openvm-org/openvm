use std::path::PathBuf;

use clap::Parser;
use openvm_sdk::{
    fs::{read_object_from_file, write_to_file_json},
    keygen::AppVerifyingKey,
};

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(long)]
    pub path: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Cli::parse();
    let vk: AppVerifyingKey = read_object_from_file(&args.path)?;
    write_to_file_json(args.path.with_extension("json"), &vk)?;
    Ok(())
}
