use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
struct ExecutorArgs {
    #[arg(long)]
    program_dir: PathBuf,
    // input -> what type?
}

fn main() {
    let _args = ExecutorArgs::parse();
    // 1. get the exe from program dir
    println!("Hello, world!");
}
