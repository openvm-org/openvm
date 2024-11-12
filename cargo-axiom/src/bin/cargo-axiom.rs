use cargo_axiom::{
    commands::{bench::BenchCmd, build::BuildCmd},
    AXVM_VERSION_MESSAGE,
};
use clap::{Parser, Subcommand};
use eyre::Result;

#[derive(Parser)]
#[command(name = "cargo", bin_name = "cargo")]
pub enum Cargo {
    Axiom(AxVmCli),
}

#[derive(clap::Args)]
#[command(author, about, long_about = None, args_conflicts_with_subcommands = true, version = AXVM_VERSION_MESSAGE)]
pub struct AxVmCli {
    #[clap(subcommand)]
    pub command: AxVmCliCommands,
}

#[derive(Subcommand)]
pub enum AxVmCliCommands {
    // New(NewCmd),
    Build(BuildCmd),
    Bench(BenchCmd),
}

fn main() -> Result<()> {
    let Cargo::Axiom(args) = Cargo::parse();
    let command = args.command;
    match command {
        AxVmCliCommands::Build(cmd) => cmd.run(),
        AxVmCliCommands::Bench(cmd) => cmd.run(),
    }
}
