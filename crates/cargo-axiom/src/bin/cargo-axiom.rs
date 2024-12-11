use cargo_axiom::{
    commands::{BenchCmd, BuildCmd, InitCmd, KeygenCmd, ProveCmd, RunCmd, VerifyCmd},
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
    Bench(BenchCmd),
    Build(BuildCmd),
    Init(InitCmd),
    Keygen(KeygenCmd),
    Prove(ProveCmd),
    Run(RunCmd),
    Verify(VerifyCmd),
}

fn main() -> Result<()> {
    let Cargo::Axiom(args) = Cargo::parse();
    let command = args.command;
    match command {
        AxVmCliCommands::Bench(cmd) => cmd.run(),
        AxVmCliCommands::Build(cmd) => cmd.run(),
        AxVmCliCommands::Run(cmd) => cmd.run(),
        AxVmCliCommands::Init(cmd) => cmd.run(),
        AxVmCliCommands::Keygen(cmd) => cmd.run(),
        AxVmCliCommands::Prove(cmd) => cmd.run(),
        AxVmCliCommands::Verify(cmd) => cmd.run(),
    }
}
