use cargo_openvm::{
    commands::{BenchCmd, BuildCmd, EvmProvingSetupCmd, KeygenCmd, ProveCmd, RunCmd, VerifyCmd},
    OPENVM_VERSION_MESSAGE,
};
use clap::{Parser, Subcommand};
use eyre::Result;

#[derive(Parser)]
#[command(name = "cargo", bin_name = "cargo")]
pub enum Cargo {
    #[command(name = "openvm")]
    OpenVm(VmCli),
}

#[derive(clap::Args)]
#[command(author, about, long_about = None, args_conflicts_with_subcommands = true, version = OPENVM_VERSION_MESSAGE)]
pub struct VmCli {
    #[clap(subcommand)]
    pub command: VmCliCommands,
}

#[derive(Subcommand)]
pub enum VmCliCommands {
    Bench(BenchCmd),
    Build(BuildCmd),
    Keygen(KeygenCmd),
    Prove(ProveCmd),
    Run(RunCmd),
    Setup(EvmProvingSetupCmd),
    Verify(VerifyCmd),
}

#[tokio::main]
async fn main() -> Result<()> {
    let Cargo::OpenVm(args) = Cargo::parse();
    let command = args.command;
    match command {
        VmCliCommands::Bench(cmd) => cmd.run(),
        VmCliCommands::Build(cmd) => cmd.run(),
        VmCliCommands::Run(cmd) => cmd.run(),
        VmCliCommands::Keygen(cmd) => cmd.run(),
        VmCliCommands::Prove(cmd) => cmd.run(),
        VmCliCommands::Setup(cmd) => cmd.run().await,
        VmCliCommands::Verify(cmd) => cmd.run(),
    }
}
