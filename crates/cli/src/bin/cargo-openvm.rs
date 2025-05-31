use cargo_openvm::{commands::*, OPENVM_VERSION_MESSAGE};
use clap::{Parser, Subcommand};
use eyre::Result;
use openvm_stark_sdk::config::setup_tracing_with_log_level;
use tracing::Level;

#[derive(Parser)]
#[command(name = "cargo", bin_name = "cargo")]
pub enum Cargo {
    #[command(name = "openvm")]
    OpenVm(VmCli),
}

#[derive(clap::Args)]
#[command(author, about, long_about = None, args_conflicts_with_subcommands = true, version = OPENVM_VERSION_MESSAGE)]
pub struct VmCli {
    #[command(subcommand)]
    pub command: VmCliCommands,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
pub enum VmCliCommands {
    Build(BuildCmd),
    Commit(CommitCmd),
    Keygen(KeygenCmd),
    Init(InitCmd),
    Prove(ProveCmd),
    Run(RunCmd),
    #[cfg(feature = "evm-verify")]
    Setup(SetupCmd),
    Verify(VerifyCmd),
}

#[tokio::main]
async fn main() -> Result<()> {
    let Cargo::OpenVm(args) = Cargo::parse();
    let command = args.command;
    setup_tracing_with_log_level(Level::WARN);
    match command {
        VmCliCommands::Build(cmd) => cmd.run(),
        VmCliCommands::Commit(cmd) => cmd.run(),
        VmCliCommands::Keygen(cmd) => cmd.run(),
        VmCliCommands::Init(cmd) => cmd.run(),
        VmCliCommands::Prove(cmd) => cmd.run(),
        VmCliCommands::Run(cmd) => cmd.run(),
        #[cfg(feature = "evm-verify")]
        VmCliCommands::Setup(cmd) => cmd.run().await,
        VmCliCommands::Verify(cmd) => cmd.run(),
    }
}
