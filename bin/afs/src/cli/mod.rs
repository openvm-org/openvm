use crate::commands::{cache, keygen, mock, prove, verify};
use crate::{
    commands::{
        cache::CacheCommand, keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand,
    },
    common::config::Config,
};
use clap::Parser;
use clap::Subcommand;

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS CLI")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    #[command(name = "mock", about = "Mock functions")]
    /// Mock functions
    Mock(mock::MockCommand),

    #[command(name = "keygen", about = "Generate partial proving and verifying keys")]
    /// Generate partial proving and verifying keys
    Keygen(keygen::KeygenCommand),

    #[command(
        name = "cache",
        about = "Create the cached trace of a page from a page file"
    )]
    /// Create cached trace of a page from a page file
    Cache(cache::CacheCommand),

    #[command(name = "prove", about = "Generates a multi-STARK proof")]
    /// Generates a multi-STARK proof
    Prove(prove::ProveCommand),

    #[command(name = "verify", about = "Verifies a multi-STARK proof")]
    /// Verifies a multi-STARK proof
    Verify(verify::VerifyCommand),
}

impl Cli {
    pub fn run(config: &Config) -> Self {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Mock(mock) => {
                mock.execute().unwrap();
            }
            CliCommand::Keygen(keygen) => {
                let cmd = KeygenCommand {
                    output_folder: keygen.output_folder.clone(),
                };
                cmd.execute(config).unwrap();
            }
            CliCommand::Cache(cache) => {
                let cmd = CacheCommand {
                    page_file: cache.page_file.clone(),
                    output_file: cache.output_file.clone(),
                };
                cmd.execute().unwrap();
            }
            CliCommand::Prove(prove) => {
                prove.execute(config).unwrap();
            }
            CliCommand::Verify(verify) => {
                verify.execute().unwrap();
            }
        }
        cli
    }
}
