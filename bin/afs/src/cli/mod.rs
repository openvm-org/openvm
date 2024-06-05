pub mod types;
use types::CliCommand;
use crate::{
    commands::{
        cache::CacheCommand,
        keygen::KeygenCommand,
        prove::ProveCommand,
        verify::VerifyCommand,
    },
    common::config::Config
};

use clap::Parser;

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: types::CliCommand,
}

impl Cli {
    pub fn run(config: &Config) -> Self {
        let cli = Self::parse();
        match &cli.command {
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
                let cmd = ProveCommand {
                    ops_file: prove.ops_file.clone(),
                    output_file: prove.output_file.clone(),
                };
                cmd.execute().unwrap();
            }
            CliCommand::Verify(verify) => {
                let cmd = VerifyCommand {
                    proof_file: verify.proof_file.clone(),
                };
                cmd.execute().unwrap();
            }
        }
        cli
    }
}
