use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use clap::Subcommand;

use crate::commands::{
    common::CommonCommands, keygen::KeygenCommand, prove::ProveCommand, verify::VerifyCommand,
};

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS Predicate CLI")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    #[command(name = "keygen", about = "Generate keys")]
    Keygen(KeygenCommand),

    #[command(name = "prove", about = "Prove the predicate operation")]
    Prove(ProveCommand),

    #[command(name = "verify", about = "Verify the predicate operation")]
    Verify(VerifyCommand),
}

impl Cli {
    pub fn run(config: &PageConfig) -> Self {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Keygen(keygen) => {
                let cmd = KeygenCommand {
                    common: CommonCommands {
                        predicate: keygen.common.predicate.clone(),
                        value: keygen.common.value.clone(),
                        table_id: keygen.common.table_id.clone(),
                        db_file_path: keygen.common.db_file_path.clone(),
                        cache_folder: keygen.common.cache_folder.clone(),
                        output_folder: keygen.common.output_folder.clone(),
                        silent: keygen.common.silent,
                    },
                };
                cmd.execute(config).unwrap();
            }
            CliCommand::Prove(prove) => {
                let cmd = ProveCommand {
                    keys_folder: prove.keys_folder.clone(),
                    common: CommonCommands {
                        predicate: prove.common.predicate.clone(),
                        value: prove.common.value.clone(),
                        table_id: prove.common.table_id.clone(),
                        db_file_path: prove.common.db_file_path.clone(),
                        cache_folder: prove.common.cache_folder.clone(),
                        output_folder: prove.common.output_folder.clone(),
                        silent: prove.common.silent,
                    },
                };
                cmd.execute(config).unwrap();
            }
            CliCommand::Verify(verify) => {
                let cmd = VerifyCommand {
                    keys_folder: verify.keys_folder.clone(),
                    common: CommonCommands {
                        predicate: verify.common.predicate.clone(),
                        value: verify.common.value.clone(),
                        table_id: verify.common.table_id.clone(),
                        db_file_path: verify.common.db_file_path.clone(),
                        cache_folder: verify.common.cache_folder.clone(),
                        output_folder: verify.common.output_folder.clone(),
                        silent: verify.common.silent,
                    },
                };
                cmd.execute(config).unwrap();
            }
        }
        cli
    }
}
