use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use clap::Subcommand;

use crate::commands::common::CommonCommands;
use crate::commands::{eq, gt, gte, lt, lte};

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS Predicate CLI")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    #[command(name = "eq", about = "Get data in a Table that is equal to the input")]
    /// Get data in a Table that is equal to the input
    Eq(eq::EqCommand),

    #[command(
        name = "gt",
        about = "Get data in a Table that is greater than the input"
    )]
    /// Get data in a Table that is greater than the input
    Gt(gt::GtCommand),

    #[command(
        name = "gte",
        about = "Get data in a Table that is greater than or equal to the input"
    )]
    /// Get data in a Table that is greater than or equal to the input
    Gte(gte::GteCommand),

    #[command(name = "lt", about = "Get data in a Table that is less than the input")]
    /// Get data in a Table that is less than the input
    Lt(lt::LtCommand),

    #[command(
        name = "lte",
        about = "Get data in a Table that is less than or equal to the input"
    )]
    /// Get data in a Table that is less than or equal to the input
    Lte(lte::LteCommand),
}

impl Cli {
    pub fn run(config: &PageConfig) -> Self {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Eq(eq) => {
                let cmd = eq::EqCommand {
                    args: CommonCommands {
                        db_file_path: eq.args.db_file_path.clone(),
                        table_id: eq.args.table_id.clone(),
                        value: eq.args.value.clone(),
                        output_file: eq.args.output_file.clone(),
                        silent: eq.args.silent,
                    },
                };
                cmd.execute(config).unwrap();
            }
            CliCommand::Gt(gt) => {
                let cmd = gt::GtCommand {
                    args: CommonCommands {
                        db_file_path: gt.args.db_file_path.clone(),
                        table_id: gt.args.table_id.clone(),
                        value: gt.args.value.clone(),
                        output_file: gt.args.output_file.clone(),
                        silent: gt.args.silent,
                    },
                };
                cmd.execute(config).unwrap();
            }
            CliCommand::Gte(gte) => {
                let cmd = gte::GteCommand {
                    args: CommonCommands {
                        db_file_path: gte.args.db_file_path.clone(),
                        table_id: gte.args.table_id.clone(),
                        value: gte.args.value.clone(),
                        output_file: gte.args.output_file.clone(),
                        silent: gte.args.silent,
                    },
                };
                cmd.execute(config).unwrap();
            }
            CliCommand::Lt(lt) => {
                let cmd = lt::LtCommand {
                    args: CommonCommands {
                        db_file_path: lt.args.db_file_path.clone(),
                        table_id: lt.args.table_id.clone(),
                        value: lt.args.value.clone(),
                        output_file: lt.args.output_file.clone(),
                        silent: lt.args.silent,
                    },
                };
                cmd.execute(config).unwrap();
            }
            CliCommand::Lte(lte) => {
                let cmd = lte::LteCommand {
                    args: CommonCommands {
                        db_file_path: lte.args.db_file_path.clone(),
                        table_id: lte.args.table_id.clone(),
                        value: lte.args.value.clone(),
                        output_file: lte.args.output_file.clone(),
                        silent: lte.args.silent,
                    },
                };
                cmd.execute(config).unwrap();
            }
        }
        cli
    }
}
