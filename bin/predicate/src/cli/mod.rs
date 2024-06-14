use clap::Parser;
use clap::Subcommand;

use crate::commands::eq;

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
    // #[command(name = "gt", about = "Get data in a Table that is greater than the input")]
    // /// Get data in a Table that is greater than the input
    // Gt(gt::GtCommand),

    // #[command(name = "gte", about = "Get data in a Table that is greater than or equal to the input")]
    // /// Get data in a Table that is greater than or equal to the input
    // Gte(gte::GteCommand),

    // #[command(name = "lt", about = "Get data in a Table that is less than the input")]
    // /// Get data in a Table that is less than the input
    // Lt(lt::LtCommand),

    // #[command(name = "lte", about = "Get data in a Table that is less than or equal to the input")]
    // /// Get data in a Table that is less than or equal to the input
    // Lte(lte::LteCommand),
}

impl Cli {
    pub fn run() -> Self {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Eq(eq) => {
                let cmd = eq::EqCommand {
                    table_id: eq.table_id.clone(),
                    value: eq.value.clone(),
                    output_file: eq.output_file.clone(),
                };
                cmd.execute().unwrap();
            } // CliCommand::Gt(gt) => {
              //     gt.execute().unwrap();
              // }
              // CliCommand::Gte(gte) => {
              //     gte.execute().unwrap();
              // }
              // CliCommand::Lt(lt) => {
              //     lt.execute().unwrap();
              // }
              // CliCommand::Lte(lte) => {
              //     lte.execute().unwrap();
              // }
        }
        cli
    }
}
