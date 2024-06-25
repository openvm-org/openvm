use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use clap::Subcommand;
use p3_uni_stark::StarkGenericConfig;

use crate::commands::run;

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS CLI")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    #[command(name = "run", about = "Run OLAP operations")]
    /// Run OLAP operations
    Run(run::RunCommand),
}

impl Cli {
    pub fn run<SC: StarkGenericConfig>(config: &PageConfig) -> Self {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Run(run) => {
                run.execute::<SC>(config).unwrap();
            }
        }
        cli
    }
}
