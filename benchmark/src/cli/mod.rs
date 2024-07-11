use clap::{Parser, Subcommand};

use crate::commands::{predicate::PredicateCommand, rw::RwCommand};

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS Benchmark")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(name = "rw", about = "Benchmark Read/Write")]
    /// Read/Write functions
    Rw(RwCommand),

    #[command(name = "predicate", about = "Benchmark Predicate")]
    /// Predicate functions
    Predicate(PredicateCommand),
}

impl Cli {
    pub fn run() {
        let cli = Self::parse();
        match cli.command {
            Commands::Rw(rw) => rw.execute().unwrap(),
            Commands::Predicate(predicate) => predicate.execute().unwrap(),
        }
    }
}
