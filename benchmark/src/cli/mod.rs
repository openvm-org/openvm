use clap::{Parser, Subcommand};

use crate::commands::{olap::OlapCommand, rw::RwCommand};

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

    #[command(name = "olap", about = "Benchmark OLAP")]
    /// OLAP functions
    Olap(OlapCommand),
}

impl Cli {
    pub fn run() {
        let cli = Self::parse();
        match cli.command {
            Commands::Rw(rw) => rw.execute().unwrap(),
            Commands::Olap(olap) => olap.execute().unwrap(),
        }
    }
}
