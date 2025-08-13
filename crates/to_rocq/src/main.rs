use clap::{Parser, Subcommand};
use eyre::Result;
use tracing::info;

mod circuit_printer;
mod commands;

use commands::{print_circuit, PrintCircuitArgs};

#[derive(Parser)]
#[command(name = "to-rocq")]
#[command(about = "Tool for pretty printing OpenVM AIR circuits")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Print a circuit in a readable format
    Print(PrintCircuitArgs),
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting to-rocq tool...");

    let cli = Cli::parse();

    match cli.command {
        Commands::Print(args) => {
            print_circuit(args)?;
        }
    }

    Ok(())
}
