use clap::Args;
use eyre::Result;
use tracing::info;

use crate::circuit_printer::CircuitPrinter;

#[derive(Args)]
pub struct PrintCircuitArgs {
    /// The type of circuit to print
    #[arg(long, short, value_enum)]
    circuit_type: CircuitType,

    /// Output format for the circuit
    #[arg(long, short, default_value = "text")]
    format: OutputFormat,

    /// Output file path (optional, defaults to stdout)
    #[arg(long, short)]
    output: Option<String>,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum CircuitType {
    Poseidon2,
    Sha256,
    Keccak256,
    Primitives,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Rocq,
}

pub fn print_circuit(args: PrintCircuitArgs) -> Result<()> {
    info!(
        "Printing circuit: {:?} in {:?} format",
        args.circuit_type, args.format
    );

    let printer = CircuitPrinter::new();

    match args.circuit_type {
        CircuitType::Poseidon2 => {
            printer.print_poseidon2_circuit(args.format, args.output.as_deref())?;
        }
        CircuitType::Sha256 => {
            printer.print_sha256_circuit(args.format, args.output.as_deref())?;
        }
        CircuitType::Keccak256 => {
            printer.print_keccak256_circuit(args.format, args.output.as_deref())?;
        }
        CircuitType::Primitives => {
            printer.print_primitives_circuit(args.format, args.output.as_deref())?;
        }
    }

    Ok(())
}
