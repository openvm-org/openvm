//! OpenVM to-rocq tool for pretty printing AIR circuits
//!
//! This crate provides functionality to inspect and format OpenVM AIR circuits
//! in various output formats including text, JSON, and Rocq-compatible formats.

pub mod circuit_printer;
pub mod commands;

pub use circuit_printer::CircuitPrinter;
pub use commands::{print_circuit, CircuitType, OutputFormat, PrintCircuitArgs};
