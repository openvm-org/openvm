# OpenVM to-rocq Tool

A tool for pretty printing OpenVM AIR circuits in various formats.

## Overview

The `to-rocq` tool allows you to inspect and format OpenVM AIR circuits, including:
- Poseidon2 circuits
- SHA256 circuits  
- Keccak256 circuits
- General circuit primitives

## Installation

This crate is part of the OpenVM workspace. To build it:

```bash
cargo build --package openvm-to-rocq
```

## Usage

### Basic Usage

```bash
# Print a Poseidon2 circuit to stdout
cargo run --package openvm-to-rocq -- print --circuit-type poseidon2

# Print a SHA256 circuit in JSON format
cargo run --package openvm-to-rocq -- print --circuit-type sha256 --format json

# Print a Keccak256 circuit and save to file
cargo run --package openvm-to-rocq -- print --circuit-type keccak256 --output keccak_circuit.txt

# Print primitives circuit in Rocq format
cargo run --package openvm-to-rocq -- print --circuit-type primitives --format rocq
```

### Command Line Options

- `--circuit-type, -c`: The type of circuit to print (poseidon2, sha256, keccak256, primitives)
- `--format, -f`: Output format (text, json, rocq) [default: text]
- `--output, -o`: Output file path (optional, defaults to stdout)

### Output Formats

1. **Text**: Human-readable plain text format
2. **JSON**: Structured JSON format for programmatic use
3. **Rocq**: Rocq-compatible circuit format (placeholder implementation)

## Circuit Types

### Poseidon2
- Field: BabyBear
- Rounds: 8
- Width: 3
- Based on Plonky3 poseidon2-air

### SHA256
- Block size: 512 bits
- Rounds: 64
- Custom AIR implementation

### Keccak256
- Block size: 1088 bits
- Rounds: 24
- Based on Plonky3 keccak-air

### Primitives
- Field arithmetic
- Boolean logic
- Range checks
- Lookup tables

## Development

This tool is designed to be extensible. To add support for new circuit types:

1. Add the new circuit type to the `CircuitType` enum in `commands.rs`
2. Implement the corresponding print method in `CircuitPrinter`
3. Add the circuit information retrieval method

## Dependencies

- OpenVM circuit crates for accessing AIR circuits
- Plonky3 dependencies for circuit inspection
- Standard Rust libraries for CLI, JSON, and file I/O
