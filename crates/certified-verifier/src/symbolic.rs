//! Encoders for the symbolic-DAG sub-tree of the verifying key.
//!
//! Constructor tags and field order mirror the corresponding types in the
//! upstream Lean `Wire.Raw` decoder. `Entry::Challenge` is rejected because
//! the Lean wire type has no matching constructor.

use std::io::{Error, ErrorKind, Result, Write};

use openvm_stark_backend::{
    air_builders::symbolic::{
        symbolic_variable::{Entry, SymbolicVariable},
        SymbolicConstraintsDag, SymbolicExpressionDag, SymbolicExpressionNode,
    },
    interaction::Interaction,
    p3_field::PrimeField32,
};

use super::primitives::{write_length_prefix, write_prime_field32, write_u32, write_usize_as_u32};

/// Encode a symbolic-variable source using the Lean constructor tags.
fn write_entry<W: Write>(writer: &mut W, entry: &Entry) -> Result<()> {
    match entry {
        Entry::Preprocessed { offset } => {
            writer.write_all(&[0x00u8])?;
            write_usize_as_u32(writer, *offset)
        }
        Entry::Main { part_index, offset } => {
            writer.write_all(&[0x01u8])?;
            write_usize_as_u32(writer, *part_index)?;
            write_usize_as_u32(writer, *offset)
        }
        Entry::Public => writer.write_all(&[0x02u8]),
        Entry::Challenge => Err(Error::new(
            ErrorKind::InvalidData,
            "wire format v1 does not support Entry::Challenge",
        )),
    }
}

/// Encode a symbolic-variable source and column index.
pub(crate) fn write_symbolic_variable<F, W: Write>(
    writer: &mut W,
    variable: &SymbolicVariable<F>,
) -> Result<()> {
    write_entry(writer, &variable.entry)?;
    write_usize_as_u32(writer, variable.index)
}

/// Encode one symbolic-expression node using the Lean constructor tags.
fn write_symbolic_expression_node<F: PrimeField32, W: Write>(
    writer: &mut W,
    node: &SymbolicExpressionNode<F>,
) -> Result<()> {
    match node {
        SymbolicExpressionNode::Variable(v) => {
            writer.write_all(&[0x00u8])?;
            write_symbolic_variable(writer, v)
        }
        SymbolicExpressionNode::IsFirstRow => writer.write_all(&[0x01u8]),
        SymbolicExpressionNode::IsLastRow => writer.write_all(&[0x02u8]),
        SymbolicExpressionNode::IsTransition => writer.write_all(&[0x03u8]),
        SymbolicExpressionNode::Constant(c) => {
            writer.write_all(&[0x04u8])?;
            write_prime_field32(writer, c)
        }
        SymbolicExpressionNode::Add {
            left_idx,
            right_idx,
            degree_multiple: _,
        } => {
            writer.write_all(&[0x05u8])?;
            write_usize_as_u32(writer, *left_idx)?;
            write_usize_as_u32(writer, *right_idx)
        }
        SymbolicExpressionNode::Sub {
            left_idx,
            right_idx,
            degree_multiple: _,
        } => {
            writer.write_all(&[0x06u8])?;
            write_usize_as_u32(writer, *left_idx)?;
            write_usize_as_u32(writer, *right_idx)
        }
        SymbolicExpressionNode::Neg {
            idx,
            degree_multiple: _,
        } => {
            writer.write_all(&[0x07u8])?;
            write_usize_as_u32(writer, *idx)
        }
        SymbolicExpressionNode::Mul {
            left_idx,
            right_idx,
            degree_multiple: _,
        } => {
            writer.write_all(&[0x08u8])?;
            write_usize_as_u32(writer, *left_idx)?;
            write_usize_as_u32(writer, *right_idx)
        }
    }
}

/// Encode the expression nodes followed by the constraint-root indices.
fn write_symbolic_expression_dag<F: PrimeField32, W: Write>(
    writer: &mut W,
    dag: &SymbolicExpressionDag<F>,
) -> Result<()> {
    write_length_prefix(writer, dag.nodes.len())?;
    for node in &dag.nodes {
        write_symbolic_expression_node(writer, node)?;
    }
    write_length_prefix(writer, dag.constraint_idx.len())?;
    for idx in &dag.constraint_idx {
        write_usize_as_u32(writer, *idx)?;
    }
    Ok(())
}

/// Encode one symbolic interaction.
///
/// `Interaction<usize>` uses node indices for `message` and `count`, which
/// correspond to `List Nat` and `Nat` on the Lean side.
fn write_symbolic_interaction<W: Write>(
    writer: &mut W,
    interaction: &Interaction<usize>,
) -> Result<()> {
    write_length_prefix(writer, interaction.message.len())?;
    for idx in &interaction.message {
        write_usize_as_u32(writer, *idx)?;
    }
    write_usize_as_u32(writer, interaction.count)?;
    write_u32(writer, interaction.bus_index as u32)?;
    write_u32(writer, interaction.count_weight)
}

/// Encode a complete symbolic constraint DAG.
///
/// The encoder takes the resolved `width` and `public_value_count`
/// explicitly because the Rust source struct does not carry those
/// fields; the caller reads them from the parent `StarkVerifyingParams`
/// (`params.width.total_width()` and `params.num_public_values`) which
/// is also where the Lean `hLayout` / `hPublicValues` invariants source
/// their truth.
pub(crate) fn write_symbolic_constraints_dag<F: PrimeField32, W: Write>(
    writer: &mut W,
    width: usize,
    public_value_count: usize,
    dag: &SymbolicConstraintsDag<F>,
) -> Result<()> {
    write_usize_as_u32(writer, width)?;
    write_usize_as_u32(writer, public_value_count)?;
    write_symbolic_expression_dag(writer, &dag.constraints)?;
    write_length_prefix(writer, dag.interactions.len())?;
    for interaction in &dag.interactions {
        write_symbolic_interaction(writer, interaction)?;
    }
    Ok(())
}
