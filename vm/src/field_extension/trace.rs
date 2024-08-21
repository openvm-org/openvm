use std::{array, vec::IntoIter};

use itertools::Itertools;
use p3_field::{Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    columns::{
        FieldExtensionArithmeticAuxCols, FieldExtensionArithmeticCols,
        FieldExtensionArithmeticIoCols,
    },
    FieldExtensionArithmetic, FieldExtensionArithmeticChip, FieldExtensionArithmeticOperation,
};
use crate::{cpu::OpCode, memory::offline_checker::columns::MemoryOfflineCheckerAuxCols};

/// Constructs a new set of columns (including auxiliary columns) given inputs.
fn generate_cols<const WORD_SIZE: usize, T: Field>(
    op: FieldExtensionArithmeticOperation<WORD_SIZE, T>,
    oc_aux_iter: &mut IntoIter<MemoryOfflineCheckerAuxCols<WORD_SIZE, T>>,
) -> FieldExtensionArithmeticCols<WORD_SIZE, T> {
    let is_add = T::from_bool(op.opcode == OpCode::FE4ADD);
    let is_sub = T::from_bool(op.opcode == OpCode::FE4SUB);
    let is_mul = T::from_bool(op.opcode == OpCode::BBE4MUL);
    let is_inv = T::from_bool(op.opcode == OpCode::BBE4INV);

    let x = op.operand1;
    let y = op.operand2;

    let inv = if x[0] == T::zero() && x[1] == T::zero() && x[2] == T::zero() && x[3] == T::zero() {
        [T::zero(), T::zero(), T::zero(), T::zero()]
    } else {
        FieldExtensionArithmetic::solve(OpCode::BBE4INV, x, y).unwrap()
    };

    FieldExtensionArithmeticCols {
        io: FieldExtensionArithmeticIoCols {
            opcode: T::from_canonical_usize(op.opcode as usize),
            clk: T::from_canonical_usize(op.clk),
            x,
            y,
            z: op.result,
        },
        aux: FieldExtensionArithmeticAuxCols {
            is_valid: T::one(),
            valid_y_read: T::one() - is_inv,
            op_a: op.op_a,
            op_b: op.op_b,
            op_c: op.op_c,
            d: op.d,
            e: op.e,
            is_add,
            is_sub,
            is_mul,
            is_inv,
            inv,
            mem_oc_aux_cols: array::from_fn(|_| oc_aux_iter.next().unwrap()),
        },
    }
}

impl<const NUM_WORDS: usize, const WORD_SIZE: usize, F: PrimeField32>
    FieldExtensionArithmeticChip<NUM_WORDS, WORD_SIZE, F>
{
    /// Generates trace for field arithmetic chip.
    pub fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        // todo[zach]: it's weird that `generate_trace` mutates the receiver
        let accesses = self.memory.take_accesses_buffer();
        let mut accesses_iter = accesses.into_iter();

        let mut trace: Vec<F> = self
            .operations
            .iter()
            .cloned()
            .flat_map(|op| generate_cols::<WORD_SIZE, F>(op, &mut accesses_iter).flatten())
            .collect();

        assert!(accesses_iter.next().is_none());

        let curr_height = self.operations.len();
        let correct_height = curr_height.next_power_of_two();

        let width = FieldExtensionArithmeticCols::<WORD_SIZE, F>::get_width(&self.air);
        trace.extend(
            (0..correct_height - curr_height)
                .flat_map(|_| self.make_blank_row().flatten())
                .collect_vec(),
        );

        RowMajorMatrix::new(trace, width)
    }
}
