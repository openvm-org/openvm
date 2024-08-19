use itertools::Itertools;
use p3_field::Field;

use crate::cpu::{trace::isize_to_field, OpCode};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Field arithmetic chip.
///
/// Carries information about opcodes (currently 6..=9) and bus index (currently 2).

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ArithmeticOperation<F> {
    pub opcode: OpCode,
    pub operand1: F,
    pub operand2: F,
    pub result: F,
}

impl<F: Field> ArithmeticOperation<F> {
    pub fn from_isize(opcode: OpCode, operand1: isize, operand2: isize, result: isize) -> Self {
        Self {
            opcode,
            operand1: isize_to_field::<F>(operand1),
            operand2: isize_to_field::<F>(operand2),
            result: isize_to_field::<F>(result),
        }
    }

    pub fn to_vec(&self) -> Vec<F> {
        vec![
            F::from_canonical_usize(self.opcode as usize),
            self.operand1,
            self.operand2,
            self.result,
        ]
    }
}

#[derive(Default, Clone, Copy)]
pub struct FieldArithmeticAir {}

impl FieldArithmeticAir {
    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for non-arithmetic operations.
    fn solve<T: Field>(op: OpCode, operands: (T, T)) -> Option<T> {
        match op {
            OpCode::FADD => Some(operands.0 + operands.1),
            OpCode::FSUB => Some(operands.0 - operands.1),
            OpCode::FMUL => Some(operands.0 * operands.1),
            OpCode::FDIV => {
                if operands.1 == T::zero() {
                    None
                } else {
                    Some(operands.0 / operands.1)
                }
            }
            _ => unreachable!(),
        }
    }
}

pub struct FieldArithmeticChip<F: Field> {
    pub air: FieldArithmeticAir,
    pub operations: Vec<ArithmeticOperation<F>>,
}

impl<F: Field> FieldArithmeticChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            air: FieldArithmeticAir {},
            operations: vec![],
        }
    }

    pub fn calculate(&mut self, op: OpCode, operands: (F, F)) -> F {
        let result = FieldArithmeticAir::solve::<F>(op, operands).unwrap();
        self.operations.push(ArithmeticOperation {
            opcode: op,
            operand1: operands.0,
            operand2: operands.1,
            result,
        });
        result
    }

    pub fn request(&mut self, ops: Vec<OpCode>, operands_vec: Vec<(F, F)>) {
        for (op, operands) in ops.iter().zip_eq(operands_vec.iter()) {
            self.calculate(*op, *operands);
        }
    }

    pub fn current_height(&self) -> usize {
        self.operations.len()
    }
}
