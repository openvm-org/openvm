use crate::cpu::{trace::FieldExtensionOperation, OpCode};
use p3_field::Field;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Field arithmetic chip.
///
/// Carries information about opcodes (currently 5..=8) and bus index (currently 2).
#[derive(Default, Clone, Copy)]
pub struct FieldExtensionAddSubAir {}

impl FieldExtensionAddSubAir {
    pub const BUS_INDEX: usize = 2;

    pub fn new() -> Self {
        Self {}
    }

    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for non-arithmetic operations.
    pub fn solve<T: Field>(op: OpCode, operands: ([T; 4], [T; 4])) -> Option<[T; 4]> {
        match op {
            OpCode::FEADD => Some([
                operands.0[0] + operands.1[0],
                operands.0[1] + operands.1[1],
                operands.0[2] + operands.1[2],
                operands.0[3] + operands.1[3],
            ]),
            OpCode::FESUB => Some([
                operands.0[0] - operands.1[0],
                operands.0[1] - operands.1[1],
                operands.0[2] - operands.1[2],
                operands.0[3] - operands.1[3],
            ]),
            _ => None,
        }
    }

    /// Vectorized solve<>
    pub fn solve_all<T: Field>(ops: Vec<OpCode>, operands: Vec<([T; 4], [T; 4])>) -> Vec<[T; 4]> {
        ops.iter()
            .zip(operands.iter())
            .filter_map(|(op, operand)| Self::solve::<T>(*op, *operand))
            .collect()
    }

    /// Converts vectorized opcodes and operands into vectorized ArithmeticOperations.
    pub fn request<T: Field>(
        ops: Vec<OpCode>,
        operands: Vec<([T; 4], [T; 4])>,
    ) -> Vec<FieldExtensionOperation<T>> {
        ops.iter()
            .zip(operands.iter())
            .map(|(op, operand)| FieldExtensionOperation {
                opcode: *op,
                operand1: operand.0,
                operand2: operand.1,
                result: Self::solve::<T>(*op, *operand).unwrap(),
            })
            .collect()
    }
}
