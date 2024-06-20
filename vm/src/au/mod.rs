use super::cpu::trace::ArithmeticOperation;
use crate::cpu::OpCode;
use p3_field::Field;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Clone, Copy)]
pub struct FieldArithmeticAir {}

impl FieldArithmeticAir {
    pub const BASE_OP: u8 = 5;
    pub const BUS_INDEX: usize = 2;

    pub fn new() -> Self {
        Self {}
    }

    pub fn solve<T: Field>(op: OpCode, operands: (T, T)) -> Option<T> {
        match op {
            OpCode::LOADW => None,
            OpCode::STOREW => None,
            OpCode::JAL => None,
            OpCode::BEQ => None,
            OpCode::BNE => None,

            OpCode::FADD => Some(operands.0 + operands.1),
            OpCode::FSUB => Some(operands.0 - operands.1),
            OpCode::FMUL => Some(operands.0 * operands.1),
            OpCode::FDIV => Some(operands.0 / operands.1),
        }
    }

    pub fn solve_all<T: Field>(ops: Vec<OpCode>, operands: Vec<(T, T)>) -> Vec<T> {
        ops.iter()
            .zip(operands.iter())
            .filter_map(|(op, operand)| Self::solve::<T>(*op, *operand))
            .collect()
    }

    pub fn request<T: Field>(
        ops: Vec<OpCode>,
        operands: Vec<(T, T)>,
    ) -> Vec<ArithmeticOperation<T>> {
        ops.iter()
            .zip(operands.iter())
            .map(|(op, operand)| ArithmeticOperation {
                opcode: *op,
                operand1: operand.0,
                operand2: operand.1,
                result: Self::solve::<T>(*op, *operand).unwrap(),
            })
            .collect()
    }
}
