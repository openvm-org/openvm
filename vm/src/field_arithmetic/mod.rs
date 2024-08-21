use std::{cell::RefCell, rc::Rc};

use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        bridge::ExecutionBus,
        chips::OpCodeExecutor,
        columns::{ExecutionState, InstructionCols},
        instructions::OpCode,
    },
    cpu::trace::Instruction,
    memory::offline_checker::MemoryChip,
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

/// Field arithmetic chip.

#[derive(Clone, Debug, PartialEq)]
pub struct ArithmeticOperation<F> {
    pub prev_state: ExecutionState<usize>,
    pub instruction: InstructionCols<F>,
    pub operand1: F,
    pub operand2: F,
    pub result: F,
}

#[derive(Clone, Copy, Debug)]
pub struct FieldArithmeticAir {
    execution_bus: ExecutionBus,
}

impl FieldArithmeticAir {
    pub const BASE_OP: u8 = OpCode::FADD as u8;
    pub const BUS_INDEX: usize = 2;

    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for non-arithmetic operations.
    pub fn solve<F: Field>(op: OpCode, operands: (F, F)) -> F {
        match op {
            OpCode::FADD => operands.0 + operands.1,
            OpCode::FSUB => operands.0 - operands.1,
            OpCode::FMUL => operands.0 * operands.1,
            OpCode::FDIV => operands.0 / operands.1,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct FieldArithmeticChip<F: PrimeField32> {
    pub air: FieldArithmeticAir,
    pub operations: Vec<ArithmeticOperation<F>>,
    pub memory_chip: Rc<RefCell<MemoryChip<1, F>>>,
}

impl<F: PrimeField32> FieldArithmeticChip<F> {
    pub fn new(execution_bus: ExecutionBus, memory_chip: Rc<RefCell<MemoryChip<1, F>>>) -> Self {
        Self {
            air: FieldArithmeticAir { execution_bus },
            operations: vec![],
            memory_chip,
        }
    }

    pub fn current_height(&self) -> usize {
        self.operations.len()
    }
}

impl<F: PrimeField32> OpCodeExecutor<F> for FieldArithmeticChip<F> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        prev_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let start_timestamp = prev_state.timestamp;
        let operand1 = self.memory_chip.borrow_mut().read_elem(
            start_timestamp,
            instruction.d,
            instruction.op_b,
        );
        let operand2 = self.memory_chip.borrow_mut().read_elem(
            start_timestamp + 1,
            instruction.e,
            instruction.op_c,
        );
        let result = FieldArithmeticAir::solve::<F>(instruction.opcode, (operand1, operand2));
        self.memory_chip.borrow_mut().write_elem(
            start_timestamp + 2,
            instruction.d,
            instruction.op_a,
            result,
        );
        self.operations.push(ArithmeticOperation {
            prev_state,
            instruction: InstructionCols::from_instruction(instruction),
            operand1,
            operand2,
            result,
        });
        ExecutionState {
            pc: prev_state.pc + 1,
            timestamp: start_timestamp + 3,
        }
    }
}
