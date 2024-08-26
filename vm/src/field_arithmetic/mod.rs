use std::{cell::RefCell, rc::Rc};

use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, FIELD_ARITHMETIC_INSTRUCTIONS},
    },
    cpu::trace::Instruction,
    field_arithmetic::columns::Operand,
    memory::manager::{trace_builder::MemoryTraceBuilder, MemoryManager},
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::FieldArithmeticAir;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FieldArithmeticOperation<F> {
    pub opcode: Opcode,
    pub from_state: ExecutionState<usize>,
    pub operand1: Operand<F>,
    pub operand2: Operand<F>,
    pub result: Operand<F>,
}

#[derive(Clone, Debug)]
pub struct FieldArithmeticChip<F: PrimeField32> {
    pub air: FieldArithmeticAir,
    pub operations: Vec<FieldArithmeticOperation<F>>,

    pub memory_manager: Rc<RefCell<MemoryManager<F>>>,
    pub memory: MemoryTraceBuilder<F>,
}

impl<F: PrimeField32> FieldArithmeticChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new(execution_bus: ExecutionBus, memory_manager: Rc<RefCell<MemoryManager<F>>>) -> Self {
        let mem_oc = memory_manager.borrow().make_offline_checker();
        Self {
            air: FieldArithmeticAir {
                execution_bus,
                mem_oc,
            },
            operations: vec![],
            memory: MemoryTraceBuilder::new(memory_manager.clone()),
            memory_manager,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for FieldArithmeticChip<F> {
    fn execute(
        &mut self,
        instruction: &Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let Instruction {
            opcode,
            op_a: z_address,
            op_b: x_address,
            op_c: y_address,
            d: z_as,
            e: x_as,
            op_f: y_as,
            ..
        } = instruction.clone();
        assert!(FIELD_ARITHMETIC_INSTRUCTIONS.contains(&opcode));

        let x = self.memory.read_elem(x_as, x_address);
        let y = self.memory.read_elem(y_as, y_address);
        let z = FieldArithmetic::solve(opcode, (x, y)).unwrap();

        self.memory.write_elem(z_as, z_address, z);

        self.operations.push(FieldArithmeticOperation {
            opcode,
            from_state,
            operand1: Operand::new(x_as, x_address, x),
            operand2: Operand::new(y_as, y_address, y),
            result: Operand::new(z_as, z_address, z),
        });
        tracing::trace!("op = {:?}", self.operations.last().unwrap());

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: from_state.timestamp + FieldArithmeticAir::TIMESTAMP_DELTA,
        }
    }
}

pub struct FieldArithmetic;
impl FieldArithmetic {
    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for non-arithmetic operations.
    fn solve<T: Field>(op: Opcode, operands: (T, T)) -> Option<T> {
        match op {
            Opcode::FADD => Some(operands.0 + operands.1),
            Opcode::FSUB => Some(operands.0 - operands.1),
            Opcode::FMUL => Some(operands.0 * operands.1),
            Opcode::FDIV => {
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
