use p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        instructions::{FieldArithmeticOpcode, UsizeOpcode},
        ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor,
    },
    field_arithmetic::columns::Operand,
    program::{bridge::ProgramBus, ExecutionError, Instruction},
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::FieldArithmeticAir;

use crate::memory::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord};

#[derive(Clone, Debug)]
pub struct FieldArithmeticRecord<F> {
    pub opcode: usize,
    pub from_state: ExecutionState<usize>,
    pub x_read: MemoryReadRecord<F, 1>,
    pub y_read: MemoryReadRecord<F, 1>,
    pub z_write: MemoryWriteRecord<F, 1>,
}

#[derive(Clone, Debug)]
pub struct FieldArithmeticChip<F: PrimeField32> {
    pub air: FieldArithmeticAir,
    pub records: Vec<FieldArithmeticRecord<F>>,

    pub memory_chip: MemoryChipRef<F>,

    offset: usize,
}

impl<F: PrimeField32> FieldArithmeticChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<F>,
        offset: usize,
    ) -> Self {
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            air: FieldArithmeticAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                offset,
            },
            records: vec![],
            memory_chip,
            offset,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for FieldArithmeticChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<usize>,
    ) -> Result<ExecutionState<usize>, ExecutionError> {
        let Instruction {
            opcode,
            op_a: z_address,
            op_b: x_address,
            op_c: y_address,
            d: z_as,
            e: x_as,
            op_f: y_as,
            ..
        } = instruction;
        let opcode = opcode - self.offset;

        let mut memory_chip = self.memory_chip.borrow_mut();

        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let x_read = memory_chip.read_cell(x_as, x_address);
        let y_read = memory_chip.read_cell(y_as, y_address);

        let x = x_read.value();
        let y = y_read.value();
        let z = FieldArithmetic::solve(FieldArithmeticOpcode::from_usize(opcode), (x, y)).unwrap();

        let z_write = memory_chip.write_cell(z_as, z_address, z);

        self.records.push(FieldArithmeticRecord {
            opcode,
            from_state,
            x_read,
            y_read,
            z_write,
        });
        tracing::trace!("op = {:?}", self.records.last().unwrap());

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: from_state.timestamp + FieldArithmeticAir::TIMESTAMP_DELTA,
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            FieldArithmeticOpcode::from_usize(opcode - self.offset)
        )
    }
}

pub struct FieldArithmetic;
impl FieldArithmetic {
    /// Evaluates given opcode using given operands.
    ///
    /// Returns None for non-arithmetic operations.
    fn solve<T: Field>(op: FieldArithmeticOpcode, operands: (T, T)) -> Option<T> {
        match op {
            FieldArithmeticOpcode::ADD => Some(operands.0 + operands.1),
            FieldArithmeticOpcode::SUB => Some(operands.0 - operands.1),
            FieldArithmeticOpcode::MUL => Some(operands.0 * operands.1),
            FieldArithmeticOpcode::DIV => {
                if operands.1 == T::zero() {
                    None
                } else {
                    Some(operands.0 / operands.1)
                }
            }
        }
    }
}
