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
    pub opcode: Opcode,
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
}

impl<F: PrimeField32> FieldArithmeticChip<F> {
    #[allow(clippy::new_without_default)]
    pub fn new(execution_bus: ExecutionBus, memory_chip: MemoryChipRef<F>) -> Self {
        let mem_oc = memory_chip.borrow().make_offline_checker();
        Self {
            air: FieldArithmeticAir {
                execution_bus,
                mem_oc,
            },
            records: vec![],
            memory_chip,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for FieldArithmeticChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
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
        } = instruction;
        assert!(FIELD_ARITHMETIC_INSTRUCTIONS.contains(&opcode));

        let mut memory_chip = self.memory_chip.borrow_mut();

        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let x_read = memory_chip.read_cell(x_as, x_address);
        let y_read = memory_chip.read_cell(y_as, y_address);

        let x = x_read.value();
        let y = y_read.value();
        let z = FieldArithmetic::solve(opcode, (x, y)).unwrap();

        let z_write = memory_chip.write_cell(z_as, z_address, z);

        self.records.push(FieldArithmeticRecord {
            opcode,
            from_state,
            x_read,
            y_read,
            z_write,
        });
        tracing::trace!("op = {:?}", self.records.last().unwrap());

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
