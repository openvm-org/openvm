use std::sync::Arc;

use afs_primitives::var_range::VariableRangeCheckerChip;
use air::UiAir;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus,
        chips::InstructionExecutor,
        columns::ExecutionState,
        instructions::{Opcode, UI_32_INSTRUCTIONS},
    },
    cpu::trace::Instruction,
    memory::{MemoryChipRef, MemoryWriteRecord},
};

mod air;
mod bridge;
mod columns;
mod trace;

#[cfg(test)]
mod tests;
pub use air::*;
pub use columns::*;

#[derive(Debug)]
pub struct UiRecord<T> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,
    pub x_write: MemoryWriteRecord<T, 1>,
}

#[derive(Debug)]
pub struct UiChip<T: PrimeField32> {
    pub air: UiAir,
    data: Vec<UiRecord<T>>,
    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<T: PrimeField32> UiChip<T> {
    pub fn new(execution_bus: ExecutionBus, memory_chip: MemoryChipRef<T>) -> Self {
        let range_checker_chip = memory_chip.borrow().range_checker.clone();
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            air: UiAir {
                execution_bus,
                memory_bridge,
            },
            data: vec![],
            memory_chip,
            range_checker_chip,
        }
    }
}

impl<T: PrimeField32> InstructionExecutor<T> for UiChip<T> {
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let Instruction {
            opcode,
            op_a: a,
            op_b: b,
            ..
        } = instruction.clone();
        assert!(UI_32_INSTRUCTIONS.contains(&opcode));

        let mut memory_chip = self.memory_chip.borrow_mut();
        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );

        let b: u32 = b.as_canonical_u32();

        self.range_checker_chip.add_count(b, 20);
        let x = match opcode {
            Opcode::LUI => Self::solve_lui(b),
            Opcode::AUIPC => Self::solve_auipc(b),
            _ => unreachable!(),
        };
        let x = x.map(T::from_canonical_u32);
        let x_write = memory_chip.write::<1>(T::one(), a, x);

        self.data.push(UiRecord {
            from_state,
            instruction: instruction.clone(),
            x_write,
        });

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        }
    }
}

impl<T: PrimeField32> UiChip<T> {
    fn solve_lui(b: u32) -> [u32; 1] {
        [b << 12]
    }

    fn solve_auipc(b: u32) -> [u32; 1] {
        unimplemented!()
    }
}
