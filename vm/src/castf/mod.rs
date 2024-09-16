use std::sync::Arc;

use afs_primitives::var_range::VariableRangeCheckerChip;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus, chips::InstructionExecutor, columns::ExecutionState,
        instructions::Opcode,
    },
    cpu::trace::Instruction,
    memory::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
};

#[cfg(test)]
pub mod tests;

mod air;
mod bridge;
mod columns;
mod trace;

pub use air::*;
pub use columns::*;

#[derive(Debug)]
pub struct CastFRecord<T> {
    pub from_state: ExecutionState<usize>,
    pub instruction: Instruction<T>,

    pub x_read: MemoryReadRecord<T, 4>,
    pub y_write: MemoryWriteRecord<T, 1>,
}

#[derive(Debug)]
pub struct CastFChip<T: PrimeField32> {
    pub air: CastFAir,
    data: Vec<CastFRecord<T>>,
    memory_chip: MemoryChipRef<T>,
    pub range_checker_chip: Arc<VariableRangeCheckerChip>,
}

impl<T: PrimeField32> CastFChip<T> {
    pub fn new(
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<T>,
        range_checker_chip: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        let bus = range_checker_chip.bus();
        let memory_bridge = memory_chip.borrow().memory_bridge();

        assert!(
            bus.range_max_bits >= LIMB_SIZE,
            "range_max_bits {} < LIMB_SIZE {}",
            bus.range_max_bits,
            LIMB_SIZE
        );
        Self {
            air: CastFAir {
                execution_bus,
                memory_bridge,
                bus,
            },
            data: vec![],
            memory_chip,
            range_checker_chip,
        }
    }
}

impl<T: PrimeField32> InstructionExecutor<T> for CastFChip<T> {
    fn execute(
        &mut self,
        instruction: Instruction<T>,
        from_state: ExecutionState<usize>,
    ) -> ExecutionState<usize> {
        let Instruction {
            opcode,
            op_a: a,
            op_b: b,
            d,
            e,
            ..
        } = instruction.clone();
        assert_eq!(opcode, Opcode::CASTF);

        let mut memory_chip = self.memory_chip.borrow_mut();

        debug_assert_eq!(
            from_state.timestamp,
            memory_chip.timestamp().as_canonical_u32() as usize
        );
        let x_read = memory_chip.read::<4>(d, a);
        let x = x_read.data.map(|x| x.as_canonical_u32());

        let y = Self::solve(&x);
        for (i, limb) in x.iter().enumerate() {
            if i == 3 {
                self.range_checker_chip.add_count(*limb, FINAL_LIMB_SIZE);
            } else {
                self.range_checker_chip.add_count(*limb, LIMB_SIZE);
            }
        }

        let y_write = memory_chip.write_cell(e, b, T::from_canonical_u32(y));

        self.data.push(CastFRecord {
            from_state,
            instruction: instruction.clone(),
            x_read,
            y_write,
        });

        ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        }
    }
}
impl<T: PrimeField32> CastFChip<T> {
    fn solve(x: &[u32; 4]) -> u32 {
        let mut y = 0;
        for (i, limb) in x.iter().enumerate() {
            y += limb << (LIMB_SIZE * i);
        }
        y
    }
}
