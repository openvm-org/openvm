use std::sync::Arc;

use afs_primitives::var_range::VariableRangeCheckerChip;
use p3_field::PrimeField32;

use crate::{
    arch::{
        instructions::Opcode, ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor,
    },
    memory::{MemoryChipRef, MemoryReadRecord, MemoryWriteRecord},
    program::{bridge::ProgramBus, ExecutionError, Instruction},
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

    pub x_write: MemoryWriteRecord<T, 4>,
    pub y_read: MemoryReadRecord<T, 1>,
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
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<T>,
    ) -> Self {
        let range_checker_chip = memory_chip.borrow().range_checker.clone();
        let memory_bridge = memory_chip.borrow().memory_bridge();
        let execution_bridge = ExecutionBridge::new(execution_bus, program_bus);
        let bus = range_checker_chip.bus();

        assert!(
            bus.range_max_bits >= LIMB_SIZE,
            "range_max_bits {} < LIMB_SIZE {}",
            bus.range_max_bits,
            LIMB_SIZE
        );
        Self {
            air: CastFAir {
                execution_bridge,
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
    ) -> Result<ExecutionState<usize>, ExecutionError> {
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

        let y_read = memory_chip.read_cell(e, b);
        let y = y_read.data[0].as_canonical_u32();

        let x = Self::solve(y);
        for (i, limb) in x.iter().enumerate() {
            if i == 3 {
                self.range_checker_chip.add_count(*limb, FINAL_LIMB_SIZE);
            } else {
                self.range_checker_chip.add_count(*limb, LIMB_SIZE);
            }
        }

        let x = x.map(T::from_canonical_u32);
        let x_write = memory_chip.write::<4>(d, a, x);

        self.data.push(CastFRecord {
            from_state,
            instruction: instruction.clone(),
            x_write,
            y_read,
        });

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: memory_chip.timestamp().as_canonical_u32() as usize,
        })
    }
}
impl<T: PrimeField32> CastFChip<T> {
    fn solve(y: u32) -> [u32; 4] {
        let mut x = [0; 4];
        for (i, limb) in x.iter_mut().enumerate() {
            *limb = (y >> (8 * i)) & 0xFF;
        }
        x
    }
}
