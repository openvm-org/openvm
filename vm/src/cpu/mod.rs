use core::panic;

use afs_primitives::xor::bus::XorBus;
pub use air::CpuAir;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bus::ExecutionBus,
        instructions::{Opcode, Opcode::*},
    },
    memory::MemoryChipRef,
};
// TODO[zach]: Restore tests once we have control flow chip.
//#[cfg(test)]
//pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const INST_WIDTH: usize = 1;

pub const READ_INSTRUCTION_BUS: usize = 8;
pub const RANGE_CHECKER_BUS: usize = 4;
pub const POSEIDON2_DIRECT_BUS: usize = 6;
pub const BYTE_XOR_BUS: XorBus = XorBus(8);
pub const RANGE_TUPLE_CHECKER_BUS: usize = 11;
pub const CPU_MAX_READS_PER_CYCLE: usize = 3;
pub const CPU_MAX_WRITES_PER_CYCLE: usize = 1;
pub const CPU_MAX_ACCESSES_PER_CYCLE: usize = CPU_MAX_READS_PER_CYCLE + CPU_MAX_WRITES_PER_CYCLE;

// [jpw] Temporary, we are going to remove cpu anyways
const WORD_SIZE: usize = 1;

fn timestamp_delta(opcode: Opcode) -> usize {
    match opcode {
        LOADW | STOREW => 3,
        LOADW2 | STOREW2 => 4,
        JAL => 1,
        BEQ | BNE => 2,
        TERMINATE => 0,
        PUBLISH => 2,
        FAIL => 0,
        PRINTF => 1,
        SHINTW => 2,
        HINT_INPUT | HINT_BITS | HINT_BYTES => 0,
        CT_START | CT_END => 0,
        NOP => 0,
        _ => panic!("Non-CPU opcode: {:?}", opcode),
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct CpuOptions {
    pub num_public_values: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CpuState {
    pub clock_cycle: usize,
    pub timestamp: usize,
    pub pc: usize,
    pub is_done: bool,
}

impl CpuState {
    pub fn initial() -> Self {
        CpuState {
            clock_cycle: 0,
            timestamp: 1,
            pc: 0,
            is_done: false,
        }
    }
}

/// Chip for the CPU. Carries all state and owns execution.
#[derive(Debug)]
pub struct CpuChip<F: PrimeField32> {
    pub air: CpuAir,
    pub rows: Vec<Vec<F>>,
    pub state: CpuState,
    /// Program counter at the start of the current segment.
    pub start_state: CpuState,
    pub public_values: Vec<Option<F>>,
    pub memory_chip: MemoryChipRef<F>,
}

impl<F: PrimeField32> CpuChip<F> {
    pub fn new(
        options: CpuOptions,
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<F>,
    ) -> Self {
        Self::from_state(options, execution_bus, memory_chip, CpuState::initial())
    }

    /// Sets the current state of the CPU.
    pub fn set_state(&mut self, state: CpuState) {
        self.state = state;
    }

    /// Sets the current state of the CPU.
    pub fn from_state(
        options: CpuOptions,
        execution_bus: ExecutionBus,
        memory_chip: MemoryChipRef<F>,
        state: CpuState,
    ) -> Self {
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            air: CpuAir {
                options,
                execution_bus,
                memory_bridge,
            },
            rows: vec![],
            state,
            start_state: state,
            public_values: vec![None; options.num_public_values],
            memory_chip,
        }
    }
}
