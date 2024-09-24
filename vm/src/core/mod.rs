use core::panic;
use std::collections::VecDeque;

use afs_primitives::xor::bus::XorBus;
pub use air::CoreAir;
use p3_field::PrimeField32;

use crate::{
    arch::{
        bridge::ExecutionBridge,
        bus::ExecutionBus,
        instructions::Opcode::{self, *},
    },
    memory::MemoryChipRef,
    program::bridge::ProgramBus,
};
// TODO[zach]: Restore tests once we have control flow chip.
//#[cfg(test)]
//pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod execute;
pub mod trace;

pub const INST_WIDTH: usize = 1;

pub const READ_INSTRUCTION_BUS: usize = 8;
pub const RANGE_CHECKER_BUS: usize = 4;
pub const POSEIDON2_DIRECT_BUS: usize = 6;
pub const BYTE_XOR_BUS: XorBus = XorBus(8);
pub const RANGE_TUPLE_CHECKER_BUS: usize = 11;
pub const CORE_MAX_READS_PER_CYCLE: usize = 3;
pub const CORE_MAX_WRITES_PER_CYCLE: usize = 1;
pub const CORE_MAX_ACCESSES_PER_CYCLE: usize = CORE_MAX_READS_PER_CYCLE + CORE_MAX_WRITES_PER_CYCLE;

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
        _ => panic!("Non-Core opcode: {:?}", opcode),
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct CoreOptions {
    pub num_public_values: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CoreState {
    pub clock_cycle: usize,
    pub timestamp: usize,
    pub pc: usize,
    pub is_done: bool,
}

impl CoreState {
    pub fn initial() -> Self {
        CoreState {
            clock_cycle: 0,
            timestamp: 1,
            pc: 0,
            is_done: false,
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Streams<F> {
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
}

/// Chip for the Core. Carries all state and owns execution.
#[derive(Debug)]
pub struct CoreChip<F: PrimeField32> {
    pub air: CoreAir,
    pub rows: Vec<Vec<F>>,
    pub state: CoreState,
    /// Program counter at the start of the current segment.
    pub start_state: CoreState,
    pub public_values: Vec<Option<F>>,
    pub memory_chip: MemoryChipRef<F>,

    // TODO[jpw] Unclear Core should own this
    pub streams: Streams<F>,
}

impl<F: PrimeField32> CoreChip<F> {
    pub fn new(
        options: CoreOptions,
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<F>,
    ) -> Self {
        Self::from_state(
            options,
            execution_bus,
            program_bus,
            memory_chip,
            CoreState::initial(),
        )
    }

    /// Sets the current state of the Core.
    pub fn set_state(&mut self, state: CoreState) {
        self.state = state;
    }

    /// Sets the current state of the Core.
    pub fn from_state(
        options: CoreOptions,
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_chip: MemoryChipRef<F>,
        state: CoreState,
    ) -> Self {
        let memory_bridge = memory_chip.borrow().memory_bridge();
        Self {
            air: CoreAir {
                options,
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
            },
            rows: vec![],
            state,
            start_state: state,
            public_values: vec![None; options.num_public_values],
            memory_chip,
            streams: Default::default(),
        }
    }
}
