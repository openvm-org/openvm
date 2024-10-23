use std::sync::Arc;

use afs_primitives::xor::XorBus;
pub use air::CoreAir;
use p3_field::PrimeField32;
use parking_lot::Mutex;

use crate::{
    arch::{
        instructions::CoreOpcode::{self, *},
        ExecutionBridge, ExecutionBus,
    },
    system::{memory::MemoryControllerRef, program::ProgramBus, vm::Streams},
};
// TODO[zach]: Restore tests once we have control flow chip.
//#[cfg(test)]
//pub mod tests;

mod air;
mod bridge;
mod columns;
mod execute;
mod trace;

pub const INST_WIDTH: u32 = 1;

pub const READ_INSTRUCTION_BUS: usize = 8;
pub const RANGE_CHECKER_BUS: usize = 4;
pub const POSEIDON2_DIRECT_BUS: usize = 6;
pub const BYTE_XOR_BUS: XorBus = XorBus(8);
pub const RANGE_TUPLE_CHECKER_BUS: usize = 11;
pub const CORE_MAX_READS_PER_CYCLE: usize = 3;
pub const CORE_MAX_WRITES_PER_CYCLE: usize = 1;
pub const CORE_MAX_ACCESSES_PER_CYCLE: usize = CORE_MAX_READS_PER_CYCLE + CORE_MAX_WRITES_PER_CYCLE;

fn timestamp_delta(opcode: CoreOpcode) -> u32 {
    match opcode {
        LOADW | STOREW => 3,
        LOADW2 | STOREW2 => 4,
        FAIL => 0,
        PRINTF => 1,
        SHINTW => 2,
        HINT_INPUT | HINT_BITS | HINT_BYTES => 0,
        CT_START | CT_END => 0,
        DUMMY => 0,
    }
}

/// Chip for the Core. Carries all state and owns execution.
#[derive(Debug)]
pub struct CoreChip<F: PrimeField32> {
    pub air: CoreAir,
    pub rows: Vec<Vec<F>>,
    pub memory_controller: MemoryControllerRef<F>,
    pub streams: Arc<Mutex<Streams<F>>>,

    offset: usize,
}

impl<F: PrimeField32> CoreChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        streams: Arc<Mutex<Streams<F>>>,
        offset: usize,
    ) -> Self {
        let memory_bridge = memory_controller.borrow().memory_bridge();
        Self {
            air: CoreAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                memory_bridge,
                offset,
            },
            rows: vec![],
            memory_controller,
            streams,
            offset,
        }
    }
}
