use std::sync::Arc;

pub use air::CoreAir;
use p3_field::PrimeField32;
use parking_lot::Mutex;

use crate::{
    arch::{ExecutionBridge, ExecutionBus},
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
