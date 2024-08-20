use core::panic;

use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;

pub use air::CpuAir;
use OpCode::*;

use crate::{
    arch::{
        bridge::ExecutionBus,
        instructions::{
            FIELD_ARITHMETIC_INSTRUCTIONS, FIELD_EXTENSION_INSTRUCTIONS, OpCode, OpCode::*,
        },
    },
    poseidon2::Poseidon2Chip,
};

//#[cfg(test)]
//pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const INST_WIDTH: usize = 1;

pub const READ_INSTRUCTION_BUS: usize = 8;
pub const MEMORY_BUS: usize = 1;
pub const RANGE_CHECKER_BUS: usize = 4;
pub const POSEIDON2_DIRECT_BUS: usize = 6;
pub const IS_LESS_THAN_BUS: usize = 7;
pub const CPU_MAX_READS_PER_CYCLE: usize = 3;
pub const CPU_MAX_WRITES_PER_CYCLE: usize = 1;
pub const CPU_MAX_ACCESSES_PER_CYCLE: usize = CPU_MAX_READS_PER_CYCLE + CPU_MAX_WRITES_PER_CYCLE;

fn timestamp_delta(opcode: OpCode) -> usize {
    // If an instruction performs a writes, it must change timestamp by WRITE_DELTA.
    const WRITE_DELTA: usize = CPU_MAX_READS_PER_CYCLE + 1;
    match opcode {
        LOADW | STOREW | LOADW2 | STOREW2 => WRITE_DELTA,
        // JAL only does WRITE, but it is done as timestamp + 2
        JAL => WRITE_DELTA,
        TERMINATE => 0,
        PUBLISH => 2,
        F_LESS_THAN => WRITE_DELTA,
        FAIL => 0,
        PRINTF => 1,
        COMP_POS2 | PERM_POS2 => {
            Poseidon2Chip::<16, BabyBear>::max_accesses_per_instruction(opcode)
        }
        SHINTW => WRITE_DELTA,
        HINT_INPUT | HINT_BITS => 0,
        CT_START | CT_END => 0,
        NOP => 0,
        _ => panic!("Non-CPU opcode: {:?}", opcode),
    }
}

#[derive(Default, Clone, Copy)]
pub struct CpuOptions {
    pub num_public_values: usize,
}

#[derive(Default, Clone, Copy)]
pub struct CpuState {
    pub clock_cycle: usize,
    pub timestamp: usize,
    pub pc: usize,
    pub is_done: bool,
}

/// Chip for the CPU. Carries all state and owns execution.
pub struct CpuChip<const WORD_SIZE: usize, F: Clone> {
    pub air: CpuAir<WORD_SIZE>,
    pub rows: Vec<Vec<F>>,
    pub state: CpuState,
    /// Program counter at the start of the current segment.
    pub start_state: CpuState,
    /// Public inputs for the current segment.
    pub pis: Vec<F>,
}

impl<const WORD_SIZE: usize, F: Clone> CpuChip<WORD_SIZE, F> {
    pub fn new(options: CpuOptions, execution_bus: ExecutionBus) -> Self {
        Self::from_state(options, execution_bus, CpuState::default())
    }

    /// Sets the current state of the CPU.
    pub fn set_state(&mut self, state: CpuState) {
        self.state = state;
    }

    /// Sets the current state of the CPU.
    pub fn from_state(options: CpuOptions, execution_bus: ExecutionBus, state: CpuState) -> Self {
        Self {
            air: CpuAir {
                options,
                execution_bus,
            },
            rows: vec![],
            state,
            start_state: state,
            pis: vec![],
        }
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> CpuChip<WORD_SIZE, F> {
    /// Writes the public inputs for the current segment (beginning and end program counters).
    ///
    /// Should only be called after segment end.
    fn generate_pvs(&mut self) {
        let first_row_pc = self.start_state.pc;
        let last_row_pc = self.state.pc;
        self.pis = vec![
            F::from_canonical_usize(first_row_pc),
            F::from_canonical_usize(last_row_pc),
        ];
    }
}
