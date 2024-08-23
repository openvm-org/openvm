use core::panic;

use p3_field::PrimeField32;

pub use air::CpuAir;
use Opcode::*;

use crate::{
    arch::instructions::{Opcode, Opcode::*},
    memory::offline_checker::bus::MemoryBus,
    vm::config::MemoryConfig,
};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const INST_WIDTH: usize = 1;

pub const READ_INSTRUCTION_BUS: usize = 0;
pub const NEW_MEMORY_BUS: MemoryBus = MemoryBus(1);
pub const RANGE_CHECKER_BUS: usize = 4;
pub const POSEIDON2_DIRECT_BUS: usize = 6;
pub const EXPAND_BUS: usize = 8;
pub const POSEIDON2_DIRECT_REQUEST_BUS: usize = 9;
pub const MEMORY_INTERFACE_BUS: usize = 10;

pub const CPU_MAX_READS_PER_CYCLE: usize = 3;
pub const CPU_MAX_WRITES_PER_CYCLE: usize = 1;
pub const CPU_MAX_ACCESSES_PER_CYCLE: usize = CPU_MAX_READS_PER_CYCLE + CPU_MAX_WRITES_PER_CYCLE;

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
        HINT_INPUT | HINT_BITS => 0,
        CT_START | CT_END => 0,
        NOP => 0,
        _ => panic!("Unknown opcode: {:?}", opcode),
    }
}

#[derive(Default, Clone, Copy)]
pub struct CpuOptions {
    pub num_public_values: usize,
}

#[derive(Default, Clone, Copy)]
/// State of the CPU.
pub struct CpuState {
    pub clock_cycle: usize,
    pub timestamp: usize,
    pub pc: usize,
    pub is_done: bool,
}

impl CpuOptions {
    pub fn poseidon2_enabled(&self) -> bool {
        self.compress_poseidon2_enabled || self.perm_poseidon2_enabled
    }

    pub fn enabled_instructions(&self) -> Vec<Opcode> {
        let mut result = CORE_INSTRUCTIONS.to_vec();
        if self.field_extension_enabled {
            result.extend(FIELD_EXTENSION_INSTRUCTIONS);
        }
        if self.field_arithmetic_enabled {
            result.extend(FIELD_ARITHMETIC_INSTRUCTIONS);
        }
        if self.modular_arithmetic_enabled {
            result.extend(MODULAR_ARITHMETIC_INSTRUCTIONS);
        }
        if self.compress_poseidon2_enabled {
            result.push(COMP_POS2);
        }
        if self.perm_poseidon2_enabled {
            result.push(PERM_POS2);
        }
        if self.is_less_than_enabled {
            result.push(F_LESS_THAN);
        }
        result
    }

    pub fn num_enabled_instructions(&self) -> usize {
        self.enabled_instructions().len()
    }
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
    pub fn new(options: CpuOptions, clk_max_bits: usize, decomp: usize) -> Self {
        Self {
            air: CpuAir::new(options, clk_max_bits, decomp),
            rows: vec![],
            state: CpuState::default(),
            start_state: CpuState::default(),
            pis: vec![],
        }
    }

    pub fn current_height(&self) -> usize {
        self.rows.len()
    }

    /// Sets the current state of the CPU.
    pub fn set_state(&mut self, state: CpuState) {
        self.state = state;
    }

    /// Sets the current state of the CPU.
    pub fn from_state(options: CpuOptions, state: CpuState, mem_config: MemoryConfig) -> Self {
        let mut chip = Self::new(options, mem_config.clk_max_bits, mem_config.decomp);
        chip.state = state;
        chip.start_state = state;
        chip
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
