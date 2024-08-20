use core::panic;
use std::fmt;

pub use air::CpuAir;
use enum_utils::FromStr;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use OpCode::*;

use crate::{
    field_extension::FieldExtensionArithmetic, memory::offline_checker::bus::MemoryBus,
    modular_multiplication::air::ModularMultiplicationVmAir, poseidon2::Poseidon2Chip,
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
pub const MEMORY_BUS: usize = 1;
pub const ARITHMETIC_BUS: usize = 2;
pub const FIELD_EXTENSION_BUS: usize = 3;
pub const RANGE_CHECKER_BUS: usize = 4;
pub const POSEIDON2_BUS: usize = 5;
pub const POSEIDON2_DIRECT_BUS: usize = 6;
// TODO[osama]: to be renamed to MEMORY_BUS
pub const NEW_MEMORY_BUS: MemoryBus = MemoryBus(7);
pub const EXPAND_BUS: usize = 8;
pub const POSEIDON2_DIRECT_REQUEST_BUS: usize = 9;
pub const MEMORY_INTERFACE_BUS: usize = 10;
pub const IS_LESS_THAN_BUS: usize = 11;

pub const CPU_MAX_READS_PER_CYCLE: usize = 3;
pub const CPU_MAX_WRITES_PER_CYCLE: usize = 1;
pub const CPU_MAX_ACCESSES_PER_CYCLE: usize = CPU_MAX_READS_PER_CYCLE + CPU_MAX_WRITES_PER_CYCLE;

pub const WORD_SIZE: usize = 1;

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromStr, PartialOrd, Ord)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum OpCode {
    LOADW = 0,
    STOREW = 1,
    LOADW2 = 2,
    STOREW2 = 3,
    JAL = 4,
    BEQ = 5,
    BNE = 6,
    TERMINATE = 7,
    PUBLISH = 8,

    FADD = 10,
    FSUB = 11,
    FMUL = 12,
    FDIV = 13,

    F_LESS_THAN = 14,

    FAIL = 20,
    PRINTF = 21,

    FE4ADD = 30,
    FE4SUB = 31,
    BBE4MUL = 32,
    BBE4INV = 33,

    PERM_POS2 = 40,
    COMP_POS2 = 41,

    /// Instruction to write the next hint word into memory.
    SHINTW = 50,

    /// Phantom instruction to prepare the next input vector for hinting.
    HINT_INPUT = 51,
    /// Phantom instruction to prepare the little-endian bit decomposition of a variable for hinting.
    HINT_BITS = 52,

    /// Phantom instruction to start tracing
    CT_START = 60,
    /// Phantom instruction to end tracing
    CT_END = 61,

    MOD_SECP256K1_ADD = 70,
    MOD_SECP256K1_SUB = 71,
    MOD_SECP256K1_MUL = 72,
    MOD_SECP256K1_DIV = 73,

    ADD256 = 80,
    SUB256 = 81,

    NOP = 100,
}

impl fmt::Display for OpCode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub const CORE_INSTRUCTIONS: [OpCode; 15] = [
    LOADW, STOREW, JAL, BEQ, BNE, TERMINATE, SHINTW, HINT_INPUT, HINT_BITS, PUBLISH, CT_START,
    CT_END, NOP, LOADW2, STOREW2,
];
pub const FIELD_ARITHMETIC_INSTRUCTIONS: [OpCode; 4] = [FADD, FSUB, FMUL, FDIV];
pub const FIELD_EXTENSION_INSTRUCTIONS: [OpCode; 4] = [FE4ADD, FE4SUB, BBE4MUL, BBE4INV];
pub const MODULAR_ARITHMETIC_INSTRUCTIONS: [OpCode; 4] = [
    MOD_SECP256K1_ADD,
    MOD_SECP256K1_SUB,
    MOD_SECP256K1_MUL,
    MOD_SECP256K1_DIV,
];

impl OpCode {
    pub fn all_opcodes() -> Vec<OpCode> {
        let mut all_opcodes = vec![];
        all_opcodes.extend(CORE_INSTRUCTIONS);
        all_opcodes.extend(FIELD_ARITHMETIC_INSTRUCTIONS);
        all_opcodes.extend(FIELD_EXTENSION_INSTRUCTIONS);
        all_opcodes.extend([FAIL, PRINTF]);
        all_opcodes.extend([PERM_POS2, COMP_POS2]);
        all_opcodes
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        Self::all_opcodes()
            .into_iter()
            .find(|&opcode| value == opcode as u8)
    }
}

fn timestamp_delta(opcode: OpCode) -> usize {
    // If an instruction performs a writes, it must change timestamp by WRITE_DELTA.
    const WRITE_DELTA: usize = CPU_MAX_READS_PER_CYCLE + 1;
    match opcode {
        LOADW | STOREW | LOADW2 | STOREW2 => WRITE_DELTA,
        // JAL only does WRITE, but it is done as timestamp + 2
        JAL => WRITE_DELTA,
        BEQ | BNE => 2,
        TERMINATE => 0,
        PUBLISH => 2,
        opcode if FIELD_ARITHMETIC_INSTRUCTIONS.contains(&opcode) => WRITE_DELTA,
        opcode if FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode) => {
            FieldExtensionArithmetic::max_accesses_per_instruction(opcode)
        }
        opcode if MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&opcode) => {
            ModularMultiplicationVmAir::max_accesses_per_instruction(opcode)
        }
        F_LESS_THAN => WRITE_DELTA,
        FAIL => 0,
        PRINTF => 1,
        COMP_POS2 | PERM_POS2 => {
            Poseidon2Chip::<16, 16, 1, BabyBear>::max_accesses_per_instruction(opcode)
        }
        SHINTW => WRITE_DELTA,
        HINT_INPUT | HINT_BITS => 0,
        CT_START | CT_END => 0,
        NOP => 0,
        _ => panic!("Unknown opcode: {:?}", opcode),
    }
}

#[derive(Default, Clone, Copy)]
pub struct CpuOptions {
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
    pub compress_poseidon2_enabled: bool,
    pub perm_poseidon2_enabled: bool,
    pub is_less_than_enabled: bool,
    pub num_public_values: usize,
    pub modular_arithmetic_enabled: bool,
}

#[derive(Default, Clone, Copy)]
/// State of the CPU.
pub struct ExecutionState {
    pub clock_cycle: usize,
    pub timestamp: usize,
    pub pc: usize,
    pub is_done: bool,
}

impl CpuOptions {
    pub fn poseidon2_enabled(&self) -> bool {
        self.compress_poseidon2_enabled || self.perm_poseidon2_enabled
    }

    pub fn enabled_instructions(&self) -> Vec<OpCode> {
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
    pub state: ExecutionState,
    /// Program counter at the start of the current segment.
    pub start_state: ExecutionState,
    /// Public inputs for the current segment.
    pub pis: Vec<F>,
}

impl<const WORD_SIZE: usize, F: Clone> CpuChip<WORD_SIZE, F> {
    pub fn new(options: CpuOptions, clk_max_bits: usize, decomp: usize) -> Self {
        Self {
            air: CpuAir::new(options, clk_max_bits, decomp),
            rows: vec![],
            state: ExecutionState::default(),
            start_state: ExecutionState::default(),
            pis: vec![],
        }
    }

    pub fn current_height(&self) -> usize {
        self.rows.len()
    }

    /// Sets the current state of the CPU.
    pub fn set_state(&mut self, state: ExecutionState) {
        self.state = state;
    }

    /// Sets the current state of the CPU.
    pub fn from_state(
        options: CpuOptions,
        state: ExecutionState,
        mem_config: MemoryConfig,
    ) -> Self {
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
