use enum_utils::FromStr;
use p3_baby_bear::BabyBear;

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

pub const CPU_MAX_READS_PER_CYCLE: usize = 2;
pub const CPU_MAX_WRITES_PER_CYCLE: usize = 1;
pub const CPU_MAX_ACCESSES_PER_CYCLE: usize = CPU_MAX_READS_PER_CYCLE + CPU_MAX_WRITES_PER_CYCLE;

pub const WORD_SIZE: usize = 1;

#[derive(Copy, Clone, Debug, PartialEq, Eq, FromStr, PartialOrd, Ord)]
#[repr(usize)]
#[allow(non_camel_case_types)]
pub enum OpCode {
    LOADW = 0,
    STOREW = 1,
    JAL = 2,
    BEQ = 3,
    BNE = 4,
    TERMINATE = 5,

    FADD = 6,
    FSUB = 7,
    FMUL = 8,
    FDIV = 9,

    FAIL = 10,
    PRINTF = 11,

    FE4ADD = 12,
    FE4SUB = 13,
    BBE4MUL = 14,
    BBE4INV = 15,

    PERM_POS2 = 16,
    COMP_POS2 = 17,

    /// Instruction to write the next hint word into memory.
    SHINTW = 18,

    /// Phantom instruction to prepare the next input vector for hinting.
    HINT_INPUT = 19,
    /// Phantom instruction to prepare the little-endian bit decomposition of a variable for hinting.
    HINT_BITS = 20,
}

impl OpCode {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(LOADW),
            1 => Some(STOREW),
            2 => Some(JAL),
            3 => Some(BEQ),
            4 => Some(BNE),
            5 => Some(TERMINATE),

            6 => Some(FADD),
            7 => Some(FSUB),
            8 => Some(FMUL),
            9 => Some(FDIV),

            10 => Some(FAIL),
            11 => Some(PRINTF),

            12 => Some(FE4ADD),
            13 => Some(FE4SUB),
            14 => Some(BBE4MUL),
            15 => Some(BBE4INV),

            16 => Some(PERM_POS2),
            17 => Some(COMP_POS2),

            18 => Some(SHINTW),
            19 => Some(HINT_INPUT),
            20 => Some(HINT_BITS),

            _ => None,
        }
    }
}

use crate::field_extension::FieldExtensionArithmeticAir;
use crate::poseidon2::Poseidon2Chip;
use OpCode::*;

pub const CORE_INSTRUCTIONS: [OpCode; 9] = [
    LOADW, STOREW, JAL, BEQ, BNE, TERMINATE, SHINTW, HINT_INPUT, HINT_BITS,
];
pub const FIELD_ARITHMETIC_INSTRUCTIONS: [OpCode; 4] = [FADD, FSUB, FMUL, FDIV];
pub const FIELD_EXTENSION_INSTRUCTIONS: [OpCode; 4] = [FE4ADD, FE4SUB, BBE4MUL, BBE4INV];

fn max_accesses_per_instruction(opcode: OpCode) -> usize {
    match opcode {
        LOADW | STOREW => 3,
        // JAL only does WRITE, but it is done as timestamp + 2
        JAL => 3,
        BEQ | BNE => 2,
        TERMINATE => 0,
        opcode if FIELD_ARITHMETIC_INSTRUCTIONS.contains(&opcode) => 3,
        opcode if FIELD_EXTENSION_INSTRUCTIONS.contains(&opcode) => {
            FieldExtensionArithmeticAir::max_accesses_per_instruction(opcode)
        }
        FAIL => 0,
        PRINTF => 1,
        COMP_POS2 | PERM_POS2 => {
            Poseidon2Chip::<16, BabyBear>::max_accesses_per_instruction(opcode)
        }
        SHINTW => 3,
        HINT_INPUT | HINT_BITS => 0,
        _ => panic!(),
    }
}

#[derive(Default, Clone, Copy)]
pub struct CpuOptions {
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
    pub compress_poseidon2_enabled: bool,
    pub perm_poseidon2_enabled: bool,
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
        if self.compress_poseidon2_enabled {
            result.push(COMP_POS2);
        }
        if self.perm_poseidon2_enabled {
            result.push(PERM_POS2);
        }
        result
    }

    pub fn num_enabled_instructions(&self) -> usize {
        self.enabled_instructions().len()
    }
}

#[derive(Default, Clone)]
pub struct CpuAir<const WORD_SIZE: usize> {
    pub options: CpuOptions,
}

impl<const WORD_SIZE: usize> CpuAir<WORD_SIZE> {
    pub fn new(options: CpuOptions) -> Self {
        Self { options }
    }
}
