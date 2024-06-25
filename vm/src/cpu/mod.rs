//use crate::range_gate::RangeCheckerGateChip;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub const WORD_SIZE: usize = 1;
pub const INST_WIDTH: usize = 1;

pub const READ_INSTRUCTION_BUS: usize = 0;
pub const MEMORY_BUS: usize = 1;
pub const ARITHMETIC_BUS: usize = 2;
pub const RANGE_CHECKER_BUS: usize = 3;

pub const NUM_CORE_OPERATIONS: usize = 6;
pub const NUM_ARITHMETIC_OPERATIONS: usize = 4;

pub const MAX_READS_PER_CYCLE: usize = 2;
pub const MAX_WRITES_PER_CYCLE: usize = 1;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
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
}

impl OpCode {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(OpCode::LOADW),
            1 => Some(OpCode::STOREW),
            2 => Some(OpCode::JAL),
            3 => Some(OpCode::BEQ),
            4 => Some(OpCode::BNE),
            5 => Some(OpCode::TERMINATE),
            6 => Some(OpCode::FADD),
            7 => Some(OpCode::FSUB),
            8 => Some(OpCode::FMUL),
            9 => Some(OpCode::FDIV),
            _ => None,
        }
    }
}

#[derive(Default, Clone, Copy)]
pub struct CpuOptions {
    pub field_arithmetic_enabled: bool,
}

impl CpuOptions {
    pub fn num_operations(&self) -> usize {
        NUM_CORE_OPERATIONS
            + if self.field_arithmetic_enabled {
                NUM_ARITHMETIC_OPERATIONS
            } else {
                0
            }
    }
}

#[derive(Default, Clone)]
pub struct CpuAir {
    pub options: CpuOptions,
}

#[derive(Default)]
pub struct CpuChip {
    pub air: CpuAir,
    //pub range_checker: Arc<RangeCheckerGateChip>,
}

impl CpuChip {
    pub fn new(
        field_arithmetic_enabled: bool,
        //range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        let air = CpuAir {
            options: CpuOptions {
                field_arithmetic_enabled,
            },
        };

        Self {
            air, /*range_checker*/
        }
    }
}
