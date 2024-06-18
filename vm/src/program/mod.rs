//use crate::range_gate::RangeCheckerGateChip;
use getset::{CopyGetters, Getters};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

//pub const WORD_SIZE: usize = 1;
pub const INST_WIDTH: usize = 1;

pub const READ_INSTRUCTION_BUS: usize = 0;
pub const MEMORY_BUS: usize = 1;
pub const ARITHMETIC_BUS: usize = 2;

pub const NUM_CORE_OPERATIONS: usize = 5;
pub const NUM_ARITHMETIC_OPERATIONS: usize = 4;

#[derive(Default, Clone, Copy, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct CPUOptions {
    field_arithmetic_enabled: bool,
}

impl CPUOptions {
    pub fn num_operations(&self) -> usize {
        NUM_CORE_OPERATIONS + if self.field_arithmetic_enabled { NUM_ARITHMETIC_OPERATIONS } else { 0 }
    }
}

#[derive(Default, Clone, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct ProgramAir {
    
}

#[derive(Default, Getters)]
pub struct CPUChip {
    pub air: CPUAir,
    //pub range_checker: Arc<RangeCheckerGateChip>,
}

impl CPUChip {
    pub fn new(
        field_arithmetic_enabled: bool,
        //range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        let air = CPUAir {
            options: CPUOptions {
                field_arithmetic_enabled,
            },
        };

        Self {
            air, /*range_checker*/
        }
    }
}
