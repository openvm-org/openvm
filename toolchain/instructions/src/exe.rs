use std::collections::BTreeMap;

use p3_field::Field;

use crate::{config::CustomOpConfig, program::Program};

/// Memory image is a map from (address space, address) to word.
pub type MemoryImage<F> = BTreeMap<(F, F), F>;

/// Executable program for AxVM.
#[derive(Clone, Debug, Default)]
pub struct AxVmExe<F> {
    /// Program to execute.
    pub program: Program<F>,
    /// Start address of pc.
    pub pc_start: u32,
    /// Initial memory image.
    pub init_memory: MemoryImage<F>,
    /// Custom operations configuration.
    pub custom_op_config: CustomOpConfig,
}

impl<F> AxVmExe<F> {
    pub fn new(program: Program<F>) -> Self {
        Self {
            program,
            pc_start: 0,
            init_memory: BTreeMap::new(),
            custom_op_config: Default::default(),
        }
    }
    pub fn with_pc_start(mut self, pc_start: u32) -> Self {
        self.pc_start = pc_start;
        self
    }
    pub fn with_init_memory(mut self, init_memory: MemoryImage<F>) -> Self {
        self.init_memory = init_memory;
        self
    }
    pub fn with_custom_op_config(mut self, custom_op_config: CustomOpConfig) -> Self {
        self.custom_op_config = custom_op_config;
        self
    }
}

impl<F: Field> From<Program<F>> for AxVmExe<F> {
    fn from(program: Program<F>) -> Self {
        Self::new(program)
    }
}
