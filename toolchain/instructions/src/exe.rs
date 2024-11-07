use std::collections::BTreeMap;

use p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::program::Program;

/// Memory image is a map from (address space, address) to word.
pub type MemoryImage<F> = BTreeMap<(F, F), F>;

/// Executable program for AxVM.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize",
    deserialize = "F: std::cmp::Ord + Deserialize<'de>"
))]
pub struct AxVmExe<F> {
    /// Program to execute.
    pub program: Program<F>,
    /// Start address of pc.
    pub pc_start: u32,
    /// Initial memory image.
    pub init_memory: MemoryImage<F>,
}

impl<F> AxVmExe<F> {
    pub fn new(program: Program<F>, pc_start: u32, init_memory: MemoryImage<F>) -> Self {
        Self {
            program,
            pc_start,
            init_memory,
        }
    }
    pub fn new_simple(program: Program<F>) -> Self {
        Self {
            program,
            pc_start: 0,
            init_memory: BTreeMap::new(),
        }
    }
    pub fn new_without_mem(program: Program<F>, pc_start: u32) -> Self {
        Self {
            program,
            pc_start,
            init_memory: BTreeMap::new(),
        }
    }
}

impl<F: Field> From<Program<F>> for AxVmExe<F> {
    fn from(program: Program<F>) -> Self {
        Self::new_simple(program)
    }
}
