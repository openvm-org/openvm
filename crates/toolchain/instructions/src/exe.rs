use std::collections::{BTreeMap, BTreeSet};

use openvm_stark_backend::p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::program::Program;

// TODO[jpw]: delete this
/// Memory image is a map from `(address space, address * size_of<CellType>)` to u8.
pub type SparseMemoryImage = BTreeMap<(u32, u32), u8>;
/// Stores the starting address, end address, and name of a set of function.
pub type FnBounds = BTreeMap<u32, FnBound>;

/// Executable program for OpenVM.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize",
    deserialize = "F: std::cmp::Ord + Deserialize<'de>"
))]
pub struct VmExe<F> {
    /// Program to execute.
    pub program: Program<F>,
    /// Start address of pc.
    pub pc_start: u32,
    /// Initial memory image.
    pub init_memory: SparseMemoryImage,
    /// Starting + ending bounds for each function.
    pub fn_bounds: FnBounds,
    /// Decoded instruction PCs that should begin a block during CFG construction.
    pub cfg_block_starts: BTreeSet<u32>,
}

impl<F> VmExe<F> {
    pub fn new(program: Program<F>) -> Self {
        Self {
            program,
            pc_start: 0,
            init_memory: BTreeMap::new(),
            fn_bounds: Default::default(),
            cfg_block_starts: Default::default(),
        }
    }
    pub fn with_pc_start(mut self, pc_start: u32) -> Self {
        self.pc_start = pc_start;
        self
    }
    pub fn with_init_memory(mut self, init_memory: SparseMemoryImage) -> Self {
        self.init_memory = init_memory;
        self
    }
}

impl<F: Field> From<Program<F>> for VmExe<F> {
    fn from(program: Program<F>) -> Self {
        Self::new(program)
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FnBound {
    pub start: u32,
    pub end: u32,
    pub name: String,
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;

    use super::*;

    #[test]
    fn vmexe_roundtrip_preserves_cfg_block_starts() {
        let exe = VmExe::<BabyBear> {
            cfg_block_starts: BTreeSet::from([4]),
            ..Default::default()
        };

        let encoded = bitcode::serialize(&exe).unwrap();
        let decoded: VmExe<BabyBear> = bitcode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.cfg_block_starts, exe.cfg_block_starts);
    }
}
