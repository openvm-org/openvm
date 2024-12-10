use std::{collections::BTreeMap, fmt::Display};

use ax_stark_backend::p3_field::{Field, PrimeField32};
use serde::{Deserialize, Serialize, Serializer};

use crate::program::Program;

/// Memory image is a map from (address space, address) to word.
pub type MemoryImage<F> = BTreeMap<(F, F), F>;
/// Stores the starting address, end address, and name of a set of function.
pub type FnBounds = BTreeMap<u32, FnBound>;

impl<F: Serialize + Display> Serialize for AxVmExe<F> {
    // We need some custom serialization specifically for the memory image:
    // using tuple as the key type does not produce a valid JSON.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut memory_image = BTreeMap::new();
        for ((addr_space, addr), value) in &self.init_memory {
            memory_image.insert(format!("{},{}", addr_space, addr), value);
        }

        let mut state = serializer.serialize_struct("AxVmExe", 4)?;
        state.serialize_field("program", &self.program)?;
        state.serialize_field("pc_start", &self.pc_start)?;
        state.serialize_field("init_memory", &memory_image)?;
        state.serialize_field("fn_bounds", &self.fn_bounds)?;
        state.end()
    }
}

impl<'de, F: PrimeField32> Deserialize<'de> for AxVmExe<F> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        #[derive(Deserialize)]
        struct Helper<F> {
            program: Program<F>,
            pc_start: u32,
            init_memory: BTreeMap<String, F>,
            fn_bounds: FnBounds,
        }

        let helper = Helper::deserialize(deserializer)?;

        let mut init_memory = BTreeMap::new();
        for (key, value) in helper.init_memory {
            let mut parts = key.split(',');
            let addr_space = parts
                .next()
                .ok_or_else(|| Error::custom("Missing address space"))?;
            let addr = parts
                .next()
                .ok_or_else(|| Error::custom("Missing address"))?;
            if parts.next().is_some() {
                return Err(Error::custom("Too many parts in memory key"));
            }

            let addr_space = F::from_canonical_u32(
                addr_space
                    .parse::<u32>()
                    .map_err(|_| Error::custom("Invalid address space"))?,
            );
            let addr = F::from_canonical_u32(
                addr.parse::<u32>()
                    .map_err(|_| Error::custom("Invalid address"))?,
            );

            init_memory.insert((addr_space, addr), value);
        }

        Ok(AxVmExe {
            program: helper.program,
            pc_start: helper.pc_start,
            init_memory,
            fn_bounds: helper.fn_bounds,
        })
    }
}

/// Executable program for AxVM.
#[derive(Clone, Debug, Default)]
pub struct AxVmExe<F> {
    /// Program to execute.
    pub program: Program<F>,
    /// Start address of pc.
    pub pc_start: u32,
    /// Initial memory image.
    pub init_memory: MemoryImage<F>,
    /// Starting + ending bounds for each function.
    pub fn_bounds: FnBounds,
}

impl<F> AxVmExe<F> {
    pub fn new(program: Program<F>) -> Self {
        Self {
            program,
            pc_start: 0,
            init_memory: BTreeMap::new(),
            fn_bounds: Default::default(),
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
}

impl<F: Field> From<Program<F>> for AxVmExe<F> {
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
