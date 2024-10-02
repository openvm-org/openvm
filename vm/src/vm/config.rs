use std::ops::Range;

use derive_new::new;
use serde::{Deserialize, Serialize};
use strum::EnumCount;

use crate::{
    arch::{chips::InstructionExecutorVariantName, instructions::*},
    core::CoreOptions,
};

pub const DEFAULT_MAX_SEGMENT_LEN: usize = (1 << 25) - 100;
pub const DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE: usize = 7; // the sbox degree used for Poseidon2

#[derive(Debug, Serialize, Deserialize, Clone, Copy, new)]
pub struct MemoryConfig {
    pub addr_space_max_bits: usize,
    pub pointer_max_bits: usize,
    pub clk_max_bits: usize,
    pub decomp: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self::new(29, 29, 29, 16)
    }
}

fn default_executor_range(executor: InstructionExecutorVariantName) -> (Range<usize>, usize) {
    let (start, len, offset) = match executor {
        InstructionExecutorVariantName::Core => (0, CoreOpcode::COUNT, 0),
        InstructionExecutorVariantName::FieldArithmetic => {
            (0x100, FieldArithmeticOpcode::COUNT, 0x100)
        }
        InstructionExecutorVariantName::FieldExtension => {
            (0x110, FieldExtensionOpcode::COUNT, 0x110)
        }
        InstructionExecutorVariantName::Poseidon2 => (0x120, Poseidon2Opcode::COUNT, 0x120),
        InstructionExecutorVariantName::Keccak256 => (0x130, Keccak256Opcode::COUNT, 0x130),
        InstructionExecutorVariantName::ModularAddSub => (0x140, 4, 0x140),
        InstructionExecutorVariantName::ModularMultDiv => (0x144, 4, 0x140),
        InstructionExecutorVariantName::ArithmeticLogicUnit256 => (0x150, 7, 0x150),
        InstructionExecutorVariantName::U256Multiplication => (0x150 + 11, 1, 0x150),
        InstructionExecutorVariantName::Shift256 => (0x150 + 7, 4, 0x150),
        InstructionExecutorVariantName::Ui => (0x160, U32Opcode::COUNT, 0x160),
        InstructionExecutorVariantName::CastF => (0x170, CastfOpcode::COUNT, 0x170),
        InstructionExecutorVariantName::Secp256k1AddUnequal => (0x180, 1, 0x180),
        InstructionExecutorVariantName::Secp256k1Double => (0x181, 1, 0x180),
    };
    (start..(start + len), offset)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmConfig {
    pub executors: Vec<(Range<usize>, InstructionExecutorVariantName, usize)>, // (range of opcodes, who executes, offset)

    pub poseidon2_max_constraint_degree: Option<usize>,
    pub memory_config: MemoryConfig,
    pub num_public_values: usize,
    pub max_segment_len: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
    pub collect_metrics: bool,
    pub bigint_limb_size: usize,
}

impl VmConfig {
    pub fn from_parameters(
        poseidon2_max_constraint_degree: Option<usize>,
        memory_config: MemoryConfig,
        num_public_values: usize,
        max_segment_len: usize,
        collect_metrics: bool,
        bigint_limb_size: usize,
    ) -> Self {
        VmConfig {
            executors: Vec::new(),
            poseidon2_max_constraint_degree,
            memory_config,
            num_public_values,
            max_segment_len,
            collect_metrics,
            bigint_limb_size,
        }
    }

    pub fn add_executor(
        mut self,
        range: Range<usize>,
        executor: InstructionExecutorVariantName,
        offset: usize,
    ) -> Self {
        self.executors.push((range, executor, offset));
        self
    }

    pub fn add_default_executor(self, executor: InstructionExecutorVariantName) -> Self {
        let (range, offset) = default_executor_range(executor);
        self.add_executor(range, executor, offset)
    }
}

impl Default for VmConfig {
    fn default() -> Self {
        Self::from_parameters(
            Some(DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE),
            Default::default(),
            0,
            DEFAULT_MAX_SEGMENT_LEN,
            false,
            8,
        )
        .add_default_executor(InstructionExecutorVariantName::FieldArithmetic)
        .add_default_executor(InstructionExecutorVariantName::FieldExtension)
        .add_default_executor(InstructionExecutorVariantName::Poseidon2)
    }
}

impl VmConfig {
    pub fn core_options(&self) -> CoreOptions {
        CoreOptions {
            num_public_values: self.num_public_values,
        }
    }

    pub fn core() -> Self {
        Self::from_parameters(
            None,
            Default::default(),
            0,
            DEFAULT_MAX_SEGMENT_LEN,
            false,
            8,
        )
        .add_default_executor(InstructionExecutorVariantName::Core)
    }

    pub fn aggregation(poseidon2_max_constraint_degree: usize) -> Self {
        VmConfig {
            poseidon2_max_constraint_degree: Some(poseidon2_max_constraint_degree),
            num_public_values: 4,
            ..VmConfig::default()
        }
    }
}

impl VmConfig {
    pub fn read_config_file(file: &str) -> Result<Self, String> {
        let file_str = std::fs::read_to_string(file)
            .map_err(|_| format!("Could not load config file from: {file}"))?;
        let config: Self = toml::from_str(file_str.as_str())
            .map_err(|e| format!("Failed to parse config file {}:\n{}", file, e))?;
        Ok(config)
    }
}
