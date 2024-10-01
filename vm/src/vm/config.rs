use std::{collections::BTreeMap, ops::Range};

use derive_new::new;
use p3_field::PrimeField32;
use serde::{Deserialize, Serialize};

use crate::{
    arch::chips::{InstructionExecutorVariant, MachineChipVariant},
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

#[derive(Debug, Clone)]
pub struct VmConfig<F: PrimeField32> {
    pub executors: BTreeMap<usize, (InstructionExecutorVariant<F>, usize)>, // (who executes, offset)
    pub chips: Vec<MachineChipVariant<F>>,

    pub poseidon2_max_constraint_degree: Option<usize>,
    pub memory_config: MemoryConfig,
    pub num_public_values: usize,
    pub max_segment_len: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
    pub collect_metrics: bool,
    pub bigint_limb_size: usize,
}

impl<F: PrimeField32> VmConfig<F> {
    pub fn from_parameters(
        poseidon2_max_constraint_degree: Option<usize>,
        memory_config: MemoryConfig,
        num_public_values: usize,
        max_segment_len: usize,
        collect_metrics: bool,
        bigint_limb_size: usize,
    ) -> Self {
        VmConfig {
            executors: BTreeMap::new(),
            chips: Vec::new(),
            poseidon2_max_constraint_degree,
            memory_config,
            num_public_values,
            max_segment_len,
            collect_metrics,
            bigint_limb_size,
        }
    }

    pub fn add_executor_custom_offset(
        mut self,
        range: Range<usize>,
        executor: InstructionExecutorVariant<F>,
        offset: usize,
    ) -> Self {
        for i in range {
            self.executors.insert(i, (executor.clone(), offset));
        }
        self
    }

    pub fn add_executor(
        mut self,
        range: Range<usize>,
        executor: InstructionExecutorVariant<F>,
    ) -> Self {
        self.add_executor_custom_offset(range, executor, range.start)
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
        // VmConfig {
        //     field_arithmetic_enabled: true,
        //     field_extension_enabled: true,
        //     compress_poseidon2_enabled: true,
        //     perm_poseidon2_enabled: true,
        //     poseidon2_max_constraint_degree: Some(DEFAULT_POSEIDON2_MAX_CONSTRAINT_DEGREE),
        //     keccak_enabled: false,
        //     modular_addsub_enabled: false,
        //     modular_multdiv_enabled: false,
        //     is_less_than_enabled: false,
        //     u256_arithmetic_enabled: false,
        //     u256_multiplication_enabled: false,
        //     shift_256_enabled: false,
        //     ui_32_enabled: false,
        //     castf_enabled: false,
        //     secp256k1_enabled: false,
        //     memory_config: Default::default(),
        //     num_public_values: 0,
        //     max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
        //     collect_metrics: false,
        //     bigint_limb_size: 8,
        }
    }
}

impl VmConfig {
    pub fn core_options(&self) -> CoreOptions {
        CoreOptions {
            num_public_values: self.num_public_values,
        }
    }

    pub fn core() -> Self {
        VmConfig {
            field_arithmetic_enabled: false,
            field_extension_enabled: false,
            compress_poseidon2_enabled: false,
            poseidon2_max_constraint_degree: None,
            perm_poseidon2_enabled: false,
            keccak_enabled: false,
            ..Default::default()
        }
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
