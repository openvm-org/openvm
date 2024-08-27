use derive_new::new;
use serde::{Deserialize, Serialize};

use crate::cpu::CpuOptions;

pub const DEFAULT_MAX_SEGMENT_LEN: usize = (1 << 25) - 100;

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

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct VmConfig {
    // TODO: VmConfig should just contain CpuOptions to reduce redundancy
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
    pub compress_poseidon2_enabled: bool,
    pub perm_poseidon2_enabled: bool,
    pub keccak_enabled: bool,
    pub modular_multiplication_enabled: bool,
    pub is_less_than_enabled: bool,
    pub memory_config: MemoryConfig,
    pub num_public_values: usize,
    pub max_segment_len: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
    pub collect_metrics: bool,
}

impl Default for VmConfig {
    fn default() -> Self {
        VmConfig {
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
            compress_poseidon2_enabled: true,
            perm_poseidon2_enabled: true,
            keccak_enabled: false,
            modular_multiplication_enabled: false,
            is_less_than_enabled: false,
            memory_config: Default::default(),
            num_public_values: 0,
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
            collect_metrics: false,
        }
    }
}

impl VmConfig {
    pub fn cpu_options(&self) -> CpuOptions {
        CpuOptions {
            num_public_values: self.num_public_values,
        }
    }

    pub fn core() -> Self {
        VmConfig {
            field_arithmetic_enabled: false,
            field_extension_enabled: false,
            compress_poseidon2_enabled: false,
            perm_poseidon2_enabled: false,
            keccak_enabled: false,
            ..Default::default()
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
