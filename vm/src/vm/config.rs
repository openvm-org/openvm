use serde::{Deserialize, Serialize};

use crate::cpu::CpuOptions;

pub const DEFAULT_MAX_SEGMENT_LEN: usize = (1 << 20) - 100;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct VmConfig {
    // TODO: VmConfig should just contain CpuOptions to reduce redundancy
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
    pub compress_poseidon2_enabled: bool,
    pub perm_poseidon2_enabled: bool,
    pub perm_keccak_enabled: bool,
    pub limb_bits: usize,
    pub decomp: usize,
    pub num_public_values: usize,
    pub max_segment_len: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
}

impl Default for VmConfig {
    fn default() -> Self {
        VmConfig {
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
            compress_poseidon2_enabled: true,
            perm_poseidon2_enabled: true,
            perm_keccak_enabled: false,
            limb_bits: 28,
            decomp: 4,
            num_public_values: 0,
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
        }
    }
}

impl VmConfig {
    pub fn cpu_options(&self) -> CpuOptions {
        CpuOptions {
            field_arithmetic_enabled: self.field_arithmetic_enabled,
            field_extension_enabled: self.field_extension_enabled,
            compress_poseidon2_enabled: self.compress_poseidon2_enabled,
            perm_poseidon2_enabled: self.perm_poseidon2_enabled,
            perm_keccak_enabled: self.perm_keccak_enabled,
            num_public_values: self.num_public_values,
        }
    }

    pub fn core() -> Self {
        VmConfig {
            field_arithmetic_enabled: false,
            field_extension_enabled: false,
            compress_poseidon2_enabled: false,
            perm_poseidon2_enabled: false,
            perm_keccak_enabled: false,
            limb_bits: 28,
            decomp: 4,
            num_public_values: 0,
            max_segment_len: DEFAULT_MAX_SEGMENT_LEN,
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
