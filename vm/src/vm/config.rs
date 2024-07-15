use serde::{Deserialize, Serialize};

use crate::cpu::CpuOptions;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct VmConfig {
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
    pub compress_poseidon2_enabled: bool,
    pub perm_poseidon2_enabled: bool,
    pub limb_bits: usize,
    pub decomp: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
}

impl VmConfig {
    pub fn cpu_options(&self) -> CpuOptions {
        CpuOptions {
            field_arithmetic_enabled: self.field_arithmetic_enabled,
            field_extension_enabled: self.field_extension_enabled,
            compress_poseidon2_enabled: self.compress_poseidon2_enabled,
            perm_poseidon2_enabled: self.perm_poseidon2_enabled,
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
