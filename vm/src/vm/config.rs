use serde::{Deserialize, Serialize};

use crate::cpu::CpuOptions;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct VmParamsConfig {
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
    pub limb_bits: usize,
    pub decomp: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
}

impl VmParamsConfig {
    pub fn cpu_options(&self) -> CpuOptions {
        CpuOptions {
            field_arithmetic_enabled: self.field_arithmetic_enabled,
            field_extension_enabled: self.field_extension_enabled,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct VmConfig {
    pub vm: VmParamsConfig,
}

impl VmConfig {
    pub fn read_config_file(file: &str) -> Result<VmConfig, String> {
        let file_str = std::fs::read_to_string(file).map_err(|_| {
            String::from("`config.toml` is required in the root directory of the project")
        })?;
        let config: VmConfig = toml::from_str(file_str.as_str())
            .map_err(|e| format!("Failed to parse config file {}:\n{}", file, e))?;
        Ok(config)
    }
}
