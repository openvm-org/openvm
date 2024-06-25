use serde::{Deserialize, Serialize};

use crate::cpu::CpuOptions;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct VMParamsConfig {
    pub field_arithmetic_enabled: bool,
    pub limb_bits: usize,
    pub decomp: usize,
    /*pub max_program_length: usize,
    pub max_operations: usize,*/
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct VMConfig {
    pub vm: VMParamsConfig,
}

impl VMConfig {
    pub fn read_config_file(file: &str) -> VMConfig {
        let file_str = std::fs::read_to_string(file).unwrap_or_else(|_| {
            panic!("`config.toml` is required in the root directory of the project");
        });
        let config: VMConfig = toml::from_str(file_str.as_str()).unwrap_or_else(|e| {
            panic!("Failed to parse config file {}:\n{}", file, e);
        });
        config
    }

    pub fn cpu_options(&self) -> CpuOptions {
        CpuOptions {
            field_arithmetic_enabled: self.vm.field_arithmetic_enabled,
        }
    }
}
