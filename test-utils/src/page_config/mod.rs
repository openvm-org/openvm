use std::fs;

use derive_more::Display;
use serde::{Deserialize, Serialize};

use crate::config::{EngineType, FriParameters};

#[derive(Debug, Clone, Default, Serialize, Deserialize, Display)]
#[serde(rename_all = "PascalCase")]
pub enum PageMode {
    #[default]
    ReadWrite,
    ReadOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageParamsConfig {
    pub index_bytes: usize,
    pub data_bytes: usize,
    pub bits_per_fe: usize,
    pub height: usize,
    pub mode: PageMode,
    pub max_rw_ops: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarkEngineConfig {
    pub engine: EngineType,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageConfig {
    pub page: PageParamsConfig,
    pub fri_params: FriParameters,
    pub stark_engine: StarkEngineConfig,
}

impl PageConfig {
    pub fn read_config_file(file: &str) -> PageConfig {
        let file_str = std::fs::read_to_string(file).unwrap_or_else(|_| {
            panic!("`config.toml` is required in the root directory of the project");
        });
        let config: PageConfig = toml::from_str(file_str.as_str()).unwrap_or_else(|e| {
            panic!("Failed to parse config file {}:\n{}", file, e);
        });
        config
    }

    pub fn generate_filename(&self) -> String {
        format!(
            "{:?}_{}x{}x{}-{}-{}_{}-{}-{}.toml",
            self.stark_engine.engine,
            self.page.index_bytes,
            self.page.data_bytes,
            self.page.height,
            self.page.max_rw_ops,
            self.page.bits_per_fe,
            self.fri_params.log_blowup,
            self.fri_params.num_queries,
            self.fri_params.proof_of_work_bits,
        )
    }

    pub fn save_to_file(&self, file: &str) {
        let file_str = toml::to_string(&self).unwrap();
        fs::write(file, file_str).unwrap();
    }
}
