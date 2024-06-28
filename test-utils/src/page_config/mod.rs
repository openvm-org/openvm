use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum PageMode {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageParamsConfig {
    pub index_bytes: usize,
    pub data_bytes: usize,
    pub bits_per_fe: usize,
    pub height: usize,
    pub mode: PageMode,
    pub max_rw_ops: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageConfig {
    pub page: PageParamsConfig,
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
}
