// use serde::Serialize;
use serde_derive::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum PageMode {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageConfig {
    pub height: u32,
    pub idx_len: u32,
    pub data_len: u32,
    pub mode: PageMode,
    pub max_rw_ops: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchemaConfig {
    pub idx_len: u32,
    pub limb_size: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub page: PageConfig,
    pub schema: SchemaConfig,
}

impl Config {
    pub fn read_config_file(file: &str) -> Config {
        let file_str = std::fs::read_to_string(file).unwrap_or_else(|_| {
            panic!("`config.toml` is required in the root directory of the project");
        });
        let config: Config = toml::from_str(file_str.as_str()).unwrap_or_else(|_| {
            panic!("Failed to parse config file: {}", file);
        });
        config
    }
}
