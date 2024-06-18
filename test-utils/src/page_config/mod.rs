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
pub struct TreeParamsConfig {
    pub init_leaf_cap: usize,
    pub init_internal_cap: usize,
    pub final_leaf_cap: usize,
    pub final_internal_cap: usize,
}

/// im keeping this here in case it is relevant
#[derive(Debug, Serialize, Deserialize)]
pub struct SchemaConfig {
    pub key_length: usize,
    pub limb_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageConfig {
    pub page: PageParamsConfig,
    pub schema: SchemaConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MultitierPageConfig {
    pub page: PageParamsConfig,
    pub tree: TreeParamsConfig,
    pub schema: SchemaConfig,
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

impl MultitierPageConfig {
    pub fn read_config_file(file: &str) -> MultitierPageConfig {
        let file_str = std::fs::read_to_string(file).unwrap_or_else(|_| {
            panic!("`config-1b.toml` is required in the root directory of the project");
        });
        let config: MultitierPageConfig = toml::from_str(file_str.as_str()).unwrap_or_else(|e| {
            panic!("Failed to parse config file {}:\n{}", file, e);
        });
        config
    }
}
