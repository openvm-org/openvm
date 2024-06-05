// use serde::Serialize;
use serde_derive::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum PageMode {
    ReadOnly,
    ReadWrite,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PageConfig {
    pub height: u32,
    pub width: u32,
    pub mode: PageMode,
    pub max_rw_ops: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SchemaConfig {
    pub key_length: u32,
    pub limb_size: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub page: PageConfig,
    pub schema: SchemaConfig,
}

impl Config {
    pub fn read_config_file(file: &str) -> Config {
        // let reader = std::fs::File::open(file).unwrap();
        let file_str = std::fs::read_to_string(file).unwrap();
        let config: Config = toml::from_str(file_str.as_str()).unwrap();
        config
    }
}

