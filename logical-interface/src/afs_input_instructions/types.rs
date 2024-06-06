use color_eyre::eyre::{eyre, Result};
use serde_derive::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InputFileHeaderOperation {
    IndexBytes,
    DataBytes,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InputFileBodyOperation {
    Read,
    Insert,
    Write,
}

impl FromStr for InputFileHeaderOperation {
    type Err = color_eyre::eyre::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "INDEX_BYTES" => Ok(Self::IndexBytes),
            "DATA_BYTES" => Ok(Self::DataBytes),
            _ => Err(eyre!("Invalid operation: {}", s)),
        }
    }
}

impl FromStr for InputFileBodyOperation {
    type Err = color_eyre::eyre::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "READ" => Ok(Self::Read),
            "INSERT" => Ok(Self::Insert),
            "WRITE" => Ok(Self::Write),
            _ => Err(eyre!("Invalid operation: {}", s)),
        }
    }
}
