use color_eyre::eyre::{eyre, Result};
use serde_derive::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    str::FromStr,
};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InputFileHeaderOperation {
    TableId,
    IndexBytes,
    DataBytes,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
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
            "TABLE_ID" => Ok(Self::TableId),
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

impl Display for InputFileBodyOperation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            InputFileBodyOperation::Read => "READ",
            InputFileBodyOperation::Insert => "INSERT",
            InputFileBodyOperation::Write => "WRITE",
        };
        write!(f, "{}", s)
    }
}
