use color_eyre::eyre::{eyre, Result};
use serde_derive::{Deserialize, Serialize};
use std::{
    fmt::{Display, Formatter},
    str::FromStr,
};

/// Represents a single operation in an AFS input file
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AfsOperation {
    pub operation: InputFileOp,
    pub args: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InputFileHeader {
    TableId,
    IndexBytes,
    DataBytes,
}

impl FromStr for InputFileHeader {
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

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InputFileOp {
    Read,
    Insert,
    Write,
    Filter,
    InnerJoin,
    GroupBy,
}

impl FromStr for InputFileOp {
    type Err = color_eyre::eyre::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "READ" => Ok(Self::Read),
            "INSERT" => Ok(Self::Insert),
            "WRITE" => Ok(Self::Write),
            "FILTER" => Ok(Self::Filter),
            "INNER_JOIN" => Ok(Self::InnerJoin),
            "GROUP_BY" => Ok(Self::GroupBy),
            _ => Err(eyre!("Invalid operation: {}", s)),
        }
    }
}

impl Display for InputFileOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            InputFileOp::Read => "READ",
            InputFileOp::Insert => "INSERT",
            InputFileOp::Write => "WRITE",
            InputFileOp::Filter => "FILTER",
            InputFileOp::InnerJoin => "INNER_JOIN",
            InputFileOp::GroupBy => "GROUP_BY",
        };
        write!(f, "{}", s)
    }
}
