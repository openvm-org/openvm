use std::{
    fs::File,
    io::{BufRead, BufReader},
    str::FromStr,
};

use color_eyre::eyre::Result;
use serde_derive::{Deserialize, Serialize};

use super::types::{AfsOperation, InputFileOp};

/// The AFS Operations File (*.afo) only contains a list of operations to be parsed into some action.
#[derive(Debug, Serialize, Deserialize)]
pub struct AfsOperationsFile {
    pub file_path: String,
    pub operations: Vec<AfsOperation>,
}

impl AfsOperationsFile {
    pub fn open(file_path: String) -> Self {
        let file = File::open(file_path.clone()).unwrap();
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().map(|line| line.unwrap()).collect();
        let operations = Self::parse(lines).unwrap();
        Self {
            file_path,
            operations,
        }
    }

    pub fn parse(lines: Vec<String>) -> Result<Vec<AfsOperation>> {
        let afs_operations = lines
            .iter()
            .map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let operation = parts[0];
                match InputFileOp::from_str(operation) {
                    Ok(operation) => AfsOperation {
                        operation,
                        args: parts[1..].iter().map(|s| s.to_string()).collect(),
                    },
                    Err(e) => {
                        panic!("Invalid operation on body: {:?}", e.to_string());
                    }
                }
            })
            .collect();

        Ok(afs_operations)
    }
}
