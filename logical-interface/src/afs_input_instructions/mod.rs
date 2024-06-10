#[cfg(test)]
pub mod tests;
pub mod types;

use color_eyre::eyre::Result;
use serde_derive::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufRead, BufReader},
    str::FromStr,
};
use types::{InputFileBodyOperation, InputFileHeaderOperation};

pub const HEADER_SIZE: usize = 3;

#[derive(Debug, Serialize, Deserialize)]
pub struct AfsInputInstructions {
    pub file_path: String,
    pub header: AfsHeader,
    pub operations: Vec<AfsOperation>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AfsHeader {
    pub table_id: String,
    pub index_bytes: usize,
    pub data_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AfsOperation {
    pub operation: InputFileBodyOperation,
    pub args: Vec<String>,
}

impl AfsInputInstructions {
    pub fn from_file(file_path: String) -> Self {
        let (header, operations) = Self::parse(file_path.clone()).unwrap_or_else(|e| {
            panic!("Failed to parse AFS input file: {:?}", e);
        });
        Self {
            file_path,
            header,
            operations,
        }
    }

    pub fn parse(file_path: String) -> Result<(AfsHeader, Vec<AfsOperation>)> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().collect::<Result<Vec<String>, _>>()?;

        let mut afs_header = AfsHeader {
            table_id: String::new(),
            index_bytes: 0,
            data_bytes: 0,
        };
        // reader.lines().take(2).map(|line| {
        for line in &lines[..HEADER_SIZE] {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let operation = parts[0];
            let value = parts[1].parse::<usize>().unwrap();
            match InputFileHeaderOperation::from_str(operation) {
                Ok(op) => {
                    println!("header {:?}:{:?}", op, parts[1]);
                    match op {
                        InputFileHeaderOperation::TableId => {
                            afs_header.table_id = parts[1].to_string();
                        }
                        InputFileHeaderOperation::IndexBytes => {
                            afs_header.index_bytes = value;
                        }
                        InputFileHeaderOperation::DataBytes => {
                            afs_header.data_bytes = value;
                        }
                    }
                }
                Err(e) => {
                    panic!("Invalid operation on header: {:?}", e.to_string());
                }
            }
        }

        if afs_header.index_bytes == 0 || afs_header.data_bytes == 0 {
            panic!("Index bytes and data bytes must be set in the header");
        }

        let afs_operations = lines[HEADER_SIZE..]
            .iter()
            .map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let operation = parts[0];
                match InputFileBodyOperation::from_str(operation) {
                    Ok(operation) => {
                        println!("{:?}:{:?}", operation, parts[1]);
                        AfsOperation {
                            operation,
                            args: parts[1..].iter().map(|s| s.to_string()).collect(),
                        }
                    }
                    Err(e) => {
                        panic!("Invalid operation on body: {:?}", e.to_string());
                    }
                }
            })
            .collect();

        Ok((afs_header, afs_operations))
    }
}
