#[cfg(test)]
pub mod tests;
pub mod types;

use color_eyre::eyre::Result;
use serde_derive::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    str::FromStr,
};
use types::{InputFileBodyOperation, InputFileHeaderOperation};

pub const HEADER_SIZE: usize = 3;
pub const MAX_OPS: usize = 1_048_576; // 2^20

/// Header of an AFS input file, which corresponds to the first 3 lines of the file
#[derive(Debug, Serialize, Deserialize)]
pub struct AfsHeader {
    pub table_id: String,
    pub index_bytes: usize,
    pub data_bytes: usize,
}

impl AfsHeader {
    pub fn new(table_id: String, index_bytes: usize, data_bytes: usize) -> Self {
        Self {
            table_id,
            index_bytes,
            data_bytes,
        }
    }
}

/// Represents a single operation in an AFS input file
#[derive(Debug, Serialize, Deserialize)]
pub struct AfsOperation {
    pub operation: InputFileBodyOperation,
    pub args: Vec<String>,
}

/// Instructions for reading an AFS input file
#[derive(Debug, Serialize, Deserialize)]
pub struct AfsInputInstructions {
    pub file_path: String,
    pub header: AfsHeader,
    pub operations: Vec<AfsOperation>,
}

impl AfsInputInstructions {
    pub fn new(file_path: &str, table_id: &str, index_bytes: usize, data_bytes: usize) -> Self {
        if table_id.is_empty() {
            panic!("`table_id` must not be empty");
        }
        if index_bytes == 0 || data_bytes == 0 {
            panic!("index/data bytes must not be 0");
        }
        Self {
            file_path: file_path.to_string(),
            header: AfsHeader {
                table_id: table_id.to_string().to_lowercase(),
                index_bytes,
                data_bytes,
            },
            operations: Vec::new(),
        }
    }

    pub fn from_file(file_path: &str) -> Result<Self> {
        let (header, operations) = Self::parse(file_path)?;
        Ok(Self {
            file_path: file_path.to_string(),
            header,
            operations,
        })
    }

    pub fn save_to_file(&self) -> Result<()> {
        let mut writer = File::create(&self.file_path)?;
        writeln!(writer, "TABLE_ID {}", self.header.table_id)?;
        writeln!(writer, "INDEX_BYTES {}", self.header.index_bytes)?;
        writeln!(writer, "DATA_BYTES {}", self.header.data_bytes)?;
        for operation in &self.operations {
            writeln!(
                writer,
                "{} {}",
                operation.operation,
                operation.args.join(" ")
            )?;
        }
        Ok(())
    }

    pub fn add_operations(&mut self, operations: Vec<AfsOperation>) {
        let total_ops = self.operations.len() + operations.len();
        Self::check_num_ops(total_ops);
        self.operations.extend(operations);
    }

    fn parse(file_path: &str) -> Result<(AfsHeader, Vec<AfsOperation>)> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().collect::<Result<Vec<String>, _>>()?;

        let mut afs_header = AfsHeader {
            table_id: String::new(),
            index_bytes: 0,
            data_bytes: 0,
        };

        for line in &lines[..HEADER_SIZE] {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let operation = parts[0];
            let value = parts[1];
            match InputFileHeaderOperation::from_str(operation) {
                Ok(op) => match op {
                    InputFileHeaderOperation::TableId => {
                        afs_header.table_id = value.to_string();
                    }
                    InputFileHeaderOperation::IndexBytes => {
                        afs_header.index_bytes = value.parse::<usize>().unwrap();
                    }
                    InputFileHeaderOperation::DataBytes => {
                        afs_header.data_bytes = value.parse::<usize>().unwrap();
                    }
                },
                Err(e) => {
                    panic!("Invalid operation on header: {:?}", e.to_string());
                }
            }
        }

        if afs_header.table_id.is_empty() {
            panic!("Table ID must be set in the header");
        }

        if afs_header.index_bytes == 0 || afs_header.data_bytes == 0 {
            panic!("Index bytes and data bytes must be set in the header");
        }

        let op_lines = &lines[HEADER_SIZE..];
        Self::check_num_ops(op_lines.len());

        let afs_operations = op_lines
            .iter()
            .map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let operation = parts[0];
                match InputFileBodyOperation::from_str(operation) {
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

        Ok((afs_header, afs_operations))
    }

    fn check_num_ops(num_ops: usize) {
        if num_ops > MAX_OPS {
            panic!(
                "Number of operations ({}) exceeds maximum ({})",
                num_ops, MAX_OPS
            );
        }
    }
}
