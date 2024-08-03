pub mod header;
pub mod operation;
pub mod operations_file;
#[cfg(test)]
pub mod tests;
pub mod types;
pub mod utils;

use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
};

use color_eyre::eyre::Result;
use header::AfsHeader;
use operations_file::AfsOperationsFile;
use serde_derive::{Deserialize, Serialize};
use types::AfsOperation;

pub const HEADER_SIZE: usize = 3;
pub const MAX_OPS: usize = 1_048_576; // 2^20

/// Instructions for reading an AFS input file
#[derive(Debug, Serialize, Deserialize)]
pub struct AfsInputFile {
    pub file_path: String,
    pub header: AfsHeader,
    pub operations: Vec<AfsOperation>,
}

impl AfsInputFile {
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

    pub fn open(file_path: &str) -> Result<Self> {
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
        let op_lines: Vec<String> = lines
            .iter()
            .skip(HEADER_SIZE)
            .map(|s| s.to_string())
            .collect();

        // let op_lines = &lines[HEADER_SIZE..];
        Self::check_num_ops(op_lines.len());

        let afs_header = AfsHeader::parse(lines)?;
        let afs_operations = AfsOperationsFile::parse(op_lines)?;

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
