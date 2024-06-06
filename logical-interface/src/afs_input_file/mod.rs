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

#[derive(Debug, Serialize, Deserialize)]
pub struct AfsInputFile {
    pub file_path: String,
    pub header: AfsHeader,
    pub contents: Vec<AfsOperation>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AfsHeader {
    pub index_bytes: usize,
    pub data_bytes: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AfsOperation {
    pub op: InputFileBodyOperation,
    pub args: Vec<String>,
}

impl AfsInputFile {
    pub fn new(file_path: String) -> Self {
        let (header, contents) = Self::parse(file_path.clone()).unwrap_or_else(|e| {
            panic!("Failed to parse AFS input file: {:?}", e);
        });
        Self {
            file_path,
            header,
            contents,
        }
    }

    pub fn parse(file_path: String) -> Result<(AfsHeader, Vec<AfsOperation>)> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().collect::<Result<Vec<String>, _>>()?;

        let mut afs_header = AfsHeader {
            index_bytes: 0,
            data_bytes: 0,
        };
        // reader.lines().take(2).map(|line| {
        for line in &lines[..2] {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let operation = parts[0];
            let value = parts[1].parse::<usize>().unwrap();
            match InputFileHeaderOperation::from_str(operation) {
                Ok(op) => {
                    println!("header {:?}:{:?}", op, parts[1]);
                    match op {
                        InputFileHeaderOperation::IndexBytes => {
                            afs_header.index_bytes = value;
                        }
                        InputFileHeaderOperation::DataBytes => {
                            afs_header.data_bytes = value;
                        }
                    }
                }
                Err(e) => {
                    panic!("Invalid operation: {:?}", e.to_string());
                }
            }
        }

        let afs_operations = lines[2..]
            .iter()
            .map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                let operation = parts[0];
                match InputFileBodyOperation::from_str(operation) {
                    Ok(op) => {
                        println!("{:?}:{:?}", op, parts[1]);
                        AfsOperation {
                            op,
                            args: parts[1..].iter().map(|s| s.to_string()).collect(),
                        }
                    }
                    Err(e) => {
                        panic!("Invalid operation: {:?}", e.to_string());
                    }
                }
            })
            .collect();

        Ok((afs_header, afs_operations))
    }
}
