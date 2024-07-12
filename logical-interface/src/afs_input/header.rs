use std::str::FromStr;

use color_eyre::eyre::Result;
use serde_derive::{Deserialize, Serialize};

use super::{types::InputFileHeader, HEADER_SIZE};

/// Header of an AFS input file, which corresponds to the first 3 lines of the file
#[derive(Debug, Serialize, Deserialize)]
pub struct AfsHeader {
    pub table_id: String,
    pub index_bytes: usize,
    pub data_bytes: usize,
}

impl AfsHeader {
    pub fn new(table_id: String, index_bytes: usize, data_bytes: usize) -> AfsHeader {
        AfsHeader {
            table_id,
            index_bytes,
            data_bytes,
        }
    }

    pub fn parse(lines: Vec<String>) -> Result<AfsHeader> {
        let mut afs_header = AfsHeader {
            table_id: String::new(),
            index_bytes: 0,
            data_bytes: 0,
        };

        for line in &lines[..HEADER_SIZE] {
            let parts: Vec<&str> = line.split_whitespace().collect();
            let operation = parts[0];
            let value = parts[1];
            match InputFileHeader::from_str(operation) {
                Ok(op) => match op {
                    InputFileHeader::TableId => {
                        afs_header.table_id = value.to_string();
                    }
                    InputFileHeader::IndexBytes => {
                        afs_header.index_bytes = value.parse::<usize>().unwrap();
                    }
                    InputFileHeader::DataBytes => {
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

        Ok(afs_header)
    }
}
