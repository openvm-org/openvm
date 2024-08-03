use std::fmt::Debug;

use serde_derive::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct FixedBytesConfig {
    pub index_bytes: usize,
    pub data_bytes: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FixedBytesCodec {
    pub table: FixedBytesConfig,
    pub db: FixedBytesConfig,
}

impl FixedBytesCodec {
    pub fn new(
        table_index_bytes: usize,
        table_data_bytes: usize,
        db_index_bytes: usize,
        db_data_bytes: usize,
    ) -> Self {
        Self {
            table: FixedBytesConfig {
                index_bytes: table_index_bytes,
                data_bytes: table_data_bytes,
            },
            db: FixedBytesConfig {
                index_bytes: db_index_bytes,
                data_bytes: db_data_bytes,
            },
        }
    }

    pub fn table_to_db_index_bytes(&self, table_index: Vec<u8>) -> Vec<u8> {
        if table_index.len() > self.db.index_bytes {
            panic!("Index size exceeds the maximum size");
        }
        let zeros_len = self.db.index_bytes - table_index.len();
        let mut fixed_index = vec![0; zeros_len];
        fixed_index.extend_from_slice(&table_index);
        if fixed_index.len() != self.db.index_bytes {
            panic!(
                "Invalid index size: {} for this codec, expected: {}",
                fixed_index.len(),
                self.db.index_bytes
            );
        }
        fixed_index
    }

    pub fn table_to_db_data_bytes(&self, table_data: Vec<u8>) -> Vec<u8> {
        if table_data.len() > self.db.data_bytes {
            panic!("Data size exceeds the maximum size");
        }
        let zeros_len = self.db.data_bytes - table_data.len();
        let mut fixed_data = vec![0; zeros_len];
        fixed_data.extend_from_slice(&table_data);
        if fixed_data.len() != self.db.data_bytes {
            panic!(
                "Invalid data size: {} for this codec, expected: {}",
                fixed_data.len(),
                self.db.data_bytes
            );
        }
        fixed_data
    }

    pub fn db_to_table_index_bytes(&self, db_index: Vec<u8>) -> Vec<u8> {
        let bytes_len = db_index.len();
        if bytes_len != self.db.index_bytes {
            panic!(
                "Index size ({}) is invalid for this codec (requires {})",
                bytes_len, self.db.index_bytes
            );
        }
        if self.table.index_bytes > bytes_len {
            panic!(
                "Index size ({}) is less than the expected size ({})",
                bytes_len, self.table.index_bytes
            );
        }

        // Get least significant size(I) bytes (big endian)
        let bytes_slice = &db_index[bytes_len - self.table.index_bytes..];
        bytes_slice.to_vec()
    }

    pub fn db_to_table_data_bytes(&self, db_data: Vec<u8>) -> Vec<u8> {
        let bytes_len = db_data.len();
        if bytes_len != self.db.data_bytes {
            panic!(
                "Data size ({}) is invalid for this codec (requires {})",
                bytes_len, self.db.data_bytes
            );
        }
        if self.table.data_bytes > bytes_len {
            panic!(
                "Data size ({}) is less than the expected size ({})",
                bytes_len, self.table.data_bytes
            );
        }

        // Get least significant size(D) bytes (big endian)
        let bytes_slice = &db_data[bytes_len - self.table.data_bytes..];
        bytes_slice.to_vec()
    }
}
