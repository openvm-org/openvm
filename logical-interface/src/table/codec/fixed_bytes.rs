use serde_derive::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::types::{Data, Index};

#[derive(Debug, Deserialize, Serialize)]
pub struct FixedBytesCodec<I, D>
where
    I: Index,
    D: Data,
{
    pub db_size_index: usize,
    pub db_size_data: usize,
    _phantom: std::marker::PhantomData<(I, D)>,
}

impl<I, D> FixedBytesCodec<I, D>
where
    I: Index,
    D: Data,
{
    const SIZE_I: usize = I::MEMORY_SIZE;
    const SIZE_D: usize = D::MEMORY_SIZE;

    pub fn new(db_size_index: usize, db_size_data: usize) -> Self {
        Self {
            db_size_index,
            db_size_data,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn index_to_fixed_bytes(&self, index: I) -> Vec<u8> {
        let index_bytes = index.to_be_bytes();
        let mut index_bytes = index_bytes.to_vec();
        if index_bytes.len() > self.db_size_index {
            panic!("Index size exceeds the maximum size");
        }
        let zeros_len = self.db_size_index - index_bytes.len();
        let mut db_index = vec![0; zeros_len];
        db_index.append(&mut index_bytes);
        if db_index.len() != self.db_size_index {
            panic!(
                "Invalid index size: {} for this codec, expected: {}",
                db_index.len(),
                self.db_size_index
            );
        }
        db_index
    }

    pub fn data_to_fixed_bytes(&self, data: D) -> Vec<u8> {
        let data_bytes = data.to_be_bytes();
        let mut data_bytes = data_bytes.to_vec();
        if data_bytes.len() > self.db_size_data {
            panic!("Data size exceeds the maximum size");
        }
        let zeros_len = self.db_size_data - data_bytes.len();
        let mut db_data = vec![0; zeros_len];
        db_data.append(&mut data_bytes);
        if db_data.len() != self.db_size_data {
            panic!(
                "Invalid data size: {} for this codec, expected: {}",
                db_data.len(),
                self.db_size_data
            );
        }
        db_data
    }

    pub fn fixed_bytes_to_index(&self, fixed_bytes: Vec<u8>) -> I {
        let bytes_len = fixed_bytes.len();
        if bytes_len != self.db_size_index {
            panic!(
                "Index size ({}) is invalid for this codec (requires {})",
                bytes_len, self.db_size_index
            );
        }
        if Self::SIZE_I > bytes_len {
            panic!(
                "Index size ({}) is less than the expected size ({})",
                bytes_len,
                Self::SIZE_I
            );
        }

        // Get least significant size(I) bytes (big endian)
        let bytes_slice = &fixed_bytes[bytes_len - Self::SIZE_I..];
        let bytes_vec = bytes_slice.to_vec();
        I::from_be_bytes(&bytes_vec).unwrap()
    }

    pub fn fixed_bytes_to_data(&self, fixed_bytes: Vec<u8>) -> D {
        let bytes_len = fixed_bytes.len();
        if bytes_len != self.db_size_data {
            panic!("Data size is invalid for this codec");
        }
        if Self::SIZE_D > bytes_len {
            panic!("Data size is less than the expected size");
        }

        // Get least significant size(D) bytes (big endian)
        let bytes_slice = &fixed_bytes[bytes_len - Self::SIZE_D..];
        let bytes_vec = bytes_slice.to_vec();
        D::from_be_bytes(&bytes_vec).unwrap()
    }
}
