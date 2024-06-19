use serde_derive::{Deserialize, Serialize};
use std::{fmt::Debug, marker::PhantomData};

use crate::types::{Data, Index};

#[derive(Debug, Deserialize, Serialize)]
pub struct FixedBytesCodec<I, D>
where
    I: Index,
    D: Data,
{
    pub index_bytes: usize,
    pub data_bytes: usize,
    pub fixed_index_bytes: usize,
    pub fixed_data_bytes: usize,
    _phantom: PhantomData<(I, D)>,
}

impl<I, D> FixedBytesCodec<I, D>
where
    I: Index,
    D: Data,
{
    pub fn new(
        index_bytes: usize,
        data_bytes: usize,
        fixed_index_bytes: usize,
        fixed_data_bytes: usize,
    ) -> Self {
        Self {
            index_bytes,
            data_bytes,
            fixed_index_bytes,
            fixed_data_bytes,
            _phantom: PhantomData,
        }
    }

    pub fn index_to_fixed_bytes(&self, index: I) -> Vec<u8> {
        let index_bytes = index.to_be_bytes();
        let mut index_bytes = index_bytes.to_vec();
        if index_bytes.len() > self.fixed_index_bytes {
            panic!("Index size exceeds the maximum size");
        }
        let zeros_len = self.fixed_index_bytes - index_bytes.len();
        let mut fixed_index = vec![0; zeros_len];
        fixed_index.append(&mut index_bytes);
        if fixed_index.len() != self.fixed_index_bytes {
            panic!(
                "Invalid index size: {} for this codec, expected: {}",
                fixed_index.len(),
                self.fixed_index_bytes
            );
        }
        fixed_index
    }

    pub fn data_to_fixed_bytes(&self, data: D) -> Vec<u8> {
        let data_bytes = data.to_be_bytes();
        let mut data_bytes = data_bytes.to_vec();
        if data_bytes.len() > self.fixed_data_bytes {
            panic!("Data size exceeds the maximum size");
        }
        let zeros_len = self.fixed_data_bytes - data_bytes.len();
        let mut fixed_data = vec![0; zeros_len];
        fixed_data.append(&mut data_bytes);
        if fixed_data.len() != self.fixed_data_bytes {
            panic!(
                "Invalid data size: {} for this codec, expected: {}",
                fixed_data.len(),
                self.fixed_data_bytes
            );
        }
        fixed_data
    }

    pub fn fixed_bytes_to_index(&self, fixed_bytes: Vec<u8>) -> I {
        let bytes_len = fixed_bytes.len();
        if bytes_len != self.fixed_index_bytes {
            panic!(
                "Index size ({}) is invalid for this codec (requires {})",
                bytes_len, self.fixed_index_bytes
            );
        }
        if self.index_bytes > bytes_len {
            panic!(
                "Index size ({}) is less than the expected size ({})",
                bytes_len, self.index_bytes
            );
        }

        // Get least significant size(I) bytes (big endian)
        let bytes_slice = &fixed_bytes[bytes_len - self.index_bytes..];
        let bytes_vec = bytes_slice.to_vec();
        I::from_be_bytes(&bytes_vec).unwrap()
    }

    pub fn fixed_bytes_to_data(&self, fixed_bytes: Vec<u8>) -> D {
        let bytes_len = fixed_bytes.len();
        if bytes_len != self.fixed_data_bytes {
            panic!(
                "Data size ({}) is invalid for this codec (requires {})",
                bytes_len, self.fixed_data_bytes
            );
        }
        if self.data_bytes > bytes_len {
            panic!(
                "Data size ({}) is less than the expected size ({})",
                bytes_len, self.data_bytes
            );
        }

        // Get least significant size(D) bytes (big endian)
        let bytes_slice = &fixed_bytes[bytes_len - self.data_bytes..];
        let bytes_vec = bytes_slice.to_vec();
        D::from_be_bytes(&bytes_vec).unwrap()
    }
}
