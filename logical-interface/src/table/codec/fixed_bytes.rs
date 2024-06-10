use num_traits::{FromBytes, ToBytes};
use serde_derive::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Deserialize, Serialize)]
pub struct FixedBytesCodec<T, U, const SIZE_T: usize, const SIZE_U: usize>
where
    T: ToBytes + FromBytes<Bytes = [u8; SIZE_T]> + Sized + PartialEq + Clone,
    U: ToBytes + FromBytes<Bytes = [u8; SIZE_U]> + Sized + Clone,
{
    pub db_size_index: usize,
    pub db_size_data: usize,
    _phantom: std::marker::PhantomData<(T, U)>,
}

impl<T, U, const SIZE_T: usize, const SIZE_U: usize> FixedBytesCodec<T, U, SIZE_T, SIZE_U>
where
    T: ToBytes + FromBytes<Bytes = [u8; SIZE_T]> + Sized + PartialEq + Clone,
    U: ToBytes + FromBytes<Bytes = [u8; SIZE_U]> + Sized + Clone,
{
    const SIZE_T: usize = std::mem::size_of::<T>();
    const SIZE_U: usize = std::mem::size_of::<U>();

    pub fn new(db_size_index: usize, db_size_data: usize) -> Self {
        if std::mem::size_of::<T>() != SIZE_T {
            panic!("Index size is invalid for this codec");
        }
        if std::mem::size_of::<U>() != SIZE_U {
            panic!("Data size is invalid for this codec");
        }
        Self {
            db_size_index,
            db_size_data,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn index_to_fixed_bytes(&self, index: T) -> Vec<u8> {
        let index_bytes = index.to_be_bytes();
        let mut index_bytes = index_bytes.as_ref().to_vec();
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

    pub fn data_to_fixed_bytes(&self, data: U) -> Vec<u8> {
        let data_bytes = data.to_be_bytes();
        let mut data_bytes = data_bytes.as_ref().to_vec();
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

    pub fn fixed_bytes_to_index(&self, bytes: Vec<u8>) -> T {
        let bytes_len = bytes.len();
        if bytes_len != self.db_size_index {
            panic!("Index size is invalid for this codec");
        }
        if Self::SIZE_T > bytes_len {
            panic!("Index size is less than the expected size");
        }

        // Get least significant size_t bytes (big endian)
        let bytes_slice = &bytes[bytes_len - Self::SIZE_T..];
        let bytes_slice: &[u8; SIZE_T] = bytes_slice.try_into().unwrap();
        T::from_be_bytes(bytes_slice)
    }

    pub fn fixed_bytes_to_data(&self, bytes: Vec<u8>) -> U {
        let bytes_len = bytes.len();
        if bytes_len != self.db_size_data {
            panic!("Data size is invalid for this codec");
        }
        if Self::SIZE_U > bytes_len {
            panic!("Data size is less than the expected size");
        }

        // Get least significant size_t bytes (big endian)
        // let bytes_slice = &bytes[bytes_len - Self::SIZE_U..];
        // U::from_be_bytes(bytes_slice)
        let bytes_slice = &bytes[bytes_len - Self::SIZE_U..bytes_len];
        let mut data_bytes = vec![0; std::mem::size_of::<U>()];
        data_bytes.copy_from_slice(bytes_slice); // Copy the slice into the array
        let bytes_slice: &[u8; SIZE_U] = bytes_slice.try_into().unwrap();
        U::from_be_bytes(bytes_slice)
    }
}
