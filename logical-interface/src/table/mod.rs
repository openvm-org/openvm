use color_eyre::eyre::{eyre, Result};
use serde_derive::{Deserialize, Serialize};

use crate::afs_input_instructions::AfsInputInstructions;

#[cfg(test)]
mod tests;

pub const MAX_OPS: usize = 128;

#[derive(Debug, Deserialize, Serialize)]
pub struct Table<T: PartialEq + Clone, U> {
    pub index: Vec<T>,
    pub data: Vec<U>,
    index_bytes: usize,
    data_bytes: usize,
}

impl<T: PartialEq + Clone, U> Table<T, U> {
    pub fn new(index: Vec<T>, data: Vec<U>) -> Self {
        if index.len() != data.len() {
            panic!("Index and data vectors must be the same length");
        }
        Self::check_index_len(index.len());

        let index_bytes = std::mem::size_of::<T>();
        let data_bytes = std::mem::size_of::<U>();

        Self {
            index,
            data,
            index_bytes,
            data_bytes,
        }
    }

    pub fn new_from_file(path: String) -> Self {
        let instructions = AfsInputInstructions::from_file(path);

        Self::new(vec![], vec![])
    }

    pub fn read(&self, index: T) -> Option<&U> {
        let i = self.find_index(index);
        if let Some(i) = i {
            Some(&self.data[i])
        } else {
            None
        }
    }

    pub fn insert(&mut self, index: T, data: U) -> Result<()> {
        if (self.index.len() + 1) > MAX_OPS {
            return Err(eyre!(
                "Table size ({}) exceeds maximum number of operations ({})",
                self.index.len() + 1,
                MAX_OPS,
            ));
        }
        let i = self.find_index(index.clone());
        if let Some(_i) = i {
            Err(eyre!("Index already exists"))
        } else {
            self.index.push(index);
            self.data.push(data);
            Ok(())
        }
    }

    pub fn write(&mut self, index: T, data: U) -> Result<()> {
        let i = self.find_index(index);
        if let Some(i) = i {
            self.data[i] = data;
            Ok(())
        } else {
            Err(eyre!("Index not found"))
        }
    }

    pub fn get_index_bytes(&self) -> usize {
        self.index_bytes
    }

    pub fn get_data_bytes(&self) -> usize {
        self.data_bytes
    }

    fn check_index_len(len: usize) {
        if len > MAX_OPS {
            panic!(
                "Table size ({}) exceeds maximum number of operations ({})",
                len, MAX_OPS
            );
        }
    }

    fn find_index(&self, index: T) -> Option<usize> {
        self.index.iter().position(|index_val| index_val == &index)
    }
}
