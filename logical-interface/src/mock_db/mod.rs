pub mod utils;

use crate::afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions};
use color_eyre::eyre::{eyre, Result};
use hex::ToHex;
use std::collections::{hash_map::Entry, HashMap};
use utils::string_to_fixed_bytes_be;

#[derive(Debug)]
pub struct MockDb<const INDEX_BYTES: usize, const DATA_BYTES: usize> {
    pub map: HashMap<[u8; INDEX_BYTES], [u8; DATA_BYTES]>,
}

impl<const INDEX_BYTES: usize, const DATA_BYTES: usize> MockDb<INDEX_BYTES, DATA_BYTES> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn new_from_afi(afi: &AfsInputInstructions) -> Self {
        let mut map = HashMap::new();
        for op in &afi.operations {
            match op.operation {
                InputFileBodyOperation::Read => {}
                InputFileBodyOperation::Insert => {
                    let index = string_to_fixed_bytes_be::<INDEX_BYTES>(op.args[0].clone());
                    let data = string_to_fixed_bytes_be::<DATA_BYTES>(op.args[1].clone());
                    map.insert(index, data);
                }
                InputFileBodyOperation::Write => {}
            };
        }
        Self { map }
    }

    pub fn insert(&mut self, key: [u8; INDEX_BYTES], value: [u8; DATA_BYTES]) -> Result<()> {
        match self.map.entry(key) {
            Entry::Occupied(_) => Err(eyre!("Key already exists in the database")),
            Entry::Vacant(entry) => {
                entry.insert(value);
                Ok(())
            }
        }
    }

    pub fn get(&self, key: [u8; INDEX_BYTES]) -> Option<&[u8; DATA_BYTES]> {
        self.map.get(&key)
    }

    pub fn remove(&mut self, key: [u8; INDEX_BYTES]) -> Option<[u8; DATA_BYTES]> {
        self.map.remove(&key)
    }
}

impl<const INDEX_BYTES: usize, const DATA_BYTES: usize> Default
    for MockDb<INDEX_BYTES, DATA_BYTES>
{
    fn default() -> Self {
        Self::new()
    }
}
