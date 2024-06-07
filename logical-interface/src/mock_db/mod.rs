pub mod utils;

use crate::afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions};
use color_eyre::eyre::{eyre, Result};
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::{self, Debug, Error, Formatter},
};
use utils::string_to_fixed_bytes_be;

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct FixedBytes<const N: usize>([u8; N]);

pub struct MockDb<const INDEX_BYTES: usize, const DATA_BYTES: usize> {
    pub map: HashMap<FixedBytes<INDEX_BYTES>, FixedBytes<DATA_BYTES>>,
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
                InputFileBodyOperation::Insert | InputFileBodyOperation::Write => {
                    let index = FixedBytes::<INDEX_BYTES>(string_to_fixed_bytes_be::<INDEX_BYTES>(
                        op.args[0].clone(),
                    ));
                    let data = FixedBytes::<DATA_BYTES>(string_to_fixed_bytes_be::<DATA_BYTES>(
                        op.args[1].clone(),
                    ));
                    map.insert(index, data);
                }
            };
        }
        Self { map }
    }

    pub fn insert(
        &mut self,
        key: FixedBytes<INDEX_BYTES>,
        value: FixedBytes<DATA_BYTES>,
    ) -> Result<()> {
        self.map.insert(key, value).unwrap();
        Ok(())
    }

    pub fn get(&self, key: FixedBytes<INDEX_BYTES>) -> Option<&FixedBytes<DATA_BYTES>> {
        self.map.get(&key)
    }

    pub fn remove(&mut self, key: FixedBytes<INDEX_BYTES>) -> Option<FixedBytes<DATA_BYTES>> {
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

impl<const N: usize> AsRef<[u8]> for FixedBytes<N> {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl<const N: usize> Debug for FixedBytes<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "0x{}", hex::encode(self))
    }
}

impl<const INDEX_BYTES: usize, const DATA_BYTES: usize> Debug for MockDb<INDEX_BYTES, DATA_BYTES> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        self.map.keys().for_each(|k| {
            writeln!(f, "{:?}: {:?}", k, self.map.get(k).unwrap()).unwrap();
        });
        Ok(())
    }
}
