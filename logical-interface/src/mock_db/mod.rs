pub mod utils;

use crate::table::TableId;
use alloy_primitives::FixedBytes;
use color_eyre::eyre::{eyre, Result};
use std::collections::HashMap;

pub struct MockDb<const INDEX_BYTES: usize, const DATA_BYTES: usize> {
    pub tables: HashMap<TableId, MockDbTable<INDEX_BYTES, DATA_BYTES>>,
}

pub struct MockDbTable<const INDEX_BYTES: usize, const DATA_BYTES: usize> {
    pub id: TableId,
    pub items: HashMap<FixedBytes<INDEX_BYTES>, FixedBytes<DATA_BYTES>>,
}

impl<const INDEX_BYTES: usize, const DATA_BYTES: usize> MockDb<INDEX_BYTES, DATA_BYTES> {
    pub fn new() -> Self {
        Self {
            tables: HashMap::new(),
        }
    }

    // pub fn new_from_afi(afi: &AfsInputInstructions) -> Self {
    //     let mut map = HashMap::new();
    //     for op in &afi.operations {
    //         match op.operation {
    //             InputFileBodyOperation::Read => {}
    //             InputFileBodyOperation::Insert | InputFileBodyOperation::Write => {
    //                 let index_input = op.args[0].clone();
    //                 let index = FixedBytes::<INDEX_BYTES>::from(string_to_fixed_bytes_be::<
    //                     INDEX_BYTES,
    //                 >(index_input));
    //                 let data_input = op.args[1].clone();
    //                 let data = FixedBytes::<DATA_BYTES>::from(
    //                     string_to_fixed_bytes_be::<DATA_BYTES>(data_input),
    //                 );
    //                 map.insert(index, data);
    //             }
    //         };
    //     }
    //     Self { map }
    // }

    pub fn get_table(&self, table_id: TableId) -> Option<&MockDbTable<INDEX_BYTES, DATA_BYTES>> {
        self.tables.get(&table_id)
    }

    pub fn create_table(
        &mut self,
        table_id: TableId,
    ) -> Option<MockDbTable<INDEX_BYTES, DATA_BYTES>> {
        if self.tables.contains_key(&table_id) {
            return None;
        }
        let table = self.tables.insert(
            table_id,
            MockDbTable {
                id: table_id,
                items: HashMap::new(),
            },
        )?;
        Some(table)
    }

    pub fn get_data(&self, table_id: TableId, key: Vec<u8>) -> Option<Vec<u8>> {
        let table = self.get_table(table_id)?;
        let index = FixedBytes::<INDEX_BYTES>::from_slice(key.as_slice());
        let value = table.items.get(&index)?;
        Some(value.to_vec())
    }

    pub fn insert_data(&mut self, table_id: TableId, key: Vec<u8>, value: Vec<u8>) -> Option<()> {
        let mut table = self.tables.get_mut(&table_id);
        if table.is_none() {
            self.create_table(table_id)?;
            table = self.tables.get_mut(&table_id);
        }
        let table = table?;
        let index = FixedBytes::<INDEX_BYTES>::from_slice(key.as_slice());
        if table.items.contains_key(&index) {
            None
        } else {
            let value = FixedBytes::<DATA_BYTES>::from_slice(value.as_slice());
            table.items.insert(index, value).unwrap();
            Some(())
        }
    }

    pub fn write_data(&mut self, table_id: TableId, key: Vec<u8>, value: Vec<u8>) -> Option<()> {
        let mut table = self.tables.get_mut(&table_id);
        if table.is_none() {
            self.create_table(table_id)?;
            table = self.tables.get_mut(&table_id);
        }
        let table = table?;
        let index = FixedBytes::<INDEX_BYTES>::from_slice(key.as_slice());
        if !table.items.contains_key(&index) {
            None
        } else {
            let value = FixedBytes::<DATA_BYTES>::from_slice(value.as_slice());
            table.items.insert(index, value).unwrap();
            Some(())
        }
    }

    pub fn remove_data(&mut self, table_id: TableId, key: Vec<u8>) -> Option<()> {
        let table = self.tables.get_mut(&table_id)?;
        let index = FixedBytes::<INDEX_BYTES>::from_slice(key.as_slice());
        let removed = table.items.remove(&index);
        if removed.is_none() {
            return None;
        } else {
            Some(())
        }
    }
}

impl<const INDEX_BYTES: usize, const DATA_BYTES: usize> Default
    for MockDb<INDEX_BYTES, DATA_BYTES>
{
    fn default() -> Self {
        Self::new()
    }
}

// impl<const INDEX_BYTES: usize, const DATA_BYTES: usize> Debug for MockDb<INDEX_BYTES, DATA_BYTES> {
//     fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
//         self.map.keys().for_each(|k| {
//             writeln!(f, "{:?}: {:?}", k, self.map.get(k).unwrap()).unwrap();
//         });
//         Ok(())
//     }
// }
