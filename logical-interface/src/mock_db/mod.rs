use serde_derive::{Deserialize, Serialize};

use crate::table::types::{TableId, TableMetadata};
use std::{
    collections::{btree_map::Entry, BTreeMap},
    fs::File,
    io::{Read, Write},
};

#[derive(Default, Serialize, Deserialize)]
pub struct MockDb {
    /// Map of table id to table
    pub tables: BTreeMap<TableId, MockDbTable>,
}

#[derive(Serialize, Deserialize)]
pub struct MockDbTable {
    /// Table id
    pub id: TableId,
    /// Metadata containing byte sizes for the db table index and data
    pub db_table_metadata: TableMetadata,
    /// Map of index to data
    pub items: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl MockDbTable {
    pub fn new(table_id: TableId, metadata: TableMetadata) -> Self {
        Self {
            id: table_id,
            db_table_metadata: metadata,
            items: BTreeMap::new(),
        }
    }
}

impl MockDb {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_file(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let mut reader = std::io::BufReader::new(file);
        let mut serialized = Vec::new();
        reader.read_to_end(&mut serialized).unwrap();
        let deserialized: MockDb = bincode::deserialize(&serialized).unwrap();
        deserialized
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let serialized = bincode::serialize(&self).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&serialized).unwrap();
        Ok(())
    }

    pub fn get_table(&self, table_id: TableId) -> Option<&MockDbTable> {
        self.tables.get(&table_id)
    }

    pub fn get_table_metadata(&self, table_id: TableId) -> Option<TableMetadata> {
        let table = self.tables.get(&table_id)?;
        Some(table.db_table_metadata.clone())
    }

    pub fn create_table(&mut self, table_id: TableId, metadata: TableMetadata) -> Option<()> {
        match self.tables.entry(table_id) {
            Entry::Occupied(_) => None,
            Entry::Vacant(entry) => {
                entry.insert(MockDbTable::new(table_id, metadata));
                Some(())
            }
        }
    }

    pub fn get_data(&self, table_id: TableId, index: Vec<u8>) -> Option<Vec<u8>> {
        let table = self.get_table(table_id)?;
        let data = table.items.get(&index)?;
        Some(data.to_vec())
    }

    pub fn insert_data(&mut self, table_id: TableId, index: Vec<u8>, data: Vec<u8>) -> Option<()> {
        let table = self.tables.get_mut(&table_id)?;
        match table.items.entry(index) {
            Entry::Occupied(_) => None,
            Entry::Vacant(entry) => {
                entry.insert(data);
                Some(())
            }
        }
    }

    pub fn write_data(&mut self, table_id: TableId, index: Vec<u8>, data: Vec<u8>) -> Option<()> {
        let table = self.tables.get_mut(&table_id)?;
        match table.items.entry(index) {
            Entry::Occupied(mut entry) => {
                entry.insert(data);
                Some(())
            }
            Entry::Vacant(_) => None,
        }
    }

    pub fn remove_data(&mut self, table_id: TableId, index: Vec<u8>) -> Option<()> {
        let table = self.tables.get_mut(&table_id)?;
        table.items.remove(&index).map(|_| ())
    }
}
