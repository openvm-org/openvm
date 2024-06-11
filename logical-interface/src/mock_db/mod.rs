use crate::table::types::{TableId, TableMetadata};
use std::collections::{hash_map::Entry, HashMap};

pub struct MockDb {
    /// Default metadata for tables created in this database
    pub default_table_metadata: TableMetadata,
    /// Map of table id to table
    pub tables: HashMap<TableId, MockDbTable>,
}

pub struct MockDbTable {
    /// Table id
    pub id: TableId,
    /// Metadata containing byte sizes for the db table index and data
    pub db_table_metadata: TableMetadata,
    /// Map of index to data
    pub items: HashMap<Vec<u8>, Vec<u8>>,
}

impl MockDbTable {
    pub fn new(table_id: TableId, metadata: TableMetadata) -> Self {
        Self {
            id: table_id,
            db_table_metadata: metadata,
            items: HashMap::new(),
        }
    }
}

impl MockDb {
    pub fn new(default_table_metadata: TableMetadata) -> Self {
        Self {
            default_table_metadata,
            tables: HashMap::new(),
        }
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
        self.check_index_size(&index);
        let table = self.get_table(table_id)?;
        let data = table.items.get(&index)?;
        Some(data.to_vec())
    }

    pub fn insert_data(&mut self, table_id: TableId, index: Vec<u8>, data: Vec<u8>) -> Option<()> {
        self.check_index_size(&index);
        self.check_data_size(&data);
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
        self.check_index_size(&index);
        self.check_data_size(&data);
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
        self.check_index_size(&index);
        let table = self.tables.get_mut(&table_id)?;
        table.items.remove(&index).map(|_| ())
    }

    fn check_index_size(&self, index: &[u8]) {
        if index.len() != self.default_table_metadata.index_bytes {
            panic!("Invalid index size: {}", index.len());
        }
    }

    fn check_data_size(&self, data: &[u8]) {
        if data.len() != self.default_table_metadata.data_bytes {
            panic!("Invalid data size: {}", data.len());
        }
    }
}
