pub mod codec;
#[cfg(test)]
pub mod tests;
pub mod types;

use crate::{
    mock_db::MockDbTable,
    table::codec::fixed_bytes::FixedBytesCodec,
    types::{Data, Index},
};
use std::collections::HashMap;
use types::{TableId, TableMetadata};

/// Read-only Table object that returns an underlying database table as simple types
pub struct Table<I: Index, D: Data> {
    /// Unique identifier for the table
    pub id: TableId,
    /// Metadata for the table
    pub metadata: TableMetadata,
    /// Body of the table, mapping index to data
    pub body: HashMap<I, D>,
}

impl<I: Index, D: Data> Table<I, D> {
    const SIZE_I: usize = I::MEMORY_SIZE;
    const SIZE_D: usize = D::MEMORY_SIZE;

    pub fn new(id: TableId, metadata: TableMetadata) -> Self {
        Self {
            id,
            metadata,
            body: HashMap::new(),
        }
    }

    pub fn from_db_table(db_table: &MockDbTable) -> Self {
        let body = db_table
            .items
            .iter()
            .map(|(k, v)| {
                let codec = FixedBytesCodec::<I, D>::new(
                    db_table.db_table_metadata.index_bytes,
                    db_table.db_table_metadata.data_bytes,
                );
                let index = codec.fixed_bytes_to_index(k.to_vec());
                let data = codec.fixed_bytes_to_data(v.to_vec());
                (index, data)
            })
            .collect::<HashMap<I, D>>();

        Self {
            id: db_table.id,
            metadata: TableMetadata::new(Self::SIZE_I, Self::SIZE_D),
            body,
        }
    }

    pub fn get_id(&self) -> TableId {
        self.id
    }

    pub fn get_id_hex(&self) -> String {
        "0x".to_string() + &self.id.to_string()
    }

    /// Reads directly from the table
    pub fn read(&self, index: I) -> Option<D> {
        self.body.get(&index).cloned()
    }

    pub fn get_index_bytes(&self) -> usize {
        std::mem::size_of::<I>()
    }

    pub fn get_data_bytes(&self) -> usize {
        std::mem::size_of::<D>()
    }
}
