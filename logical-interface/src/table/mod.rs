pub mod codec;
#[cfg(test)]
pub mod tests;
pub mod types;

use crate::{
    mock_db::MockDbTable,
    table::codec::fixed_bytes::FixedBytesCodec,
    types::{Data, Index},
    utils::fixed_bytes_to_field_vec,
};
use std::collections::BTreeMap;
use types::{TableId, TableMetadata};

/// Read-only Table object that returns an underlying database table as simple types
pub struct Table<I: Index, D: Data> {
    /// Unique identifier for the table
    pub id: TableId,
    /// Metadata for the table
    pub metadata: TableMetadata,
    /// Body of the table, mapping index to data
    pub body: BTreeMap<I, D>,
}

impl<I: Index, D: Data> Table<I, D> {
    const SIZE_I: usize = I::MEMORY_SIZE;
    const SIZE_D: usize = D::MEMORY_SIZE;

    pub fn new(id: TableId, metadata: TableMetadata) -> Self {
        Self {
            id,
            metadata,
            body: BTreeMap::new(),
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
            .collect::<BTreeMap<I, D>>();

        Self {
            id: db_table.id,
            metadata: TableMetadata::new(Self::SIZE_I, Self::SIZE_D),
            body,
        }
    }

    pub fn from_page(id: TableId, page: Vec<Vec<u32>>) -> Self {
        let row_size = 1 + Self::SIZE_I / 2 + Self::SIZE_D / 2;

        let codec = FixedBytesCodec::<I, D>::new(Self::SIZE_I, Self::SIZE_D);

        let mut body = page
            .iter()
            .map(|row| {
                if row.len() != row_size {
                    panic!(
                        "Invalid row size: {} for this codec, expected: {}",
                        row.len(),
                        row_size
                    );
                }
                let index_bytes: Vec<u8> = row
                    .iter()
                    .skip(1)
                    .take(Self::SIZE_I / 2)
                    .flat_map(|&x| {
                        let bytes = x.to_be_bytes();
                        bytes[2..4].to_vec()
                    })
                    .collect::<Vec<u8>>();
                let data_bytes: Vec<u8> = row
                    .iter()
                    .skip(1 + Self::SIZE_I / 2)
                    .take(Self::SIZE_D / 2)
                    .flat_map(|&x| {
                        let bytes = x.to_be_bytes();
                        bytes[2..4].to_vec()
                    })
                    .collect::<Vec<u8>>();
                let index = codec.fixed_bytes_to_index(index_bytes);
                let data = codec.fixed_bytes_to_data(data_bytes);
                (index, data)
            })
            .collect::<BTreeMap<I, D>>();

        // Remove the 0 index which is from the padding
        let index_zero: Vec<u8> = vec![0; Self::SIZE_I];
        body.remove(&I::from_be_bytes(&index_zero).unwrap());

        Self {
            id,
            metadata: TableMetadata::new(Self::SIZE_I, Self::SIZE_D),
            body,
        }
    }

    pub fn to_page(&self, page_size: usize) -> Vec<Vec<u32>> {
        if self.body.len() > page_size {
            panic!(
                "Table size {} cannot be bigger than `page_size` {}",
                self.body.len(),
                page_size
            );
        }
        let codec =
            FixedBytesCodec::<I, D>::new(self.metadata.index_bytes, self.metadata.data_bytes);
        let mut page: Vec<Vec<u32>> = self
            .body
            .iter()
            .map(|(index, data)| {
                let is_alloc: Vec<u32> = vec![1];
                let index_bytes = codec.index_to_fixed_bytes(index.clone());
                let index_fields = fixed_bytes_to_field_vec(index_bytes);
                let data_bytes = codec.data_to_fixed_bytes(data.clone());
                let data_fields = fixed_bytes_to_field_vec(data_bytes);
                let mut page = is_alloc;
                page.extend(index_fields);
                page.extend(data_fields);
                page
            })
            .collect();
        let zeros: Vec<u32> =
            vec![0; 1 + self.metadata.index_bytes / 2 + self.metadata.data_bytes / 2];
        let remaining_rows = page_size - self.body.len();
        for _ in 0..remaining_rows {
            page.push(zeros.clone());
        }
        page
    }

    pub fn id(&self) -> TableId {
        self.id
    }

    pub fn id_hex(&self) -> String {
        "0x".to_string() + &self.id.to_string()
    }

    pub fn len(&self) -> usize {
        self.body.len()
    }

    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    /// Reads directly from the table
    pub fn read(&self, index: I) -> Option<D> {
        self.body.get(&index).cloned()
    }

    pub fn size_of_index(&self) -> usize {
        std::mem::size_of::<I>()
    }

    pub fn size_of_data(&self) -> usize {
        std::mem::size_of::<D>()
    }
}
