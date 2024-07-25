pub mod codec;
#[cfg(test)]
pub mod tests;
pub mod types;

use crate::{
    mock_db::MockDbTable, table::codec::fixed_bytes::FixedBytesCodec, utils::fixed_bytes_to_u16_vec,
};
use afs_page::common::{page::Page, page_cols::PageCols};
use serde_derive::{Deserialize, Serialize};
use std::collections::BTreeMap;
use types::{TableId, TableMetadata};

/// Read-only Table object that returns an underlying database table as simple types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    /// Unique identifier for the table
    pub id: TableId,
    /// Metadata for the table
    pub metadata: TableMetadata,
    /// Body of the table, mapping index to data
    pub body: BTreeMap<Vec<u8>, Vec<u8>>,
}

impl Table {
    pub fn new(id: TableId, metadata: TableMetadata) -> Self {
        Self {
            id,
            metadata,
            body: BTreeMap::new(),
        }
    }

    pub fn from_db_table(db_table: &MockDbTable, index_bytes: usize, data_bytes: usize) -> Self {
        let body = db_table
            .items
            .iter()
            .map(|(k, v)| {
                let codec = FixedBytesCodec::new(
                    index_bytes,
                    data_bytes,
                    db_table.db_table_metadata.index_bytes,
                    db_table.db_table_metadata.data_bytes,
                );
                let index = codec.db_to_table_index_bytes(k.to_vec());
                let data = codec.db_to_table_data_bytes(v.to_vec());
                (index, data)
            })
            .collect::<BTreeMap<Vec<u8>, Vec<u8>>>();

        Self {
            id: db_table.id,
            metadata: TableMetadata::new(
                db_table.db_table_metadata.index_bytes,
                db_table.db_table_metadata.data_bytes,
            ),
            body,
        }
    }

    pub fn from_page(id: TableId, page: Page, index_bytes: usize, data_bytes: usize) -> Self {
        let codec = FixedBytesCodec::new(
            index_bytes,
            data_bytes,
            page.idx_len() * 2,
            page.data_len() * 2,
        );
        let mut body = page
            .iter()
            .filter_map(|row| {
                let is_alloc_bytes = row.is_alloc.to_be_bytes();
                let is_alloc = u32::from_be_bytes(is_alloc_bytes);
                if is_alloc == 0 {
                    return None;
                }
                let index_bytes: Vec<u8> = row
                    .idx
                    .iter()
                    .flat_map(|&x| {
                        let bytes = x.to_be_bytes();
                        bytes[2..4].to_vec()
                    })
                    .collect::<Vec<u8>>();
                let data_bytes: Vec<u8> = row
                    .data
                    .iter()
                    .flat_map(|&x| {
                        let bytes = x.to_be_bytes();
                        bytes[2..4].to_vec()
                    })
                    .collect::<Vec<u8>>();
                let index = codec.db_to_table_index_bytes(index_bytes);
                let data = codec.db_to_table_data_bytes(data_bytes);
                Some((index, data))
            })
            .collect::<BTreeMap<Vec<u8>, Vec<u8>>>();

        // Remove the 0 index which is from the padding
        let index_zero: Vec<u8> = vec![0; index_bytes];
        body.remove(&codec.db_to_table_index_bytes(index_zero));

        Self {
            id,
            metadata: TableMetadata::new(index_bytes, data_bytes),
            body,
        }
    }

    pub fn to_page(&self, page_index_bytes: usize, page_data_bytes: usize, height: usize) -> Page {
        assert_eq!(self.metadata.index_bytes, page_index_bytes);
        assert_eq!(self.metadata.data_bytes, page_data_bytes);
        if self.body.len() > height {
            panic!(
                "Table height {} cannot be bigger than `height` {}",
                self.body.len(),
                height
            );
        }
        let codec = FixedBytesCodec::new(
            self.metadata.index_bytes,
            self.metadata.data_bytes,
            page_index_bytes,
            page_data_bytes,
        );
        let mut rows: Vec<PageCols<u32>> = self
            .body
            .iter()
            .map(|(index, data)| {
                let is_alloc: u32 = 1;
                let index_bytes = codec.table_to_db_index_bytes(index.clone());
                let index_fields = fixed_bytes_to_u16_vec(index_bytes);
                let data_bytes = codec.table_to_db_data_bytes(data.clone());
                let data_fields = fixed_bytes_to_u16_vec(data_bytes);
                PageCols {
                    is_alloc,
                    idx: index_fields,
                    data: data_fields,
                }
            })
            .collect::<Vec<PageCols<u32>>>();
        let zeros: PageCols<u32> = PageCols {
            is_alloc: 0,
            idx: vec![0; page_index_bytes / 2],
            data: vec![0; page_data_bytes / 2],
        };
        let remaining_rows = height - self.body.len();
        for _ in 0..remaining_rows {
            rows.push(zeros.clone());
        }
        Page::from_page_cols(rows)
    }

    pub fn id(&self) -> TableId {
        self.id
    }

    pub fn id_hex(&self) -> String {
        "0x".to_string() + &self.id.to_string()
    }

    pub fn read(&self, index: Vec<u8>) -> Option<Vec<u8>> {
        self.body.get(&index).cloned()
    }

    pub fn len(&self) -> usize {
        self.body.len()
    }

    pub fn is_empty(&self) -> bool {
        self.body.is_empty()
    }

    pub fn size_of_index(&self) -> usize {
        self.metadata.index_bytes
    }

    pub fn size_of_data(&self) -> usize {
        self.metadata.data_bytes
    }
}
