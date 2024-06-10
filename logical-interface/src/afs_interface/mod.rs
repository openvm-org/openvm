#[cfg(test)]
pub mod tests;
use crate::{
    mock_db::MockDb,
    table::{codec::fixed_bytes::FixedBytesCodec, Table, TableId},
};
use color_eyre::eyre::Result;
use num_traits::ToBytes;
use serde::Serialize;
use std::{error::Error, hash::Hash};

pub struct AfsInterface<
    'a,
    T: ToBytes
        + Hash
        + Eq
        + PartialEq
        + Clone
        + AsRef<[u8]>
        + for<'b> TryFrom<&'b [u8], Error = Box<dyn Error>>,
    U: ToBytes + Clone + AsRef<[u8]> + for<'b> TryFrom<&'b [u8], Error = Box<dyn Error>>,
    const INDEX_BYTES: usize,
    const DATA_BYTES: usize,
> {
    pub db_ref: &'a mut MockDb<INDEX_BYTES, DATA_BYTES>,
    pub current_table: Option<Table<T, U, INDEX_BYTES, DATA_BYTES>>,
}

impl<
        'a,
        T: ToBytes
            + Hash
            + Eq
            + PartialEq
            + Clone
            + AsRef<[u8]>
            + for<'b> TryFrom<&'b [u8], Error = Box<dyn Error>>,
        U: ToBytes + Clone + AsRef<[u8]> + for<'b> TryFrom<&'b [u8], Error = Box<dyn Error>>,
        const INDEX_BYTES: usize,
        const DATA_BYTES: usize,
    > AfsInterface<'a, T, U, INDEX_BYTES, DATA_BYTES>
{
    pub fn new(db_ref: &'a mut MockDb<INDEX_BYTES, DATA_BYTES>) -> Self {
        Self {
            db_ref,
            current_table: None,
        }
    }

    pub fn get_table(&mut self, table_id: TableId) -> &Table<T, U, INDEX_BYTES, DATA_BYTES> {
        let db_table = self.db_ref.get_table(table_id);
        self.current_table = Some(Table::from_db_table(db_table));
        self.current_table.as_ref().unwrap()
    }

    pub fn read(&mut self, table_id: TableId, index: T) -> Option<U> {
        if let Some(table) = self.current_table.as_ref() {
            let id = table.id;
            if id != table_id {
                self.get_table(table_id);
            }
        } else {
            self.get_table(table_id);
        }
        self.current_table.as_ref().unwrap().read(index)
    }

    pub fn insert(&mut self, table_id: TableId, index: T, data: U) -> Result<()> {
        let index_bytes =
            FixedBytesCodec::<T, U, INDEX_BYTES, DATA_BYTES>::index_to_fixed_bytes(index);
        let data_bytes =
            FixedBytesCodec::<T, U, INDEX_BYTES, DATA_BYTES>::data_to_fixed_bytes(data);
        self.db_ref.insert_data(table_id, index_bytes, data_bytes)?;
        Ok(())
    }

    pub fn write(&mut self, table_id: TableId, index: T, data: U) -> Result<()> {
        let index_bytes =
            FixedBytesCodec::<T, U, INDEX_BYTES, DATA_BYTES>::index_to_fixed_bytes(index);
        let data_bytes =
            FixedBytesCodec::<T, U, INDEX_BYTES, DATA_BYTES>::data_to_fixed_bytes(data);
        self.db_ref.insert_data(table_id, index_bytes, data_bytes)?;
        Ok(())
    }
}
