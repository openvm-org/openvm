#[cfg(test)]
pub mod tests;
use crate::{
    mock_db::MockDb,
    table::{codec::fixed_bytes::FixedBytesCodec, Table, TableId},
};
use num_traits::{FromBytes, ToBytes};
use std::hash::Hash;

pub struct AfsInterface<
    'a,
    T: ToBytes + FromBytes<Bytes = [u8; SIZE_T]> + Sized + Hash + Eq + PartialEq + Clone,
    U: ToBytes + FromBytes<Bytes = [u8; SIZE_U]> + Sized + Clone,
    const SIZE_T: usize,
    const SIZE_U: usize,
    const INDEX_BYTES: usize,
    const DATA_BYTES: usize,
> {
    pub db_ref: &'a mut MockDb<INDEX_BYTES, DATA_BYTES>,
    pub current_table: Option<Table<T, U, SIZE_T, SIZE_U, INDEX_BYTES, DATA_BYTES>>,
}

impl<
        'a,
        T: ToBytes + FromBytes<Bytes = [u8; SIZE_T]> + Sized + Hash + Eq + PartialEq + Clone,
        U: ToBytes + FromBytes<Bytes = [u8; SIZE_U]> + Sized + Clone,
        const SIZE_T: usize,
        const SIZE_U: usize,
        const INDEX_BYTES: usize,
        const DATA_BYTES: usize,
    > AfsInterface<'a, T, U, SIZE_T, SIZE_U, INDEX_BYTES, DATA_BYTES>
{
    pub fn new(db_ref: &'a mut MockDb<INDEX_BYTES, DATA_BYTES>) -> Self {
        Self {
            db_ref,
            current_table: None,
        }
    }

    pub fn get_table(
        &mut self,
        table_id: TableId,
    ) -> Option<&Table<T, U, SIZE_T, SIZE_U, INDEX_BYTES, DATA_BYTES>> {
        let db_table = self.db_ref.get_table(table_id)?;
        self.current_table = Some(Table::from_db_table(db_table));
        self.current_table.as_ref()
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

    pub fn insert(&mut self, table_id: TableId, index: T, data: U) -> Option<()> {
        let codec = FixedBytesCodec::<T, U, SIZE_T, SIZE_U>::new(INDEX_BYTES, DATA_BYTES);
        let index_bytes = codec.index_to_fixed_bytes(index);
        let data_bytes = codec.data_to_fixed_bytes(data);
        self.db_ref.insert_data(table_id, index_bytes, data_bytes)?;
        Some(())
    }

    pub fn write(&mut self, table_id: TableId, index: T, data: U) -> Option<()> {
        let codec = FixedBytesCodec::<T, U, SIZE_T, SIZE_U>::new(INDEX_BYTES, DATA_BYTES);
        let index_bytes = codec.index_to_fixed_bytes(index);
        let data_bytes = codec.data_to_fixed_bytes(data);
        self.db_ref.write_data(table_id, index_bytes, data_bytes)?;
        Some(())
    }
}
