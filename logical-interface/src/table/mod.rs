pub mod codec;
#[cfg(test)]
pub mod tests;

use std::{collections::HashMap, error::Error, hash::Hash};

use crate::{
    afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions},
    mock_db::{utils::string_to_fixed_bytes_be, MockDb, MockDbTable},
    table::codec::fixed_bytes::FixedBytesCodec,
};
use alloy_primitives::wrap_fixed_bytes;
use color_eyre::eyre::{eyre, Result};
use num_traits::{FromBytes, ToBytes};

pub const MAX_OPS: usize = 128;

wrap_fixed_bytes!(pub struct TableId<32>;);

pub struct Table<
    T,
    U,
    const SIZE_T: usize,
    const SIZE_U: usize,
    const INDEX_BYTES: usize,
    const DATA_BYTES: usize,
> where
    T: ToBytes + FromBytes<Bytes = [u8; SIZE_T]> + Sized + Hash + Eq + PartialEq + Clone,
    U: ToBytes + FromBytes<Bytes = [u8; SIZE_U]> + Sized + Clone,
{
    pub id: TableId,
    // pub rows: Vec<TableRow<T, U, E>>,
    pub body: HashMap<T, U>,
    // pub db_ref: &'a MockDb<INDEX_BYTES, DATA_BYTES>,
}

impl<
        T,
        U,
        const SIZE_T: usize,
        const SIZE_U: usize,
        const INDEX_BYTES: usize,
        const DATA_BYTES: usize,
    > Table<T, U, SIZE_T, SIZE_U, INDEX_BYTES, DATA_BYTES>
where
    T: ToBytes + FromBytes<Bytes = [u8; SIZE_T]> + Sized + Hash + Eq + PartialEq + Clone,
    U: ToBytes + FromBytes<Bytes = [u8; SIZE_U]> + Sized + Clone,
{
    const SIZE_T: usize = std::mem::size_of::<T>();
    const SIZE_U: usize = std::mem::size_of::<U>();

    pub fn new() -> Self {
        Self {
            id: TableId::random(),
            body: HashMap::new(),
        }
    }

    // pub fn from_id(table_id: TableId) -> Self {
    // let db_table = db_ref.get_table(table_id);
    pub fn from_db_table(db_table: &MockDbTable<INDEX_BYTES, DATA_BYTES>) -> Self {
        let body = db_table
            .items
            .iter()
            .map(|(k, v)| {
                let codec = FixedBytesCodec::<T, U, SIZE_T, SIZE_U>::new(INDEX_BYTES, DATA_BYTES);
                let index = codec.fixed_bytes_to_index(k.to_vec());
                let data = codec.fixed_bytes_to_data(v.to_vec());
                (index, data)
            })
            .collect::<HashMap<T, U>>();

        Self {
            id: db_table.id,
            body,
        }
    }

    // pub fn new_with_rows(rows: Vec<TableRow<T, U, E>>) -> Self {
    //     Self::check_num_rows(rows.len());
    //     Self {
    //         id: TableId::random(),
    //         rows,
    //         db_ref: MockDbTable::new(),
    //     }
    // }

    // pub fn new_from_vecs(index: Vec<T>, data: Vec<U>) -> Self {
    //     if index.len() != data.len() {
    //         panic!("Index and data vectors must be the same length");
    //     }
    //     let rows = index
    //         .iter()
    //         .zip(data.iter())
    //         .map(|(index, data)| TableRow {
    //             index: index.clone(),
    //             data: data.clone(),
    //         })
    //         .collect();
    //     Self::new_with_rows(rows)
    // }

    // pub fn new_from_file(path: String) -> Self {
    //     let instructions = AfsInputInstructions::from_file(path);

    //     for op in &instructions.operations {
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

    //     Self::new()
    // }

    pub fn get_id(&self) -> TableId {
        self.id
    }

    pub fn get_id_hex(&self) -> String {
        "0x".to_string() + &self.id.to_string()
    }

    /// Reads directly from the table
    pub fn read(&self, index: T) -> Option<U> {
        self.body.get(&index).cloned()
    }

    /// Inserts into the table and the database
    // pub fn insert(&mut self, index: T, data: U) -> Result<()> {
    //     if (self.body.len() + 1) > MAX_OPS {
    //         return Err(eyre!(
    //             "Table size ({}) exceeds maximum number of operations ({})",
    //             self.body.len() + 1,
    //             MAX_OPS,
    //         ));
    //     }
    //     let i = self.find_index(index.clone());
    //     if let Some(_i) = i {
    //         Err(eyre!("Index already exists"))
    //     } else {
    //         self.index.push(index);
    //         self.data.push(data);
    //         Ok(())
    //     }
    // }

    // /// Writes to the table and the database
    // pub fn write(&mut self, index: T, data: U) -> Result<()> {
    //     if let Some(data_ref) = self.body.get_mut(&index) {
    //         *data_ref = data;
    //         Ok(())
    //     } else {
    //         Err(eyre!("Index not found"))
    //     }
    // }

    pub fn get_index_bytes(&self) -> usize {
        std::mem::size_of::<T>()
    }

    pub fn get_data_bytes(&self) -> usize {
        std::mem::size_of::<U>()
    }

    // fn check_num_rows(len: usize) {
    //     if len > MAX_OPS {
    //         panic!(
    //             "Table size ({}) exceeds maximum number of operations ({})",
    //             len, MAX_OPS
    //         );
    //     }
    // }
}

impl<
        T,
        U,
        const SIZE_T: usize,
        const SIZE_U: usize,
        const INDEX_BYTES: usize,
        const DATA_BYTES: usize,
    > Default for Table<T, U, SIZE_T, SIZE_U, INDEX_BYTES, DATA_BYTES>
where
    T: ToBytes + FromBytes<Bytes = [u8; SIZE_T]> + Sized + Hash + Eq + PartialEq + Clone,
    U: ToBytes + FromBytes<Bytes = [u8; SIZE_U]> + Sized + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}
