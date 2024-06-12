#[cfg(test)]
pub mod tests;
pub mod utils;

use crate::{
    afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions},
    mock_db::MockDb,
    table::{codec::fixed_bytes::FixedBytesCodec, types::TableMetadata, Table},
    types::{Data, Index},
    utils::string_to_fixed_bytes_be_vec,
};
use color_eyre::eyre::{eyre, Result};
use utils::string_to_table_id;

pub struct AfsInterface<'a, I: Index, D: Data> {
    /// Reference to the mock database
    db_ref: &'a mut MockDb,
    /// Stores current table in memory for faster reads
    current_table: Option<Table<I, D>>,
}

impl<'a, I: Index, D: Data> AfsInterface<'a, I, D> {
    pub fn new(db_ref: &'a mut MockDb) -> Self {
        Self {
            db_ref,
            current_table: None,
        }
    }

    pub fn load_input_file(&mut self, path: String) -> Result<&Table<I, D>> {
        let instructions = AfsInputInstructions::from_file(path)?;

        let table_id = instructions.header.table_id;
        let table_id_bytes = string_to_table_id(table_id.clone());

        for op in &instructions.operations {
            match op.operation {
                InputFileBodyOperation::Read => {}
                InputFileBodyOperation::Insert => {
                    if op.args.len() != 2 {
                        return Err(eyre!("Invalid number of arguments for insert operation"));
                    }
                    let index_input = op.args[0].clone();
                    let index =
                        string_to_fixed_bytes_be_vec(index_input, instructions.header.index_bytes);
                    let data_input = op.args[1].clone();
                    let data =
                        string_to_fixed_bytes_be_vec(data_input, instructions.header.data_bytes);
                    let table = self.db_ref.get_table(table_id_bytes);
                    if table.is_none() {
                        self.db_ref.create_table(
                            table_id_bytes,
                            TableMetadata::new(
                                instructions.header.index_bytes,
                                instructions.header.data_bytes,
                            ),
                        );
                    }
                    self.db_ref.insert_data(table_id_bytes, index, data);
                }
                InputFileBodyOperation::Write => {
                    if op.args.len() != 2 {
                        return Err(eyre!("Invalid number of arguments for write operation"));
                    }
                    let index_input = op.args[0].clone();
                    let index =
                        string_to_fixed_bytes_be_vec(index_input, instructions.header.index_bytes);
                    let data_input = op.args[1].clone();
                    let data =
                        string_to_fixed_bytes_be_vec(data_input, instructions.header.data_bytes);
                    let table = self.db_ref.get_table(table_id_bytes);
                    if table.is_none() {
                        self.db_ref.create_table(
                            table_id_bytes,
                            TableMetadata::new(
                                instructions.header.index_bytes,
                                instructions.header.data_bytes,
                            ),
                        );
                    }
                    self.db_ref.write_data(table_id_bytes, index, data);
                }
            };
        }

        let get_table = self.get_table(table_id);
        match get_table {
            Some(table) => Ok(table),
            None => Err(eyre!("Error getting table")),
        }
    }

    pub fn get_db_ref(&mut self) -> &mut MockDb {
        self.db_ref
    }

    pub fn get_current_table(&self) -> Option<&Table<I, D>> {
        self.current_table.as_ref()
    }

    pub fn create_table(&mut self, table_id: String, metadata: TableMetadata) -> Option<()> {
        let table_id_bytes = string_to_table_id(table_id);
        self.db_ref.create_table(table_id_bytes, metadata)
    }

    pub fn get_table(&mut self, table_id: String) -> Option<&Table<I, D>> {
        let table_id_bytes = string_to_table_id(table_id);
        let db_table = self.db_ref.get_table(table_id_bytes)?;
        self.current_table = Some(Table::from_db_table(db_table));
        self.current_table.as_ref()
    }

    pub fn read(&mut self, table_id: String, index: I) -> Option<D> {
        if let Some(table) = self.current_table.as_ref() {
            let id = table.id;
            let table_id_bytes = string_to_table_id(table_id.clone());
            if id != table_id_bytes {
                self.get_table(table_id);
            }
        } else {
            self.get_table(table_id);
        }
        self.current_table.as_ref().unwrap().read(index)
    }

    pub fn insert(&mut self, table_id: String, index: I, data: D) -> Option<()> {
        let table_id_bytes = string_to_table_id(table_id);
        let metadata = self.db_ref.get_table_metadata(table_id_bytes)?;
        let codec = FixedBytesCodec::<I, D>::new(metadata.index_bytes, metadata.data_bytes);
        let index_bytes = codec.index_to_fixed_bytes(index);
        let data_bytes = codec.data_to_fixed_bytes(data);
        self.db_ref
            .insert_data(table_id_bytes, index_bytes, data_bytes)?;
        Some(())
    }

    pub fn write(&mut self, table_id: String, index: I, data: D) -> Option<()> {
        let table_id_bytes = string_to_table_id(table_id);
        let metadata = self.db_ref.get_table_metadata(table_id_bytes)?;
        let codec = FixedBytesCodec::<I, D>::new(metadata.index_bytes, metadata.data_bytes);
        let index_bytes = codec.index_to_fixed_bytes(index);
        let data_bytes = codec.data_to_fixed_bytes(data);
        self.db_ref
            .write_data(table_id_bytes, index_bytes, data_bytes)?;
        Some(())
    }
}
