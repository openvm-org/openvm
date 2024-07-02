#[cfg(test)]
pub mod tests;
pub mod utils;

use crate::{
    afs_input::{types::InputFileOp, AfsInputFile},
    mock_db::MockDb,
    table::{codec::fixed_bytes::FixedBytesCodec, types::TableMetadata, Table},
    utils::string_to_u8_vec,
};
use color_eyre::eyre::{eyre, Result};
use utils::string_to_table_id;

pub struct AfsInterface<'a> {
    /// Number of bytes for the index
    index_bytes: usize,
    /// Number of bytes for the data
    data_bytes: usize,
    /// Reference to the mock database
    db_ref: &'a mut MockDb,
    /// Stores current table in memory for faster reads
    current_table: Option<Table>,
}

impl<'a> AfsInterface<'a> {
    pub fn new(index_bytes: usize, data_bytes: usize, db_ref: &'a mut MockDb) -> Self {
        Self {
            index_bytes,
            data_bytes,
            db_ref,
            current_table: None,
        }
    }

    /// Gets a table from the DB and creates a new AfsInterface with its index and data byte lengths.
    /// The table is then stored in current_table for easy access.
    pub fn new_with_table(table_id: String, db_ref: &'a mut MockDb) -> Self {
        let table_id_bytes = string_to_table_id(table_id);
        let table = db_ref.get_table(table_id_bytes).unwrap();
        let index_bytes = table.db_table_metadata.index_bytes;
        let data_bytes = table.db_table_metadata.data_bytes;
        let table = Table::from_db_table(table, index_bytes, data_bytes);
        Self {
            index_bytes,
            data_bytes,
            db_ref,
            current_table: Some(table),
        }
    }

    pub fn load_input_file(&mut self, path: &str) -> Result<&Table> {
        let instructions = AfsInputFile::open(path)?;

        let table_id = instructions.header.table_id;
        let table_id_bytes = string_to_table_id(table_id.clone());

        let get_table = self.get_table(table_id.clone());
        match get_table {
            Some(_) => {}
            None => {
                self.create_table(
                    table_id.clone(),
                    TableMetadata::new(
                        instructions.header.index_bytes,
                        instructions.header.data_bytes,
                    ),
                )
                .unwrap();
                self.get_table(table_id.clone()).unwrap();
            }
        }

        for op in &instructions.operations {
            match op.operation {
                InputFileOp::Read => {}
                InputFileOp::Insert => {
                    if op.args.len() != 2 {
                        return Err(eyre!("Invalid number of arguments for insert operation"));
                    }
                    let index_input = op.args[0].clone();
                    let index = string_to_u8_vec(index_input, instructions.header.index_bytes);
                    let data_input = op.args[1].clone();
                    let data = string_to_u8_vec(data_input, instructions.header.data_bytes);
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
                InputFileOp::Write => {
                    if op.args.len() != 2 {
                        return Err(eyre!("Invalid number of arguments for write operation"));
                    }
                    let index_input = op.args[0].clone();
                    let index = string_to_u8_vec(index_input, instructions.header.index_bytes);
                    let data_input = op.args[1].clone();
                    let data = string_to_u8_vec(data_input, instructions.header.data_bytes);
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
                InputFileOp::Where => {}
                InputFileOp::InnerJoin => {}
                InputFileOp::GroupBy => {}
            };
        }

        Ok(self.get_table(table_id).unwrap())
    }

    pub fn get_db_ref(&mut self) -> &mut MockDb {
        self.db_ref
    }

    pub fn current_table(&self) -> Option<&Table> {
        self.current_table.as_ref()
    }

    pub fn create_table(&mut self, table_id: String, metadata: TableMetadata) -> Option<()> {
        let table_id_bytes = string_to_table_id(table_id);
        self.db_ref.create_table(table_id_bytes, metadata)
    }

    pub fn get_table(&mut self, table_id: String) -> Option<&Table> {
        let table_id_bytes = string_to_table_id(table_id);
        let db_table = self.db_ref.get_table(table_id_bytes)?;
        self.current_table = Some(Table::from_db_table(
            db_table,
            self.index_bytes,
            self.data_bytes,
        ));
        self.current_table.as_ref()
    }

    pub fn read(&mut self, table_id: String, index: Vec<u8>) -> Option<Vec<u8>> {
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

    pub fn insert(&mut self, table_id: String, index: Vec<u8>, data: Vec<u8>) -> Option<()> {
        let table_id_bytes = string_to_table_id(table_id);
        let db_table_metadata = self.db_ref.get_table_metadata(table_id_bytes)?;
        let codec = FixedBytesCodec::new(
            index.len(),
            data.len(),
            db_table_metadata.index_bytes,
            db_table_metadata.data_bytes,
        );
        let index_bytes = codec.table_to_db_index_bytes(index);
        let data_bytes = codec.table_to_db_data_bytes(data);
        self.db_ref
            .insert_data(table_id_bytes, index_bytes, data_bytes)?;
        Some(())
    }

    pub fn write(&mut self, table_id: String, index: Vec<u8>, data: Vec<u8>) -> Option<()> {
        let table_id_bytes = string_to_table_id(table_id);
        let db_table_metadata = self.db_ref.get_table_metadata(table_id_bytes)?;
        let codec = FixedBytesCodec::new(
            index.len(),
            data.len(),
            db_table_metadata.index_bytes,
            db_table_metadata.data_bytes,
        );
        let index_bytes = codec.table_to_db_index_bytes(index);
        let data_bytes = codec.table_to_db_data_bytes(data);
        self.db_ref
            .write_data(table_id_bytes, index_bytes, data_bytes)?;
        Some(())
    }
}
