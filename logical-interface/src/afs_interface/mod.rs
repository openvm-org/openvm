#[cfg(test)]
pub mod tests;
use color_eyre::eyre::{eyre, Result};

use crate::{
    afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions},
    mock_db::MockDb,
    table::{
        codec::fixed_bytes::FixedBytesCodec,
        types::{string_to_table_id, TableId, TableMetadata},
        Table,
    },
    types::{Data, Index},
    utils::string_to_fixed_bytes_be_vec,
};

pub struct AfsInterface<'a, I: Index, D: Data> {
    db_ref: &'a mut MockDb,
    current_table: Option<Table<I, D>>,
}

impl<'a, I: Index, D: Data> AfsInterface<'a, I, D> {
    pub fn new(db_ref: &'a mut MockDb) -> Self {
        Self {
            db_ref,
            current_table: None,
        }
    }

    pub fn load_input_file(&mut self, path: String) -> Result<()> {
        let instructions = AfsInputInstructions::from_file(path)?;

        let table_id = string_to_table_id(instructions.header.table_id);

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
                    let table = self.db_ref.get_table(table_id);
                    if table.is_none() {
                        self.db_ref.create_table(
                            table_id,
                            TableMetadata::new(
                                instructions.header.index_bytes,
                                instructions.header.data_bytes,
                            ),
                        );
                    }
                    self.db_ref.insert_data(table_id, index, data);
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
                    let table = self.db_ref.get_table(table_id);
                    if table.is_none() {
                        self.db_ref.create_table(
                            table_id,
                            TableMetadata::new(
                                instructions.header.index_bytes,
                                instructions.header.data_bytes,
                            ),
                        );
                    }
                    self.db_ref.write_data(table_id, index, data);
                }
            };
        }

        let get = self.get_table(table_id);
        if get.is_none() {
            return Err(eyre!("Error getting table"));
        }

        Ok(())
    }

    pub fn get_db_ref(&mut self) -> &mut MockDb {
        self.db_ref
    }

    pub fn get_current_table(&self) -> Option<&Table<I, D>> {
        self.current_table.as_ref()
    }

    pub fn get_table(&mut self, table_id: TableId) -> Option<&Table<I, D>> {
        let db_table = self.db_ref.get_table(table_id)?;
        self.current_table = Some(Table::from_db_table(db_table));
        self.current_table.as_ref()
    }

    pub fn read(&mut self, table_id: TableId, index: I) -> Option<D> {
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

    pub fn insert(&mut self, table_id: TableId, index: I, data: D) -> Option<()> {
        let metadata = self.db_ref.get_table_metadata(table_id)?;
        let codec = FixedBytesCodec::<I, D>::new(metadata.index_bytes, metadata.data_bytes);
        let index_bytes = codec.index_to_fixed_bytes(index);
        let data_bytes = codec.data_to_fixed_bytes(data);
        self.db_ref.insert_data(table_id, index_bytes, data_bytes)?;
        Some(())
    }

    pub fn write(&mut self, table_id: TableId, index: I, data: D) -> Option<()> {
        let metadata = self.db_ref.get_table_metadata(table_id)?;
        let codec = FixedBytesCodec::<I, D>::new(metadata.index_bytes, metadata.data_bytes);
        let index_bytes = codec.index_to_fixed_bytes(index);
        let data_bytes = codec.data_to_fixed_bytes(data);
        self.db_ref.write_data(table_id, index_bytes, data_bytes)?;
        Some(())
    }
}
