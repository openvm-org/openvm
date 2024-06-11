#[cfg(test)]
pub mod tests;
use crate::{
    mock_db::MockDb,
    table::{codec::fixed_bytes::FixedBytesCodec, types::TableId, Table},
    types::{Data, Index},
};

pub struct AfsInterface<'a, I: Index, D: Data> {
    db_ref: &'a mut MockDb,
    current_table: Option<Table<I, D>>,
}

impl<'a, I: Index, D: Data> AfsInterface<'a, I, D> {
    const SIZE_I: usize = std::mem::size_of::<I>();
    const SIZE_D: usize = std::mem::size_of::<D>();

    pub fn new(db_ref: &'a mut MockDb) -> Self {
        Self {
            db_ref,
            current_table: None,
        }
    }

    // pub fn load_afi(path: String) -> Self {
    //     let instructions = AfsInputInstructions::from_file(path);

    //     for op in &instructions.operations {
    //         match op.operation {
    //             InputFileBodyOperation::Read => {}
    //             InputFileBodyOperation::Insert | InputFileBodyOperation::Write => {
    //                 let index_input = op.args[0].clone();
    //                 let index = Vec::<u8>::from(string_to_fixed_bytes_be_vec(index_input));
    //                 let data_input = op.args[1].clone();
    //                 let data =
    //                     Vec::<u8>::from(string_to_fixed_bytes_be_vec::<DATA_BYTES>(data_input));
    //                 map.insert(index, data);
    //             }
    //         };
    //     }
    // }

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
