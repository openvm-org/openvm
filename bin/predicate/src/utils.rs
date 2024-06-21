use logical_interface::{
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::{types::TableMetadata, Table},
    types::{Data, Index},
};

pub fn get_table_from_db<I: Index, D: Data>(
    table_id: String,
    db_file_path: Option<String>,
) -> Table<I, D> {
    let mut db = if let Some(db_file_path) = db_file_path {
        println!("db_file_path: {}", db_file_path);
        MockDb::from_file(&db_file_path)
    } else {
        let default_table_metadata = TableMetadata::new(32, 1024);
        MockDb::new(default_table_metadata)
    };

    let mut interface = AfsInterface::<I, D>::new(&mut db);
    interface.get_table(table_id).unwrap().clone()
}
