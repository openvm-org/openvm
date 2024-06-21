use logical_interface::{afs_interface::AfsInterface, mock_db::MockDb, table::Table};

pub fn get_table_from_db(
    table_id: String,
    db_file_path: Option<String>,
    index_bytes: usize,
    data_bytes: usize,
) -> Table {
    let mut db = if let Some(db_file_path) = db_file_path {
        println!("db_file_path: {}", db_file_path);
        MockDb::from_file(&db_file_path)
    } else {
        panic!("Table does not exist");
    };

    let mut interface = AfsInterface::new(index_bytes, data_bytes, &mut db);
    interface.get_table(table_id).unwrap().clone()
}
