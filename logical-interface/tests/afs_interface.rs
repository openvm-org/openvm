use logical_interface::{
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::types::{TableId, TableMetadata},
};

fn insert_data(interface: &mut AfsInterface<u32, u64>, table_id: TableId, key: u32, value: u64) {
    let result = interface.insert(table_id, key, value);
    match result {
        Some(_) => (),
        None => panic!("Error inserting data"),
    }
}

#[test]
pub fn test_interface_mock_db() {
    let metadata = TableMetadata::new(32, 32);
    let mut mock_db = MockDb::new(metadata.clone());
    let mut interface = AfsInterface::<u32, u64>::new(&mut mock_db);
    let table_id = TableId::ZERO;
    interface.get_db_ref().create_table(table_id, metadata);
    insert_data(&mut interface, table_id, 2, 4);
    insert_data(&mut interface, table_id, 4, 8);
}

#[test]
pub fn test_interface_get_table() {
    let metadata = TableMetadata::new(32, 32);
    let mut mock_db = MockDb::new(metadata.clone());
    let mut interface = AfsInterface::<u32, u64>::new(&mut mock_db);
    let table_id = TableId::ZERO;
    interface.get_db_ref().create_table(table_id, metadata);
    insert_data(&mut interface, table_id, 2, 4);
    insert_data(&mut interface, table_id, 4, 8);
    let table = interface.get_table(table_id).expect("Error getting table");
    let v0 = table.read(2);
    assert_eq!(v0, Some(4));
    let v1 = table.read(4);
    assert_eq!(v1, Some(8));
}
