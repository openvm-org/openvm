use logical_interface::{afs_interface::AfsInterface, mock_db::MockDb, table::TableId};

#[test]
pub fn test_interface_mock_db() {
    let mut mock_db = MockDb::<32, 32>::new();
    let mut interface = AfsInterface::<u32, u64, 4, 8, 32, 32>::new(&mut mock_db);
    let table_id = TableId::ZERO;
    let i0 = interface.insert(table_id, 2, 4);
    match i0 {
        Some(_) => (),
        None => panic!("Error inserting data"),
    }
    let i1 = interface.insert(table_id, 4, 8);
    match i1 {
        Some(_) => (),
        None => panic!("Error inserting data"),
    }
}
