use crate::{
    afs_interface::utils::string_to_table_id, mock_db::MockDb, table::types::TableMetadata,
};

use super::AfsInterface;

#[test]
pub fn test_initialize_interface() {
    let default_table_metadata = TableMetadata::new(32, 1024);
    let mut db = MockDb::new(default_table_metadata);
    let mut _interface = AfsInterface::<u64, u64>::new(&mut db);
}

#[test]
pub fn test_initialize_interface_from_file() {
    let file_path = String::from("tests/data/test_input_file_8_8.afi");
    let default_table_metadata = TableMetadata::new(8, 8);
    let mut db = MockDb::new(default_table_metadata);
    let mut interface = AfsInterface::<u32, u64>::new(&mut db);
    interface.load_input_file(file_path).unwrap();
    let table = interface.current_table.unwrap();
    assert_eq!(
        table.id,
        string_to_table_id(String::from(
            "0xf221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c576"
        ))
    );
    assert_eq!(table.read(555).unwrap(), 1);
    assert_eq!(table.read(5006).unwrap(), 9);
    assert_eq!(table.read(26892).unwrap(), 5);
}
