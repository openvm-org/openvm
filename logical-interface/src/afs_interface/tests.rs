use crate::{
    afs_interface::utils::string_to_table_id,
    mock_db::MockDb,
    table::types::TableMetadata,
    utils::{string_to_u8_vec, uint_to_be_vec},
};

use super::AfsInterface;

#[test]
pub fn test_initialize_interface() {
    let default_table_metadata = TableMetadata::new(32, 1024);
    let mut db = MockDb::new(default_table_metadata);
    let mut _interface = AfsInterface::new(8, 8, &mut db);
}

#[test]
pub fn test_initialize_interface_from_file() {
    let file_path = "tests/data/test_input_file_8_8.afi";
    let default_table_metadata = TableMetadata::new(8, 8);
    let mut db = MockDb::new(default_table_metadata);
    let mut interface = AfsInterface::new(4, 8, &mut db);
    match interface.load_input_file(file_path) {
        Ok(_) => {}
        Err(e) => panic!("Error loading input file: {}", e),
    }
    let table = interface.current_table.unwrap();
    assert_eq!(
        table.id,
        string_to_table_id(String::from(
            "0xf221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c576"
        ))
    );
    assert_eq!(
        table
            .read(string_to_u8_vec(String::from("555"), 4))
            .unwrap(),
        uint_to_be_vec(1, 8)
    );
    assert_eq!(
        table
            .read(string_to_u8_vec(String::from("5006"), 4))
            .unwrap(),
        uint_to_be_vec(9, 8)
    );
    assert_eq!(
        table
            .read(string_to_u8_vec(String::from("26892"), 4))
            .unwrap(),
        uint_to_be_vec(5, 8)
    );
}
