use alloy_primitives::FixedBytes;
use logical_interface::{
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::types::TableMetadata,
    utils::{string_to_u8_vec, uint_to_be_vec},
};

fn insert_data(interface: &mut AfsInterface, table_id: String, key: Vec<u8>, value: Vec<u8>) {
    let result = interface.insert(table_id, key, value);
    match result {
        Some(_) => (),
        None => panic!("Error inserting data"),
    }
}

#[test]
pub fn test_interface_mock_db() {
    let table_metadata = TableMetadata::new(32, 32);
    let index_bytes = 4;
    let data_bytes = 8;
    let mut mock_db = MockDb::new();
    let mut interface = AfsInterface::new(index_bytes, data_bytes, &mut mock_db);
    let table_id = String::from("0");
    let create = interface.create_table(table_id.clone(), table_metadata);
    assert!(create.is_some());
    insert_data(
        &mut interface,
        table_id.clone(),
        uint_to_be_vec(2, index_bytes),
        uint_to_be_vec(4, data_bytes),
    );
    insert_data(
        &mut interface,
        table_id,
        uint_to_be_vec(4, index_bytes),
        uint_to_be_vec(8, data_bytes),
    );
}

#[test]
pub fn test_interface_get_table() {
    let table_metadata = TableMetadata::new(32, 32);
    let index_bytes = 4;
    let data_bytes = 8;
    let mut mock_db = MockDb::new();
    let mut interface = AfsInterface::new(index_bytes, data_bytes, &mut mock_db);
    let table_id = String::from("10");
    let create = interface.create_table(table_id.clone(), table_metadata);
    assert!(create.is_some());
    insert_data(
        &mut interface,
        table_id.clone(),
        uint_to_be_vec(2, index_bytes),
        uint_to_be_vec(4, data_bytes),
    );
    insert_data(
        &mut interface,
        table_id.clone(),
        uint_to_be_vec(4, index_bytes),
        uint_to_be_vec(8, data_bytes),
    );
    let table = interface.get_table(table_id).expect("Error getting table");
    let v0 = table.read(uint_to_be_vec(2, index_bytes));
    assert_eq!(v0, Some(uint_to_be_vec(4, data_bytes)));
    let v1 = table.read(uint_to_be_vec(4, index_bytes));
    assert_eq!(v1, Some(uint_to_be_vec(8, data_bytes)));
}

#[test]
pub fn test_interface_large_values() {
    let table_metadata = TableMetadata::new(32, 1024);
    let index_bytes = 32;
    let data_bytes = 32;
    let mut mock_db = MockDb::new();
    let mut interface = AfsInterface::new(index_bytes, data_bytes, &mut mock_db);
    let table_id = String::from("0x1234");
    let create = interface.create_table(table_id.clone(), table_metadata);
    assert!(create.is_some());
    insert_data(
        &mut interface,
        table_id.clone(),
        string_to_u8_vec(
            "0xf221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c576".to_string(),
            index_bytes,
        ),
        uint_to_be_vec(500, data_bytes),
    );
    insert_data(
        &mut interface,
        table_id.clone(),
        uint_to_be_vec(1000, index_bytes),
        string_to_u8_vec(
            "0x1f221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c5".to_string(),
            data_bytes,
        ),
    );
    let table = interface
        .get_table(table_id.clone())
        .expect("Error getting table");
    let read0 = table.read(string_to_u8_vec(
        "0xf221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c576".to_string(),
        index_bytes,
    ));
    assert_eq!(read0, Some(uint_to_be_vec(500, data_bytes)));
    let read1 = table.read(uint_to_be_vec(1000, index_bytes));
    assert_eq!(
        read1,
        Some(string_to_u8_vec(
            "0x1f221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c5".to_string(),
            data_bytes,
        ))
    );

    let res2 = interface.write(
        table_id.clone(),
        uint_to_be_vec(1000, index_bytes),
        uint_to_be_vec(2000, data_bytes),
    );
    assert_eq!(res2, Some(()));
    let table = interface.get_table(table_id).expect("Error getting table");
    let read2 = table.read(uint_to_be_vec(1000, index_bytes));
    assert_eq!(read2, Some(uint_to_be_vec(2000, data_bytes)));
}

#[test]
pub fn test_interface_large_tables() {
    let table_metadata = TableMetadata::new(32, 1024);
    let index_bytes = 32;
    let data_bytes = 32;
    let mut mock_db = MockDb::new();
    let mut interface = AfsInterface::new(index_bytes, data_bytes, &mut mock_db);

    for table_id in 0..10 {
        let create = interface.create_table(table_id.to_string(), table_metadata.clone());
        assert!(create.is_some());
        for i in 0..128 {
            let value: Vec<u8> = FixedBytes::<32>::random().to_vec();
            insert_data(
                &mut interface,
                table_id.to_string(),
                uint_to_be_vec(i, index_bytes),
                value,
            );
        }
    }

    // mock_db.save_to_file("tests/data/afs_db.mockdb").unwrap();
}

#[test]
#[should_panic]
pub fn test_table_input_too_large() {
    let table_metadata = TableMetadata::new(2, 1024);
    let index_bytes = 8;
    let data_bytes = 8;
    let mut mock_db = MockDb::new();
    let mut interface = AfsInterface::new(index_bytes, data_bytes, &mut mock_db);
    let table_id = String::from("0x01");
    let _create = interface.create_table(table_id.clone(), table_metadata);
    insert_data(
        &mut interface,
        table_id.clone(),
        uint_to_be_vec(1, index_bytes),
        uint_to_be_vec(1, data_bytes),
    );
}

#[test]
pub fn test_vec_index() {
    let table_metadata = TableMetadata::new(32, 1024);
    let index_bytes = 8;
    let data_bytes = 32;
    let mut mock_db = MockDb::new();
    let idx0 = Vec::from([1; 8]);
    let idx1 = Vec::from([2; 8]);
    let mut interface = AfsInterface::new(index_bytes, data_bytes, &mut mock_db);
    let table_id = String::from("0x100000000");
    let create = interface.create_table(table_id.clone(), table_metadata);
    assert!(create.is_some());
    insert_data(
        &mut interface,
        table_id.clone(),
        idx0.clone(),
        uint_to_be_vec(1, data_bytes),
    );
    insert_data(
        &mut interface,
        table_id.clone(),
        idx1.clone(),
        uint_to_be_vec(2, data_bytes),
    );
    let table = interface.get_table(table_id).expect("Error getting table");
    let v0 = table.read(idx0);
    assert_eq!(v0, Some(uint_to_be_vec(1, data_bytes)));
    let v1 = table.read(idx1);
    assert_eq!(v1, Some(uint_to_be_vec(2, data_bytes)));
}

#[test]
pub fn test_vec_data() {
    let table_metadata = TableMetadata::new(32, 1024);
    let index_bytes = 32;
    let data_bytes = 1024;
    let mut mock_db = MockDb::new();
    let data0 = Vec::from([1; 1024]);
    let data1 = Vec::from([2; 1024]);
    let mut interface = AfsInterface::new(index_bytes, data_bytes, &mut mock_db);
    let table_id = String::from("0xffaaccee");
    let create = interface.create_table(table_id.clone(), table_metadata);
    assert!(create.is_some());
    insert_data(
        &mut interface,
        table_id.clone(),
        uint_to_be_vec(1, index_bytes),
        data0.clone(),
    );
    insert_data(
        &mut interface,
        table_id.clone(),
        uint_to_be_vec(2, index_bytes),
        data1.clone(),
    );
    let table = interface.get_table(table_id).expect("Error getting table");
    let v0 = table.read(uint_to_be_vec(1, index_bytes));
    assert_eq!(v0, Some(data0));
    let v1 = table.read(uint_to_be_vec(2, index_bytes));
    assert_eq!(v1, Some(data1));
}
