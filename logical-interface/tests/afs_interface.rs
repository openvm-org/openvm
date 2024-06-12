use alloy_primitives::U256;
use logical_interface::{
    afs_interface::AfsInterface,
    mock_db::MockDb,
    table::types::TableMetadata,
    types::{Data, Index},
};
use std::str::FromStr;

fn insert_data<I: Index, D: Data>(
    interface: &mut AfsInterface<I, D>,
    table_id: String,
    key: I,
    value: D,
) {
    let result = interface.insert(table_id, key, value);
    match result {
        Some(_) => (),
        None => panic!("Error inserting data"),
    }
}

#[test]
pub fn test_interface_mock_db() {
    let default_table_metadata = TableMetadata::new(32, 32);
    let mut mock_db = MockDb::new(default_table_metadata.clone());
    let mut interface = AfsInterface::<u32, u64>::new(&mut mock_db);
    let table_id = String::from("0");
    let create = interface.create_table(table_id.clone(), default_table_metadata);
    assert!(create.is_some());
    insert_data::<u32, u64>(&mut interface, table_id.clone(), 2, 4);
    insert_data::<u32, u64>(&mut interface, table_id, 4, 8);
}

#[test]
pub fn test_interface_get_table() {
    let default_table_metadata = TableMetadata::new(32, 32);
    let mut mock_db = MockDb::new(default_table_metadata.clone());
    let mut interface = AfsInterface::<u32, u64>::new(&mut mock_db);
    let table_id = String::from("10");
    let create = interface.create_table(table_id.clone(), default_table_metadata);
    assert!(create.is_some());
    insert_data::<u32, u64>(&mut interface, table_id.clone(), 2, 4);
    insert_data::<u32, u64>(&mut interface, table_id.clone(), 4, 8);
    let table = interface.get_table(table_id).expect("Error getting table");
    let v0 = table.read(2);
    assert_eq!(v0, Some(4));
    let v1 = table.read(4);
    assert_eq!(v1, Some(8));
}

#[test]
pub fn test_interface_large_table() {
    let default_table_metadata = TableMetadata::new(32, 1024);
    let mut mock_db = MockDb::new(default_table_metadata.clone());
    let mut interface = AfsInterface::<U256, U256>::new(&mut mock_db);
    let table_id = String::from("0x1234");
    let create = interface.create_table(table_id.clone(), default_table_metadata);
    assert!(create.is_some());
    insert_data::<U256, U256>(
        &mut interface,
        table_id.clone(),
        U256::from_str("0xf221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c576")
            .unwrap(),
        U256::from(500),
    );
    insert_data::<U256, U256>(
        &mut interface,
        table_id.clone(),
        U256::from(1000),
        U256::from_str("0x1f221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c5")
            .unwrap(),
    );
    let table = interface
        .get_table(table_id.clone())
        .expect("Error getting table");
    let read0 = table.read(
        U256::from_str("0xf221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c576")
            .unwrap(),
    );
    assert_eq!(read0, Some(U256::from(500)));
    let read1 = table.read(U256::from(1000));
    assert_eq!(
        read1,
        Some(
            U256::from_str("0x1f221eb52f500a1db8bf0de52d2f2da5d208498b03cef6597be489c2207e1c5")
                .unwrap()
        )
    );

    let res2 = interface.write(table_id.clone(), U256::from(1000), U256::from(2000));
    assert_eq!(res2, Some(()));
    let table = interface.get_table(table_id).expect("Error getting table");
    let read2 = table.read(U256::from(1000));
    assert_eq!(read2, Some(U256::from(2000)));
}

#[test]
#[should_panic]
pub fn test_table_input_too_large() {
    let default_table_metadata = TableMetadata::new(2, 1024);
    let mut mock_db = MockDb::new(default_table_metadata.clone());
    let mut interface = AfsInterface::<u32, u32>::new(&mut mock_db);
    let table_id = String::from("0x01");
    let _create = interface.create_table(table_id.clone(), default_table_metadata);
    insert_data::<u32, u32>(&mut interface, table_id, 1, 1);
}

#[test]
pub fn test_vec_index() {
    let default_table_metadata = TableMetadata::new(32, 1024);
    let mut mock_db = MockDb::new(default_table_metadata.clone());
    let mut interface = AfsInterface::<[u8; 8], U256>::new(&mut mock_db);
    let table_id = String::from("0x100000000");
    let create = interface.create_table(table_id.clone(), default_table_metadata);
    assert!(create.is_some());
    insert_data::<[u8; 8], U256>(&mut interface, table_id.clone(), [1; 8], U256::from(1));
    insert_data::<[u8; 8], U256>(&mut interface, table_id.clone(), [2; 8], U256::from(2));
    let table = interface.get_table(table_id).expect("Error getting table");
    let v0 = table.read([1; 8]);
    assert_eq!(v0, Some(U256::from(1)));
    let v1 = table.read([2; 8]);
    assert_eq!(v1, Some(U256::from(2)));
}

#[test]
pub fn test_vec_data() {
    let default_table_metadata = TableMetadata::new(32, 1024);
    let mut mock_db = MockDb::new(default_table_metadata.clone());
    let mut interface = AfsInterface::<U256, [u32; 8]>::new(&mut mock_db);
    let table_id = String::from("0xffaaccee");
    let create = interface.create_table(table_id.clone(), default_table_metadata);
    assert!(create.is_some());
    insert_data::<U256, [u32; 8]>(&mut interface, table_id.clone(), U256::from(1), [1; 8]);
    insert_data::<U256, [u32; 8]>(&mut interface, table_id.clone(), U256::from(2), [2; 8]);
    let table = interface.get_table(table_id).expect("Error getting table");
    let v0 = table.read(U256::from(1));
    assert_eq!(v0, Some([1; 8]));
    let v1 = table.read(U256::from(2));
    assert_eq!(v1, Some([2; 8]));
}
