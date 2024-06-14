use crate::{
    afs_interface::utils::string_to_table_id,
    table::types::{TableId, TableMetadata},
};

use super::Table;

fn create_table() -> Table<u32, u64> {
    let table_id = TableId::new([0; 32]);
    let mut table = Table::<u32, u64>::new(table_id, TableMetadata::new(4, 8));
    table.body.insert(1, 2);
    table.body.insert(2, 4);
    table.body.insert(4, 8);
    table.body.insert(8, 16);
    table.body.insert(16, 32);
    table.body.insert(32, 64);
    table.body.insert(64, 128);
    table.body.insert(128, 256);
    table.body.insert(1000, 65536);
    table.body.insert(2000, 65792);
    table
}

#[test]
pub fn test_create_new_table() {
    let table_id = TableId::new([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ]);
    let table = Table::<u32, u64>::new(table_id, TableMetadata::new(4, 8));
    assert_eq!(table.id, string_to_table_id("1".to_string()));
}

#[test]
pub fn test_convert_to_page() {
    let page_size = 16;
    let table = create_table();
    let page = table.to_page(page_size);
    assert_eq!(page.len(), page_size);
    for (i, row) in page.iter().enumerate() {
        println!("{:?}", row);
        assert_eq!(row[0], if i < table.body.len() { 1 } else { 0 });
    }
}

#[test]
#[should_panic]
pub fn test_convert_to_page_too_small() {
    let table = create_table();
    table.to_page(4);
}
