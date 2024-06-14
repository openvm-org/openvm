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

fn create_page() -> Vec<Vec<u32>> {
    vec![
        vec![1, 0, 1, 0, 0, 0, 2],
        vec![1, 0, 2, 0, 0, 0, 4],
        vec![1, 0, 4, 0, 0, 0, 8],
        vec![1, 0, 8, 0, 0, 0, 16],
        vec![1, 0, 16, 0, 0, 0, 32],
        vec![1, 0, 32, 0, 0, 0, 64],
        vec![1, 0, 64, 0, 0, 0, 128],
        vec![1, 0, 128, 0, 0, 0, 256],
        vec![1, 0, 1000, 0, 0, 1, 0],
        vec![1, 0, 2000, 0, 0, 1, 256],
        vec![0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0],
        vec![0, 0, 0, 0, 0, 0, 0],
    ]
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
    }
    let comparison_page = create_page();
    assert_eq!(page, comparison_page);
}

#[test]
#[should_panic]
pub fn test_convert_to_page_too_small() {
    let table = create_table();
    table.to_page(4);
}

#[test]
pub fn test_convert_from_page() {
    let page = create_page();
    let table = Table::<u32, u64>::from_page(TableId::new([1; 32]), page);
    println!("{:?}", table.body);
    let comparison_table = create_table();
    assert_eq!(table.body, comparison_table.body);
}
