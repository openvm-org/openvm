use crate::{
    afs_interface::utils::string_to_table_id,
    table::types::{TableId, TableMetadata},
};

use super::Table;

#[test]
pub fn test_create_new_table() {
    let table_id = TableId::new([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ]);
    let table = Table::<u32, u64>::new(table_id, TableMetadata::new(4, 8));
    assert_eq!(table.id, string_to_table_id("1".to_string()));
}
