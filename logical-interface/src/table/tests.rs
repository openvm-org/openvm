use afs_chips::common::{page::Page, page_cols::PageCols};

use crate::{
    afs_interface::utils::string_to_table_id,
    table::types::{TableId, TableMetadata},
    utils::uint_to_be_vec,
};

use super::Table;

fn create_table() -> Table {
    let table_id = TableId::new([0; 32]);
    let index_bytes = 4;
    let data_bytes = 8;
    let mut table = Table::new(table_id, TableMetadata::new(index_bytes, data_bytes));
    table.body.insert(
        uint_to_be_vec(1, index_bytes),
        uint_to_be_vec(2, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(2, index_bytes),
        uint_to_be_vec(4, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(4, index_bytes),
        uint_to_be_vec(8, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(8, index_bytes),
        uint_to_be_vec(16, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(16, index_bytes),
        uint_to_be_vec(32, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(32, index_bytes),
        uint_to_be_vec(64, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(64, index_bytes),
        uint_to_be_vec(128, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(128, index_bytes),
        uint_to_be_vec(256, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(1000, index_bytes),
        uint_to_be_vec(65536, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(65791, index_bytes),
        uint_to_be_vec(65792, data_bytes),
    );
    table
}

fn create_page() -> Page {
    Page::from_page_cols(vec![
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 1],
            data: vec![0, 0, 0, 2],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 2],
            data: vec![0, 0, 0, 4],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 4],
            data: vec![0, 0, 0, 8],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 8],
            data: vec![0, 0, 0, 16],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 16],
            data: vec![0, 0, 0, 32],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 32],
            data: vec![0, 0, 0, 64],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 64],
            data: vec![0, 0, 0, 128],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 128],
            data: vec![0, 0, 0, 256],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![0, 1000],
            data: vec![0, 0, 1, 0],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![1, 255],
            data: vec![0, 0, 1, 256],
        },
        PageCols::<u32> {
            is_alloc: 0,
            idx: vec![0, 0],
            data: vec![0, 0, 0, 0],
        },
        PageCols::<u32> {
            is_alloc: 0,
            idx: vec![0, 0],
            data: vec![0, 0, 0, 0],
        },
        PageCols::<u32> {
            is_alloc: 0,
            idx: vec![0, 0],
            data: vec![0, 0, 0, 0],
        },
        PageCols::<u32> {
            is_alloc: 0,
            idx: vec![0, 0],
            data: vec![0, 0, 0, 0],
        },
        PageCols::<u32> {
            is_alloc: 0,
            idx: vec![0, 0],
            data: vec![0, 0, 0, 0],
        },
        PageCols::<u32> {
            is_alloc: 0,
            idx: vec![0, 0],
            data: vec![0, 0, 0, 0],
        },
    ])
}

#[test]
pub fn test_create_new_table() {
    let table_id = TableId::new([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ]);
    let table = Table::new(table_id, TableMetadata::new(4, 8));
    assert_eq!(table.id, string_to_table_id("1".to_string()));
}

#[test]
pub fn test_convert_to_page() {
    let page_size = 16;
    let table = create_table();
    let page = table.to_page(4, 8, page_size);
    assert_eq!(page.height(), page_size);
    for row in page.iter() {
        println!("{:?}", row);
    }
    let comparison_page = create_page();
    for (i, row) in page.iter().enumerate() {
        assert_eq!(row.is_alloc, comparison_page[i].is_alloc);
        assert_eq!(row.idx, comparison_page[i].idx);
        assert_eq!(row.data, comparison_page[i].data);
    }

    // Save page as Json
    // let serialized = serde_json::to_string(&page).unwrap();
    // std::fs::write("tests/data/page.json", serialized).unwrap();

    // Save page as binary
    // let serialized = bincode::serialize(&page).unwrap();
    // std::fs::write("tests/data/page.afp", serialized).unwrap();
}

#[test]
#[should_panic]
pub fn test_convert_to_page_too_small() {
    let table = create_table();
    table.to_page(4, 8, 4);
}

#[test]
pub fn test_convert_from_page() {
    let page = create_page();
    let table = Table::from_page(
        TableId::new([1; 32]),
        page,
        std::mem::size_of::<u32>(),
        std::mem::size_of::<u64>(),
    );
    println!("{:?}", table.body);
    let comparison_table = create_table();
    assert_eq!(table.body, comparison_table.body);
}

#[test]
pub fn test_unordered_table() {
    let table_id = TableId::new([0; 32]);
    let index_bytes = 4;
    let data_bytes = 8;
    let mut table = Table::new(table_id, TableMetadata::new(index_bytes, data_bytes));
    table.body.insert(
        uint_to_be_vec(16, index_bytes),
        uint_to_be_vec(32, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(2, index_bytes),
        uint_to_be_vec(4, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(128, index_bytes),
        uint_to_be_vec(256, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(1000, index_bytes),
        uint_to_be_vec(65536, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(65791, index_bytes),
        uint_to_be_vec(65792, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(4, index_bytes),
        uint_to_be_vec(8, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(64, index_bytes),
        uint_to_be_vec(128, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(8, index_bytes),
        uint_to_be_vec(16, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(1, index_bytes),
        uint_to_be_vec(2, data_bytes),
    );
    table.body.insert(
        uint_to_be_vec(32, index_bytes),
        uint_to_be_vec(64, data_bytes),
    );
    let comparison_table = create_table();
    assert_eq!(table.body, comparison_table.body);
    let page = table.to_page(index_bytes, data_bytes, 16);
    let comparison_page = create_page();
    for (i, row) in page.iter().enumerate() {
        assert_eq!(row.is_alloc, comparison_page[i].is_alloc);
        assert_eq!(row.idx, comparison_page[i].idx);
        assert_eq!(row.data, comparison_page[i].data);
    }
}
