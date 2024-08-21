use afs_page::common::{page::Page, page_cols::PageCols};
use datafusion::arrow::datatypes::{DataType, Field, Schema};

/// Generate a schema and save it to disk
#[test]
#[ignore]
pub fn gen_schema() {
    let fields = vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
        Field::new("c", DataType::UInt32, false),
        Field::new("d", DataType::UInt32, false),
    ];
    let schema = Schema::new(fields);
    let serialized_schema = bincode::serialize(&schema).unwrap();
    std::fs::write("tests/data/example.schema.bin", serialized_schema).unwrap();

    let schema_file = std::fs::read("tests/data/example.schema.bin").unwrap();
    let schema_new = bincode::deserialize(&schema_file).unwrap();
    assert_eq!(schema, schema_new);
}

/// Generate a page and save it to disk
#[test]
#[ignore]
pub fn gen_page() {
    let page = Page::from_page_cols(vec![
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![2],
            data: vec![1, 0, 4],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![4],
            data: vec![2, 0, 8],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![8],
            data: vec![3, 0, 16],
        },
        PageCols::<u32> {
            is_alloc: 1,
            idx: vec![16],
            data: vec![4, 0, 32],
        },
    ]);
    let serialized = bincode::serialize(&page).unwrap();
    std::fs::write("tests/data/example.page.bin", serialized).unwrap();
}
