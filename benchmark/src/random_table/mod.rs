use afs_chips::common::page::Page;
use afs_test_utils::page_config::PageConfig;
use logical_interface::{
    afs_interface::utils::string_to_table_id,
    mock_db::MockDb,
    table::{types::TableMetadata, Table},
};
use rand::thread_rng;

pub fn generate_random_table(config: &PageConfig, table_id: String, db_file_path: String) -> Table {
    let index_bytes = config.page.index_bytes;
    let data_bytes = config.page.data_bytes;
    let height = config.page.height;
    let index_len = (index_bytes + 1) / 2;
    let data_len = (data_bytes + 1) / 2;

    let metadata = TableMetadata::new(index_bytes, data_bytes);
    let mut db = MockDb::new(metadata.clone());

    let table_id = string_to_table_id(table_id);

    let mut rng = thread_rng();
    let page = Page::random(
        &mut rng,
        index_len,
        data_len,
        u16::MAX as u32,
        u16::MAX as u32,
        height,
        height,
    );

    let table = Table::from_page(table_id, page.clone(), index_bytes, data_bytes);
    db.create_table(table_id, metadata);
    for row in page.rows {
        let index = row
            .idx
            .iter()
            .flat_map(|x| {
                x.to_be_bytes()
                    .to_vec()
                    .iter()
                    .skip(2)
                    .cloned()
                    .collect::<Vec<u8>>()
            })
            .collect::<Vec<u8>>();
        let data = row
            .data
            .iter()
            .flat_map(|x| {
                x.to_be_bytes()
                    .to_vec()
                    .iter()
                    .skip(2)
                    .cloned()
                    .collect::<Vec<u8>>()
            })
            .collect::<Vec<u8>>();
        db.write_data(table_id, index, data);
    }

    db.save_to_file(&db_file_path);
    table
}
