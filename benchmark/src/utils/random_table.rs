use std::{collections::HashSet, fs::File, io::Write};

use afs_chips::common::page::Page;
use afs_test_utils::page_config::PageConfig;
use color_eyre::eyre::Result;
use core::cmp::min;
use logical_interface::{
    afs_input::header::AfsHeader,
    afs_interface::utils::string_to_table_id,
    mock_db::MockDb,
    table::{types::TableMetadata, Table},
    utils::u16_vec_to_u8_vec,
};
use rand::{prelude::IteratorRandom, thread_rng, Rng};

pub fn generate_random_table(
    config: &PageConfig,
    table_id: String,
    db_file_path: String,
) -> Result<Table> {
    let index_bytes = config.page.index_bytes;
    let data_bytes = config.page.data_bytes;
    let height = config.page.height;
    let index_len = (index_bytes + 1) / 2;
    let data_len = (data_bytes + 1) / 2;

    let metadata = TableMetadata::new(index_bytes, data_bytes);
    let mut db = MockDb::new();

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
        let index = u16_vec_to_u8_vec(row.idx);
        let data = u16_vec_to_u8_vec(row.data);
        db.write_data(table_id, index, data);
    }

    db.save_to_file(&db_file_path)?;
    Ok(table)
}

pub fn generate_random_afi_rw(
    config: &PageConfig,
    table_id: String,
    afi_path: String,
    percent_reads: usize,
    percent_writes: usize,
) -> Result<()> {
    let index_bytes = config.page.index_bytes;
    let data_bytes = config.page.data_bytes;
    let height = config.page.height;
    let max_rw_ops = config.page.max_rw_ops;
    let max_writes = max_rw_ops * percent_writes / 100;
    let max_reads = max_rw_ops * percent_reads / 100;

    let mut file = File::create(afi_path.as_str())?;

    // Write AFI header
    let header = AfsHeader::new(table_id, index_bytes, data_bytes);
    writeln!(file, "TABLE_ID {}", header.table_id)?;
    writeln!(file, "INDEX_BYTES {}", header.index_bytes)?;
    writeln!(file, "DATA_BYTES {}", header.data_bytes)?;

    // Keep track of inserted indexes
    let mut inserted_indexes: HashSet<String> = HashSet::new();

    let max_inserts = min(height, max_writes);

    // Generate `INSERT` instructions
    for _ in 0..max_inserts {
        let mut idx = generate_random_hex_string(index_bytes);
        while inserted_indexes.contains(&idx) {
            idx = generate_random_hex_string(index_bytes);
        }
        let data = generate_random_hex_string(data_bytes);

        inserted_indexes.insert(idx.clone());
        writeln!(file, "INSERT {} {}", idx, data)?;
    }

    // Generate `WRITE` instructions
    if max_inserts < max_writes {
        for _ in max_inserts..max_writes {
            if let Some(random_index) = inserted_indexes.iter().choose(&mut thread_rng()) {
                let data = generate_random_hex_string(data_bytes);
                writeln!(file, "WRITE {} {}", random_index, data)?;
            }
        }
    }

    // Generate `READ` instructions
    for _ in 0..max_reads {
        if let Some(random_index) = inserted_indexes.iter().choose(&mut thread_rng()) {
            writeln!(file, "READ {}", random_index)?;
        }
    }

    Ok(())
}

fn generate_random_hex_string(num_bytes: usize) -> String {
    let mut rng = thread_rng();
    let bytes = (0..num_bytes)
        .map(|_| rng.gen_range(0..=u8::MAX))
        .collect::<Vec<u8>>();
    let hex_bytes = hex::encode(bytes);
    format!("0x{}", hex_bytes)
}
