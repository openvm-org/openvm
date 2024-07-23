use afs_chips::{
    common::page::Page,
    inner_join::controller::{IJBuses, T2Format, TableFormat},
};
use afs_test_utils::page_config::PageConfig;
use logical_interface::{
    afs_input::{operation::InnerJoinOp, types::AfsOperation},
    afs_interface::AfsInterface,
    mock_db::MockDb,
};

use crate::{commands::CommonCommands, RANGE_CHECK_BITS};

const RANGE_BUS: usize = 0;
const T1_INTERSECTOR_BUS: usize = 1;
const T2_INTERSECTOR_BUS: usize = 2;
const INTERSECTOR_T2_BUS: usize = 3;
const T1_OUTPUT_BUS: usize = 4;
const T2_OUTPUT_BUS: usize = 5;

fn get_page_info(db: &mut MockDb, table_id: String, height: usize) -> (Page, usize, usize) {
    let interface = AfsInterface::new_with_table(table_id, db);
    let table = interface.current_table().unwrap();
    let page = table.to_page(
        table.metadata.index_bytes,
        table.metadata.data_bytes,
        height,
    );
    let index_len = (table.metadata.index_bytes + 1) / 2;
    let data_len = (table.metadata.data_bytes + 1) / 2;

    (page, index_len, data_len)
}

pub fn inner_join_setup(
    config: &PageConfig,
    common: &CommonCommands,
    op: AfsOperation,
) -> (
    TableFormat,
    T2Format,
    IJBuses,
    InnerJoinOp,
    Page,
    Page,
    usize,
    usize,
) {
    let mut db = MockDb::from_file(&common.db_path);
    let height = config.page.height;
    let bits_per_fe = config.page.bits_per_fe;
    let range_chip_idx_decomp = RANGE_CHECK_BITS;

    let inner_join_op = InnerJoinOp::parse(op.args).unwrap();

    // Get input pages from database
    let (page_left, index_len_left, data_len_left) =
        get_page_info(&mut db, inner_join_op.table_id_left.to_string(), height);
    let (page_right, index_len_right, data_len_right) =
        get_page_info(&mut db, inner_join_op.table_id_right.to_string(), height);

    if !common.silent {
        println!("Left page:");
        page_left.pretty_print(bits_per_fe);
        println!("Right page:");
        page_right.pretty_print(bits_per_fe);
    }

    let inner_join_buses = IJBuses {
        range_bus_index: RANGE_BUS,
        t1_intersector_bus_index: T1_INTERSECTOR_BUS,
        t2_intersector_bus_index: T2_INTERSECTOR_BUS,
        intersector_t2_bus_index: INTERSECTOR_T2_BUS,
        t1_output_bus_index: T1_OUTPUT_BUS,
        t2_output_bus_index: T2_OUTPUT_BUS,
    };
    let t1_format = TableFormat::new(index_len_left, data_len_left, bits_per_fe);
    let t2_table_format = TableFormat::new(index_len_right, data_len_right, bits_per_fe);
    let t2_format = T2Format::new(
        t2_table_format,
        inner_join_op.fkey_start,
        inner_join_op.fkey_end,
    );

    (
        t1_format,
        t2_format,
        inner_join_buses,
        inner_join_op,
        page_left,
        page_right,
        height,
        range_chip_idx_decomp,
    )
}
