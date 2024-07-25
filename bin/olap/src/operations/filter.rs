use std::time::Instant;

use afs_page::single_page_index_scan::page_index_scan_input::Comp;
use afs_test_utils::page_config::PageConfig;
use logical_interface::afs_input::{operation::FilterOp, types::AfsOperation};

use crate::RANGE_CHECK_BITS;

pub const PAGE_BUS_INDEX: usize = 0;
pub const RANGE_BUS_INDEX: usize = 1;

pub fn filter_setup(
    config: &PageConfig,
    op: AfsOperation,
) -> (
    Instant,
    FilterOp,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    let start = Instant::now();
    let filter_op = FilterOp::parse(op.args).unwrap();
    let idx_len = config.page.index_bytes / 2;
    let data_len = config.page.data_bytes / 2;
    let page_width = 1 + idx_len + data_len;
    let page_height = config.page.height;
    let idx_limb_bits = config.page.bits_per_fe;
    let idx_decomp = RANGE_CHECK_BITS;
    let range_max = 1 << idx_decomp;
    (
        start,
        filter_op,
        idx_len,
        data_len,
        page_width,
        page_height,
        idx_limb_bits,
        idx_decomp,
        range_max,
    )
}

pub fn string_to_comp(p: String) -> Comp {
    match p.to_lowercase().as_str() {
        "eq" | "=" => Comp::Eq,
        "lt" | "<" => Comp::Lt,
        "lte" | "<=" => Comp::Lte,
        "gt" | ">" => Comp::Gt,
        "gte" | ">=" => Comp::Gte,
        _ => panic!("Invalid comparison predicate: {}", p),
    }
}

pub fn comp_value_to_string(comp: Comp, value: Vec<u32>) -> String {
    let mut value_str = String::new();
    for v in value {
        value_str += &format!("{:x}", v);
    }
    format!("{}0x{}", comp, value_str)
}
