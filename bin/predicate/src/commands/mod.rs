pub mod keygen;
pub mod prove;
pub mod verify;

use std::time::Instant;

use afs_chips::single_page_index_scan::page_index_scan_input::Comp;
use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use p3_util::log2_strict_usize;

pub const PAGE_BUS_INDEX: usize = 0;
pub const RANGE_BUS_INDEX: usize = 1;

#[derive(Debug, Parser)]
pub struct CommonCommands {
    #[arg(
        long = "predicate",
        short = 'p',
        help = "The comparison predicate to prove",
        required = true
    )]
    pub predicate: String,

    #[arg(
        long = "cache-folder",
        short = 'c',
        help = "Folder that contains cached traces",
        required = false,
        default_value = "cache"
    )]
    pub cache_folder: String,

    #[arg(
        long = "output-folder",
        short = 'o',
        help = "Folder to save output files to",
        required = false,
        default_value = "bin/common/data/predicate"
    )]
    pub output_folder: String,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
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

pub fn common_setup(
    config: &PageConfig,
    predicate: String,
) -> (
    Instant,
    Comp,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    let start = Instant::now();
    let comp = string_to_comp(predicate);
    let idx_len = config.page.index_bytes / 2;
    let data_len = config.page.data_bytes / 2;
    let page_width = 1 + idx_len + data_len;
    let page_height = config.page.height;
    let idx_limb_bits = config.page.bits_per_fe;
    let idx_decomp = log2_strict_usize(page_height);
    let range_max = 1 << idx_decomp;
    (
        start,
        comp,
        idx_len,
        data_len,
        page_width,
        page_height,
        idx_limb_bits,
        idx_decomp,
        range_max,
    )
}
