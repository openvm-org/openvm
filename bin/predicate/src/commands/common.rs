use afs_chips::single_page_index_scan::page_index_scan_input::Comp;
use clap::Parser;

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
        long = "value",
        short = 'v',
        help = "Value to prove the predicate against",
        required = true
    )]
    pub value: String,

    #[arg(
        long = "table-id",
        short = 't',
        help = "Table id to run the predicate on",
        required = true
    )]
    pub table_id: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "Path to the database file",
        required = true
    )]
    pub db_file_path: String,

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
