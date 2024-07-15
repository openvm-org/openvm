use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::{
    operation::{FilterOp, GroupByOp, InnerJoinOp},
    operations_file::AfsOperationsFile,
    types::InputFileOp,
};

pub mod keygen;

#[derive(Debug, Parser)]
pub struct CommonCommands {
    #[arg(
        long = "db-path",
        short = 'd',
        help = "The path to the database",
        required = true
    )]
    pub db_path: String,

    #[arg(
        long = "afo-path",
        short = 'f',
        help = "The path to the .afo file",
        required = true
    )]
    pub afo_path: String,

    #[arg(
        long = "output-path",
        short = 'o',
        help = "The path to the output file",
        required = false
    )]
    pub output_path: Option<String>,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
}

pub fn parse_afo_file(afo_path: String) -> Result<()> {
    let afo = AfsOperationsFile::open(afo_path.clone());
    afo.operations.iter().for_each(|op| match op.operation {
        InputFileOp::Insert | InputFileOp::Write | InputFileOp::Read => {
            panic!("Operation not supported");
        }
        InputFileOp::Filter => {
            let filter = FilterOp::parse(op.args.clone()).unwrap();
            execute_filter_op::<SC>(cfg, self, &mut db, filter).unwrap();
        }
        InputFileOp::GroupBy => {
            let group_by = GroupByOp::parse(op.args.clone()).unwrap();
            execute_group_by::<SC>(cfg, self, &mut db, group_by).unwrap();
        }
        InputFileOp::InnerJoin => {
            let inner_join = InnerJoinOp::parse(op.args.clone()).unwrap();
            execute_inner_join::<SC>(cfg, self, &mut db, inner_join).unwrap();
        }
    });
}

pub fn inner_join_setup(
    config: &PageConfig,
    common: &InnerJoinCommonCommands,
    db_path: String,
    afo_path: String,
) -> (Table, Table, usize, usize, usize) {
    let mut db = MockDb::from_file(&db_path);
    let afo = AfsOperationsFile::open(afo_path.clone());
    let op = afo.operations.get(0).unwrap();
    let interface_left = AfsInterface::new_with_table(op.table_id_left.to_string(), &mut db);
    let table_left = interface_left.current_table().unwrap();
    let page_left = table_left.to_page(
        table_left.metadata.index_bytes,
        table_left.metadata.data_bytes,
        height,
    );
    let index_len_left = (table_left.metadata.index_bytes + 1) / 2;
    let data_len_left = (table_left.metadata.data_bytes + 1) / 2;

    let interface_right = AfsInterface::new_with_table(op.table_id_right.to_string(), db);
    let table_right = interface_right.current_table().unwrap();
    let page_right = table_right.to_page(
        table_right.metadata.index_bytes,
        table_right.metadata.data_bytes,
        height,
    );
    let index_len_right = (table_right.metadata.index_bytes + 1) / 2;
    let data_len_right = (table_right.metadata.data_bytes + 1) / 2;

    (table_left, table_right, height, bits_per_fe, degree)
}
