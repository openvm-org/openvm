use clap::Parser;
use logical_interface::afs_input::operations_file::AfsOperationsFile;

pub mod cache;
pub mod keygen;
pub mod prove;
pub mod verify;

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
        help = "The path to the .afo file containing the OLAP commands",
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

pub fn parse_afo_file(afo_path: String) -> AfsOperationsFile {
    AfsOperationsFile::open(afo_path.clone())
    // afo.operations.iter().for_each(|op| match op.operation {
    //     InputFileOp::Insert | InputFileOp::Write | InputFileOp::Read => {
    //         panic!("Operation not supported");
    //     }
    //     InputFileOp::Filter => {
    //         let filter = FilterOp::parse(op.args.clone()).unwrap();
    //         execute_filter_op::<SC>(cfg, self, &mut db, filter).unwrap();
    //     }
    //     InputFileOp::GroupBy => {
    //         let group_by = GroupByOp::parse(op.args.clone()).unwrap();
    //         execute_group_by::<SC>(cfg, self, &mut db, group_by).unwrap();
    //     }
    //     InputFileOp::InnerJoin => {
    //         let inner_join = InnerJoinOp::parse(op.args.clone()).unwrap();
    //         execute_inner_join::<SC>(cfg, self, &mut db, inner_join).unwrap();
    //     }
    // });
}
