pub mod filter;
pub mod group_by;
pub mod inner_join;

use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use color_eyre::eyre::Result;
use group_by::execute_group_by;
use inner_join::execute_inner_join;
use logical_interface::{
    afs_input::{
        operation::{FilterOp, GroupByOp, InnerJoinOp},
        operations_file::AfsOperationsFile,
        types::InputFileOp,
    },
    mock_db::MockDb,
};
use p3_uni_stark::StarkGenericConfig;

use self::filter::execute_filter_op;

#[derive(Debug, Parser)]
pub struct RunCommand {
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

impl RunCommand {
    pub fn execute<SC: StarkGenericConfig>(&self, cfg: &PageConfig) -> Result<()> {
        let mut db = MockDb::from_file(&self.db_path);
        let afo = AfsOperationsFile::open(self.afo_path.clone());
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
        Ok(())
    }
}
