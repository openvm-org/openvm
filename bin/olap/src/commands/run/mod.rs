pub mod group_by;
pub mod inner_join;
pub mod utils;

use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use color_eyre::eyre::Result;
use group_by::execute_group_by;
use inner_join::execute_inner_join;
use logical_interface::{
    afs_input::{
        operation::{GroupByOp, InnerJoinOp, WhereOp},
        operations_file::AfsOperationsFile,
        types::InputFileOp,
    },
    mock_db::MockDb,
};
use p3_uni_stark::StarkGenericConfig;

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
    pub fn execute<SC: StarkGenericConfig>(&self, config: &PageConfig) -> Result<()> {
        let mut db = MockDb::from_file(&self.db_path);
        let afo = AfsOperationsFile::open(self.afo_path.clone());
        afo.operations.iter().for_each(|op| match op.operation {
            InputFileOp::Insert | InputFileOp::Write | InputFileOp::Read => {
                panic!("Operation not supported");
            }
            InputFileOp::Where => {
                let where_op = WhereOp::parse(op.args.clone()).unwrap();
                println!("where_op: {:?}", where_op);
            }
            InputFileOp::GroupBy => {
                let group_by = GroupByOp::parse(op.args.clone()).unwrap();
                execute_group_by::<SC>(config, self, &mut db, group_by).unwrap();
            }
            InputFileOp::InnerJoin => {
                let inner_join = InnerJoinOp::parse(op.args.clone()).unwrap();
                execute_inner_join::<SC>(config, self, &mut db, inner_join).unwrap();
            }
        });

        // let mut db = if let Some(db_path) = &self.db_path {
        //     println!("db_path: {}", db_path);
        //     MockDb::from_file(db_path)
        // } else {
        //     let default_table_metadata =
        //         TableMetadata::new(config.page.index_bytes, config.page.data_bytes);
        //     MockDb::new(default_table_metadata)
        // };

        // let mut interface =
        //     AfsInterface::new(config.page.index_bytes, config.page.data_bytes, &mut db);

        Ok(())
    }
}
