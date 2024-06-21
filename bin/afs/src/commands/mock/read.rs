use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_interface::AfsInterface, mock_db::MockDb, table::types::TableMetadata,
};

#[derive(Debug, Parser)]
pub struct ReadCommand {
    #[arg(long = "table-id", short = 't', help = "The table ID", required = true)]
    pub table_id: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "Mock DB file input (default: new empty DB)",
        required = false
    )]
    pub db_file_path: Option<String>,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
}

/// `mock read` subcommand
impl ReadCommand {
    /// Execute the `mock read` command
    pub fn execute(&self, config: &PageConfig) -> Result<()> {
        let mut db = if let Some(db_file_path) = &self.db_file_path {
            println!("db_file_path: {}", db_file_path);
            MockDb::from_file(db_file_path)
        } else {
            let default_table_metadata =
                TableMetadata::new(config.page.index_bytes, config.page.data_bytes);
            MockDb::new(default_table_metadata)
        };

        let mut interface =
            AfsInterface::new(config.page.index_bytes, config.page.data_bytes, &mut db);

        let table_id = &self.table_id;
        let table = interface.get_table(table_id.clone());
        match table {
            Some(table) => {
                if !self.silent {
                    println!("Table ID: {}", table.id);
                    println!("{:?}", table.metadata);
                    for (index, data) in table.body.iter() {
                        println!("{:?}: {:?}", index, data);
                    }
                }
            }
            None => {
                panic!("No table at table_id: {}", table_id);
            }
        }

        Ok(())
    }
}
