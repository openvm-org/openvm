use afs_test_utils::page_config::PageConfig;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
<<<<<<< HEAD
    afs_interface::AfsInterface, mock_db::MockDb, table::types::TableMetadata,
=======
    afs_interface::AfsInterface, mock_db::MockDb, table::types::TableId, utils::string_to_u8_vec,
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b
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
<<<<<<< HEAD
    pub fn execute(self, _config: &PageConfig) -> Result<()> {
        let mut db = if let Some(db_file_path) = self.db_file_path {
            println!("db_file_path: {}", db_file_path);
            MockDb::from_file(&db_file_path)
        } else {
            let default_table_metadata = TableMetadata::new(32, 1024);
            MockDb::new(default_table_metadata)
        };

        let mut interface = AfsInterface::new(32, 32, &mut db);

        let table_id = self.table_id;
=======
    pub fn execute(&self, _config: &PageConfig) -> Result<()> {
        let mut db = if let Some(db_file_path) = &self.db_file_path {
            println!("db_file_path: {}", db_file_path);
            MockDb::from_file(db_file_path)
        } else {
            MockDb::new()
        };

        let table_metadata = db
            .get_table_metadata(TableId::from_slice(
                string_to_u8_vec(self.table_id.clone(), 32).as_slice(),
            ))
            .unwrap();

        let mut interface = AfsInterface::new(
            table_metadata.index_bytes,
            table_metadata.data_bytes,
            &mut db,
        );

        let table_id = &self.table_id;
>>>>>>> d74b0541394676b6966e07196adf50328a41d65b
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
