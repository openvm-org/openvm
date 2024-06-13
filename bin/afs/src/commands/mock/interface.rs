use alloy_primitives::U256;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input_instructions::AfsInputInstructions, afs_interface::AfsInterface, mock_db::MockDb,
    table::types::TableMetadata,
};

#[derive(Debug, Parser)]
pub struct InterfaceCommand {
    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

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
        long = "print",
        short = 'p',
        help = "Print the table",
        required = false
    )]
    pub print: bool,
}

/// `mock interface` subcommand
impl InterfaceCommand {
    /// Execute the `mock interface` command
    pub fn execute(self) -> Result<()> {
        println!("afi_file_path: {}", self.afi_file_path);
        let _instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;

        let mut db = if let Some(db_file_path) = self.db_file_path {
            println!("db_file_path: {}", db_file_path);
            MockDb::from_file(&db_file_path)
        } else {
            let default_table_metadata = TableMetadata::new(32, 1024);
            MockDb::new(default_table_metadata)
        };

        let mut interface = AfsInterface::<U256, U256>::new(&mut db);

        let table_id = self.table_id;
        let table = interface.get_table(table_id.clone());
        match table {
            Some(table) => {
                if self.print {
                    println!("Table ID: {}", table.id);
                    println!("Table Metadata {:?}", table.metadata);
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
