use alloy_primitives::U256;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input_instructions::AfsInputInstructions, afs_interface::AfsInterface, mock_db::MockDb,
    table::types::TableMetadata,
};

#[derive(Debug, Parser)]
pub struct AfiCommand {
    #[arg(
        long = "afi-file",
        short = 'f',
        help = "The .afi file input",
        required = true
    )]
    pub afi_file_path: String,

    #[arg(
        long = "db-file",
        short = 'd',
        help = "Mock DB file input (default: new empty DB)",
        required = false
    )]
    pub db_file_path: Option<String>,
}

/// `mock afi` subcommand
impl AfiCommand {
    /// Execute the `mock` command
    pub fn execute(self) -> Result<()> {
        println!("afi_file_path: {}", self.afi_file_path);
        let _instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;
        Ok(())

        // let mut db = if let Some(db_file_path) = self.db_file_path {
        //     println!("db_file_path: {}", db_file_path);
        //     MockDb::from_file(&db_file_path)
        // } else {
        //     let default_table_metadata = TableMetadata::new(32, 1024);
        //     MockDb::new(default_table_metadata)
        // };

        // let mut interface = AfsInterface::<U256, U256>::new(&mut db);
        // let table = interface.get_table(table_id)?;

        // Ok(())
    }
}
