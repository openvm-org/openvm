use alloy_primitives::U256;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input_instructions::AfsInputInstructions, afs_interface::AfsInterface, mock_db::MockDb,
    table::types::TableMetadata,
};

#[derive(Debug, Parser)]
pub struct WriteCommand {
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

    #[arg(
        long = "output-db-file",
        short = 'o',
        help = "Output DB file path (default: no output file saved)",
        required = false
    )]
    pub output_db_file_path: Option<String>,

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
}

/// `mock read` subcommand
impl WriteCommand {
    /// Execute the `mock read` command
    pub fn execute(self) -> Result<()> {
        let mut db = if let Some(db_file_path) = self.db_file_path {
            println!("db_file_path: {}", db_file_path);
            MockDb::from_file(&db_file_path)
        } else {
            let default_table_metadata = TableMetadata::new(32, 32);
            MockDb::new(default_table_metadata)
        };

        println!("afi_file_path: {}", self.afi_file_path);
        let instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;
        let table_id = instructions.header.table_id.clone();

        let mut interface = AfsInterface::<U256, U256>::new(&mut db);
        interface.load_input_file(&self.afi_file_path)?;
        let table = interface.get_table(table_id).unwrap();
        if !self.silent {
            println!("Table ID: {}", table.id);
            println!("{:?}", table.metadata);
            for (index, data) in table.body.iter() {
                println!("{:?}: {:?}", index, data);
            }
        }

        if let Some(output_db_file_path) = self.output_db_file_path {
            db.save_to_file(&output_db_file_path)?;
        }

        Ok(())
    }
}
