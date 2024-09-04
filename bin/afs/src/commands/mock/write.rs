use ax_sdk::page_config::PageConfig;
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::{afs_input::AfsInputFile, afs_interface::AfsInterface, mock_db::MockDb};

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
    pub fn execute(&self, _config: &PageConfig) -> Result<()> {
        let db_exists = self.db_file_path.is_some();
        let mut db = if db_exists {
            let db_file_path = self.db_file_path.as_ref().unwrap();
            println!("db_file_path: {}", db_file_path);
            MockDb::from_file(db_file_path)
        } else {
            MockDb::new()
        };

        println!("afi_file_path: {}", self.afi_file_path);
        let instructions = AfsInputFile::open(&self.afi_file_path)?;
        let table_id = instructions.header.table_id.clone();

        let mut interface = AfsInterface::new(
            instructions.header.index_bytes,
            instructions.header.data_bytes,
            &mut db,
        );
        interface.load_input_file(&self.afi_file_path)?;
        let table = interface.get_table(table_id).unwrap();
        if !self.silent {
            println!("Table ID: {}", table.id);
            println!("{:?}", table.metadata);
            for (index, data) in table.body.iter() {
                println!("{:?}: {:?}", index, data);
            }
        }

        if let Some(output_db_file_path) = &self.output_db_file_path {
            db.save_to_file(output_db_file_path)?;
        }

        Ok(())
    }
}
