use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::mock_db::MockDb;

#[derive(Debug, Parser)]
pub struct DescribeCommand {
    #[arg(long = "db-file", short = 'd', help = "Mock DB file input")]
    pub db_file_path: String,
}

/// `mock describe` subcommand
impl DescribeCommand {
    /// Execute the `mock describe` command
    pub fn execute(&self) -> Result<()> {
        let db = MockDb::from_file(&self.db_file_path);
        for (table_id, table) in db.tables.iter() {
            println!("Table ID: {}", table_id);
            println!("{:?}", table.db_table_metadata);
        }
        Ok(())
    }
}
