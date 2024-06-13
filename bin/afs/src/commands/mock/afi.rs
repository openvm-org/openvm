use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input_instructions::AfsInputInstructions;

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
    /// Execute the `mock afi` command
    pub fn execute(self) -> Result<()> {
        println!("afi_file_path: {}", self.afi_file_path);
        let _instructions = AfsInputInstructions::from_file(&self.afi_file_path)?;
        Ok(())
    }
}
