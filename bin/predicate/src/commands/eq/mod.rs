use clap::Parser;
use color_eyre::eyre::Result;

#[derive(Debug, Parser)]
pub struct EqCommand {
    #[arg(
        long = "table-id",
        short = 't',
        help = "The table id to run the predicate on",
        required = true
    )]
    pub table_id: String,

    #[arg(
        long = "value",
        short = 'v',
        help = "The value to compare against",
        required = true
    )]
    pub value: String,

    #[arg(
        long = "output-file",
        short = 'o',
        help = "Save the output to file",
        required = false,
        default_value = "output/eq.csv"
    )]
    pub output_file: Option<String>,
}

/// `eq` command
impl EqCommand {
    /// Execute the `eq` command
    pub fn execute(self) -> Result<()> {
        println!("Running eq command on table: {}", self.table_id);

        // For each row in the Table, check the predicate on the row's data

        Ok(())
    }
}
