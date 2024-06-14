use clap::Parser;

#[derive(Debug, Parser)]
pub struct CommonCommands {
    #[arg(
        long = "db-file",
        short = 'd',
        help = "Mock DB file input (default: new empty DB)",
        required = false
    )]
    pub db_file_path: Option<String>,

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

    #[arg(
        long = "silent",
        short = 's',
        help = "Don't print the output to stdout",
        required = false
    )]
    pub silent: bool,
}
