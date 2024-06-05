use clap::Parser;
use color_eyre::eyre::Result;

/// `afs keygen` command
#[derive(Debug, Parser)]
pub struct KeygenCommand {
    #[arg(
        long = "output-folder",
        short = 'o',
        help = "The folder to output the keys to",
        required = false,
        default_value = "output"
    )]
    pub output_folder: String,
}

impl KeygenCommand {
    /// Execute the `keygen` command
    pub fn execute(self) -> Result<()> {
        // println!("Caching page file: {}", self.page_file);
        // keygen::keygen_page(&self.page_file).await?;
        Ok(())
    }
}