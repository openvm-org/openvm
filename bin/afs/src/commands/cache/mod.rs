use clap::Parser;
use color_eyre::eyre::Result;

/// `afs cache` command
#[derive(Debug, Parser)]
pub struct CacheCommand {
    #[arg(
        long = "page-file",
        short = 'f',
        help = "The path to the page file",
        required = true
    )]
    pub page_file: String,

    #[arg(
        long = "output-file",
        short = 'o',
        help = "The path to the output file",
        required = false,
        default_value = "output/cache.bin"
    )]
    pub output_file: String,
}

impl CacheCommand {
    /// Execute the `cache` command
    pub fn execute(self) -> Result<()> {
        println!("Caching page file: {}", self.page_file);
        // cache::cache_page(&self.page_file).await?;
        Ok(())
    }
}