use clap::Parser;
use color_eyre::eyre::Result;

#[cfg(test)]
pub mod tests;

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
        // WIP: wait for PR #45: https://github.com/axiom-crypto/afs-prototype/pull/45
        Ok(())
    }

    pub fn read_page_file(&self) -> Result<Vec<Vec<u32>>> {
        let page_file = std::fs::read(&self.page_file)?;
        let page_file: Vec<Vec<u32>> = serde_json::from_slice(&page_file)?;
        Ok(page_file)
    }

    pub fn write_output_file(&self, output: Vec<u8>) -> Result<()> {
        std::fs::write(&self.output_file, output)?;
        Ok(())
    }
}