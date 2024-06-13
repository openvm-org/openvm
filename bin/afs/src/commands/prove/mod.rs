use clap::Parser;
use color_eyre::eyre::Result;

/// `afs prove` command
#[derive(Debug, Parser)]
pub struct ProveCommand {
    #[arg(
        long = "ops-file",
        short = 'f',
        help = "The path to the ops file",
        required = true
    )]
    pub ops_file: String,

    #[arg(
        long = "output-file",
        short = 'o',
        help = "The path to the output file",
        required = false,
        default_value = "output/prove.bin"
    )]
    pub output_file: String,
}

impl ProveCommand {
    /// Execute the `prove` command
    pub fn execute(self) -> Result<()> {
        println!("Proving ops file: {}", self.ops_file);
        // prove::prove_ops(&self.ops_file).await?;
        Ok(())
    }
}
