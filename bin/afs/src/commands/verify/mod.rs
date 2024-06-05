use clap::Parser;
use color_eyre::eyre::Result;

/// `afs verify` command
#[derive(Debug, Parser)]
pub struct VerifyCommand {
    #[arg(
        long = "proof-file",
        short = 'f',
        help = "The path to the proof file",
        required = true
    )]
    pub proof_file: String,
}

impl VerifyCommand {
    /// Execute the `verify` command
    pub fn execute(self) -> Result<()> {
        println!("Verifying proof file: {}", self.proof_file);
        // verify::verify_ops(&self.proof_file).await?;
        Ok(())
    }
}