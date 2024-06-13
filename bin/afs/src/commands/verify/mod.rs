use clap::Parser;
use color_eyre::eyre::Result;

/// `afs verify` command
/// Uses information from config.toml to verify a proof using the verifying key in `output-folder`
/// as */prove.bin.
#[derive(Debug, Parser)]
pub struct VerifyCommand {
    #[arg(
        long = "proof-file",
        short = 'f',
        help = "The path to the proof file",
        required = true
    )]
    pub proof_file: String,

    #[arg(
        long = "output-folder",
        short = 'o',
        help = "The folder to output the keys to",
        required = false,
        default_value = "output"
    )]
    pub output_folder: String,
}

impl VerifyCommand {
    /// Execute the `verify` command
    pub fn execute(self) -> Result<()> {
        println!("Verifying proof file: {}", self.proof_file);
        // verify::verify_ops(&self.proof_file).await?;
        Ok(())
    }
}
