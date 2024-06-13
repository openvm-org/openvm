use clap::Parser;
use color_eyre::eyre::Result;

use crate::common::config::Config;

/// `afs keygen` command
/// Uses information from config.toml to generate partial proving and verifying keys and
/// saves them to the specified `output-folder` as *.partial.pk and *.partial.vk.
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
    pub fn execute(self, _config: &Config) -> Result<()> {
        // WIP: Wait for ReadWrite chip in https://github.com/axiom-crypto/afs-prototype/pull/45
        Ok(())
    }
}
