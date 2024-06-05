use clap::Subcommand;
use crate::commands::{keygen, cache, prove, verify};

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    #[command(
        name = "keygen",
        about = "Generate partial proving and verifying keys"
    )]
    /// Generate partial proving and verifying keys
    Keygen(keygen::KeygenCommand),

    #[command(
        name = "cache",
        about = "Create the cached trace of a page from a page file"
    )]
    /// Create cached trace of a page from a page file
    Cache(cache::CacheCommand),

    #[command(
        name = "prove",
        about = "Generates a multi-STARK proof"
    )]
    /// Generates a multi-STARK proof
    Prove(prove::ProveCommand),

    #[command(
        name = "verify",
        about = "Verifies a multi-STARK proof"
    )]
    /// Verifies a multi-STARK proof
    Verify(verify::VerifyCommand),
}
