use clap::{Parser, Subcommand};
use stark_vm::vm::config::VmConfig;

use crate::commands::{keygen, keygen::KeygenCommand, prove, verify};

#[derive(Debug, Parser)]
#[command(author, version, about = "VM STARK CLI")]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    #[command(name = "keygen", about = "Generate partial proving and verifying keys")]
    /// Generate partial proving and verifying keys
    Keygen(keygen::KeygenCommand),

    #[command(name = "prove", about = "Generates a multi-STARK proof")]
    /// Generates a multi-STARK proof
    Prove(prove::ProveCommand),

    #[command(name = "verify", about = "Verifies a multi-STARK proof")]
    /// Verifies a multi-STARK proof
    Verify(verify::VerifyCommand),
}

impl Cli {
    pub fn run(config: VmConfig) -> Self {
        let cli = Self::parse();
        match &cli.command {
            CliCommand::Keygen(keygen) => {
                let cmd = KeygenCommand {
                    output_folder: keygen.output_folder.clone(),
                    asm_file_path: keygen.asm_file_path.clone(),
                };
                cmd.execute(config).unwrap();
            }
            CliCommand::Prove(prove) => {
                prove.execute(config).unwrap();
            }
            CliCommand::Verify(verify) => {
                verify.execute(config).unwrap();
            }
        }
        cli
    }
}
