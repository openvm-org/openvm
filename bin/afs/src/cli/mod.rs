pub mod types;

// use crate::commands::{cache, keygen, prove, verify};
// use crate::cli::types::CliCommand;

// use std::ffi::OsString;
// use std::fmt;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(author, version, about = "AFS CLI")]
pub struct Cli {
    #[command(subcommand)]
    pub command: types::CliCommand,
}

// impl Cli {
//     pub fn try_parse_args_from<I, T>(itr: I) -> Result<Self, clap::error::Error>
//     where
//         I: IntoIterator<Item = T>,
//         T: Into<OsString> + Clone,
//     {
//         Self::try_parse_from(itr)
//     }
// }
