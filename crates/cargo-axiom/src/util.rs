use std::{fmt::Display, path::PathBuf, str::FromStr};

use clap::ValueEnum;

#[derive(ValueEnum, Clone)]
pub(crate) enum ProofMode {
    App,
    E2e,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) enum Input {
    FilePath(PathBuf),
    HexBytes(Vec<u8>),
}

impl FromStr for Input {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if is_valid_hex_string(s) {
            // Remove 0x prefix if present
            let s = if s.starts_with("0x") {
                s.strip_prefix("0x").unwrap()
            } else {
                s
            };
            if s.is_empty() {
                return Ok(Input::HexBytes(Vec::new()));
            }
            if !s.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err("Invalid hex string.".to_string());
            }
            let bytes = hex::decode(s).map_err(|e| e.to_string())?;
            Ok(Input::HexBytes(bytes))
        } else if PathBuf::from(s).exists() {
            Ok(Input::FilePath(PathBuf::from(s)))
        } else {
            Err("Input must be a valid file path or hex string.".to_string())
        }
    }
}

pub(crate) fn is_valid_hex_string(s: &str) -> bool {
    if s.len() % 2 != 0 {
        return false;
    }
    // All hex digits with optional 0x prefix
    s.starts_with("0x") && s[2..].chars().all(|c| c.is_ascii_hexdigit())
        || s.chars().all(|c| c.is_ascii_hexdigit())
}

pub(crate) fn write_status(style: &dyn Display, status: &str, msg: &str) {
    println!("{style}{status:>12}{style:#} {msg}");
}
