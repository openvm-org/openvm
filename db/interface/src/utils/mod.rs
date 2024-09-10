use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use datafusion::arrow::error::Result;
use rand::{rngs::OsRng, RngCore};

pub mod pk;
pub mod table;

pub fn write_bytes(bytes: &[u8], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    Ok(())
}

pub fn read_bytes(path: &Path) -> Option<Vec<u8>> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buf = vec![];
    reader.read_to_end(&mut buf).unwrap();
    Some(buf)
}

pub fn generate_random_alpha_string(num_chars: usize) -> String {
    let chars = "abcdefghijklmnopqrstuvwxyz";
    let mut rng = OsRng;
    let mut bytes = vec![0u8; num_chars];
    rng.fill_bytes(&mut bytes);
    bytes
        .iter()
        .map(|b| chars.chars().nth((*b as usize) % chars.len()).unwrap())
        .collect()
}
