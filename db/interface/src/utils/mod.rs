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

pub fn generate_random_bytes(num_bytes: usize) -> String {
    let mut bytes = vec![0u8; num_bytes];
    OsRng.fill_bytes(&mut bytes);
    let h = hex::encode(bytes);
    format!("0x{}", h)
}
