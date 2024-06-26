use color_eyre::eyre::Result;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

pub mod keygen;
pub mod prove;
pub mod verify;

fn read_from_path(path: &Path) -> Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = vec![];
    reader.read_to_end(&mut buf)?;
    Ok(buf)
}

fn write_bytes(bytes: &[u8], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    Ok(())
}
