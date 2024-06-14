use color_eyre::eyre::Result;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
};

pub mod cache;
pub mod keygen;
pub mod mock;
pub mod prove;
pub mod verify;

fn read_from_path(path: String) -> Option<Vec<u8>> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buf = vec![];
    reader.read_to_end(&mut buf).unwrap();
    Some(buf)
}

fn write_bytes(bytes: &Vec<u8>, path: String) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    Ok(())
}
