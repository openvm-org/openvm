use afs_test_utils::page_config::{MultitierPageConfig, PageConfig};
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

pub const BABYBEAR_COMMITMENT_LEN: usize = 8;
pub const DECOMP_BITS: usize = 8;
pub const LIMB_BITS: usize = 16;

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

fn create_prefix(config: &MultitierPageConfig) -> String {
    format!(
        "{:?}_{}_{}_{}_{}_{}_cap_{}_{}_{}_{}",
        config.page.mode,
        config.page.index_bytes,
        config.page.data_bytes,
        config.page.height,
        config.page.bits_per_fe,
        config.page.max_rw_ops,
        config.tree.init_leaf_cap,
        config.tree.init_internal_cap,
        config.tree.final_leaf_cap,
        config.tree.final_internal_cap,
    )
}
