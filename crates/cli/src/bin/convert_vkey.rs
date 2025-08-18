use std::path::PathBuf;

use clap::Parser;
use openvm_circuit::{arch::ADDR_SPACE_OFFSET, system::memory::dimensions::MemoryDimensions};
use openvm_sdk::{
    fs::{read_object_from_file, write_object_to_file, write_to_file_json},
    keygen::{AppVerifyingKey, OldAppVerifyingKey},
};

#[derive(Parser, Debug)]
pub struct Cli {
    #[arg(long)]
    pub path: PathBuf,
}

fn main() -> eyre::Result<()> {
    let args = Cli::parse();
    let old_vk: OldAppVerifyingKey = read_object_from_file(&args.path)?;
    assert_eq!(old_vk.memory_dimensions.as_offset, ADDR_SPACE_OFFSET);
    let vk = AppVerifyingKey {
        fri_params: old_vk.fri_params,
        vk: old_vk.vk,
        memory_dimensions: MemoryDimensions {
            addr_space_height: old_vk.memory_dimensions.addr_space_height,
            address_height: old_vk.memory_dimensions.address_height,
        },
    };
    write_object_to_file(&args.path.with_extension("1.vk"), &vk)?;
    Ok(())
}
