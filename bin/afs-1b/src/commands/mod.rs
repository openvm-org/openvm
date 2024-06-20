use afs_chips::pagebtree::PageBTree;
use afs_stark_backend::prover::trace::ProverTraceData;
use afs_test_utils::page_config::MultitierPageConfig;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input_instructions::{types::InputFileBodyOperation, AfsInputInstructions},
    utils::{fixed_bytes_to_field_vec, string_to_be_vec},
};
use p3_uni_stark::StarkGenericConfig;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
};

pub mod keygen;
pub mod mock;
pub mod prove;
pub mod verify;

pub const BABYBEAR_COMMITMENT_LEN: usize = 8;
pub const DECOMP_BITS: usize = 16;
pub const LIMB_BITS: usize = 16;
pub const LEAF_HEIGHT: usize = 32;
pub const INTERNAL_HEIGHT: usize = 32;

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

pub fn commit_to_string(commit: &Vec<u32>) -> String {
    commit.iter().fold("".to_owned(), |acc, x| {
        acc.to_owned() + &format!("{:08x}", x)
    })
}

pub fn get_prover_data_from_file<SC: StarkGenericConfig>(path: String) -> ProverTraceData<SC>
where
    ProverTraceData<SC>: Serialize + DeserializeOwned,
{
    let data = read_from_path(path).unwrap();
    bincode::deserialize::<ProverTraceData<SC>>(&data).unwrap()
}

pub fn load_input_file<const COMMITMENT_LEN: usize>(
    db: &mut PageBTree<COMMITMENT_LEN>,
    instructions: &AfsInputInstructions,
) {
    for op in &instructions.operations {
        match op.operation {
            InputFileBodyOperation::Read => {}
            InputFileBodyOperation::Insert => {
                // if op.args.len() != 2 {
                //     return Err(eyre!("Invalid number of arguments for insert operation"));
                // }
                assert!(op.args.len() == 2);
                let index_input = op.args[0].clone();
                let index = string_to_be_vec(index_input, instructions.header.index_bytes);
                let index = fixed_bytes_to_field_vec(index);
                let data_input = op.args[1].clone();
                let data = string_to_be_vec(data_input, instructions.header.data_bytes);
                let data = fixed_bytes_to_field_vec(data);
                db.update(&index, &data)
            }
            InputFileBodyOperation::Write => {
                // if op.args.len() != 2 {
                //     return Err(eyre!("Invalid number of arguments for write operation"));
                // }
                assert!(op.args.len() == 2);
                let index_input = op.args[0].clone();
                let index = string_to_be_vec(index_input, instructions.header.index_bytes);
                let index = fixed_bytes_to_field_vec(index);
                let data_input = op.args[1].clone();
                let data = string_to_be_vec(data_input, instructions.header.data_bytes);
                let data = fixed_bytes_to_field_vec(data);
                db.update(&index, &data)
            }
        };
    }
}
