use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    sync::Arc,
};

use afs_page::{
    execution_air::ExecutionAir,
    multitier_page_rw_checker::page_controller::{
        MyLessThanTupleParams, PageController, PageTreeParams,
    },
    page_btree::PageBTree,
};
use afs_primitives::range_gate::RangeCheckerGateChip;
use afs_stark_backend::{config::Com, prover::trace::ProverTraceData};
use afs_test_utils::page_config::MultitierPageConfig;
use color_eyre::eyre::Result;
use logical_interface::{
    afs_input::{types::InputFileOp, AfsInputFile},
    utils::string_to_u16_vec,
};
use p3_field::{AbstractField, PrimeField64};
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

pub mod keygen;
pub mod mock;
pub mod prove;
pub mod verify;

pub const BABYBEAR_COMMITMENT_LEN: usize = 8;
pub const DECOMP_BITS: usize = 16;
pub const LIMB_BITS: usize = 16;
pub const LEAF_HEIGHT: usize = 32;
pub const INTERNAL_HEIGHT: usize = 32;
pub const DATA_BUS: usize = 0;
pub const INTERNAL_DATA_BUS: usize = 1;
pub const LT_BUS: usize = 2;
pub const INIT_PATH_BUS: usize = 3;
pub const FINAL_PATH_BUS: usize = 4;
pub const OPS_BUS: usize = 5;

fn read_from_path(path: String) -> Option<Vec<u8>> {
    let file = File::open(path).unwrap();
    let mut reader = BufReader::new(file);
    let mut buf = vec![];
    reader.read_to_end(&mut buf).unwrap();
    Some(buf)
}

fn write_bytes(bytes: &[u8], path: String) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(bytes)?;
    Ok(())
}

fn create_prefix(config: &MultitierPageConfig) -> String {
    format!(
        "{:?}_{}_{}_{}_{}_{}_{}_cap_{}_{}_{}_{}",
        config.page.mode,
        config.page.index_bytes,
        config.page.data_bytes,
        config.page.leaf_height,
        config.page.internal_height,
        config.page.bits_per_fe,
        config.page.max_rw_ops,
        config.tree.init_leaf_cap,
        config.tree.init_internal_cap,
        config.tree.final_leaf_cap,
        config.tree.final_internal_cap,
    )
}

pub fn commit_to_string(commit: &[u32]) -> String {
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
    instructions: &AfsInputFile,
) {
    let idx_len = (instructions.header.index_bytes + 1) / 2;
    let data_len = (instructions.header.data_bytes + 1) / 2;
    for op in &instructions.operations {
        match op.operation {
            InputFileOp::Read => {}
            InputFileOp::Insert => {
                // if op.args.len() != 2 {
                //     return Err(eyre!("Invalid number of arguments for insert operation"));
                // }
                assert!(op.args.len() == 2);
                let index_input = op.args[0].clone();
                let index = string_to_u16_vec(index_input, idx_len);
                let data_input = op.args[1].clone();
                let data = string_to_u16_vec(data_input, data_len);
                db.update(&index, &data)
            }
            InputFileOp::Write => {
                // if op.args.len() != 2 {
                //     return Err(eyre!("Invalid number of arguments for write operation"));
                // }
                assert!(op.args.len() == 2);
                let index_input = op.args[0].clone();
                let index = string_to_u16_vec(index_input, idx_len);
                let data_input = op.args[1].clone();
                let data = string_to_u16_vec(data_input, data_len);
                db.update(&index, &data)
            }
            _ => panic!(),
        };
    }
}

pub fn get_page_controller<SC: StarkGenericConfig, const COMMITMENT_LEN: usize>(
    config: &MultitierPageConfig,
    idx_len: usize,
    data_len: usize,
) -> PageController<SC, COMMITMENT_LEN>
where
    Val<SC>: AbstractField + PrimeField64,
    Com<SC>: Into<[Val<SC>; COMMITMENT_LEN]>,
{
    let range_checker = Arc::new(RangeCheckerGateChip::new(2, 1 << DECOMP_BITS));

    PageController::new(
        DATA_BUS,
        INTERNAL_DATA_BUS,
        OPS_BUS,
        LT_BUS,
        idx_len,
        data_len,
        PageTreeParams {
            path_bus_index: INIT_PATH_BUS,
            leaf_cap: Some(config.tree.init_leaf_cap),
            internal_cap: Some(config.tree.init_internal_cap),
            leaf_page_height: config.page.leaf_height,
            internal_page_height: config.page.internal_height,
        },
        PageTreeParams {
            path_bus_index: FINAL_PATH_BUS,
            leaf_cap: Some(config.tree.final_leaf_cap),
            internal_cap: Some(config.tree.final_internal_cap),
            leaf_page_height: config.page.leaf_height,
            internal_page_height: config.page.internal_height,
        },
        MyLessThanTupleParams {
            limb_bits: config.page.bits_per_fe,
            decomp: DECOMP_BITS,
        },
        range_checker,
    )
}

pub fn get_ops_sender(idx_len: usize, data_len: usize) -> ExecutionAir {
    ExecutionAir::new(OPS_BUS, idx_len, data_len)
}
