use std::marker::PhantomData;

use afs_stark_backend::config::PcsProverData;
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::InputFileOp;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::de::DeserializeOwned;

use self::{filter::VerifyFilterCommand, inner_join::VerifyInnerJoinCommand};
use super::{parse_afo_file, CommonCommands};

pub mod filter;
pub mod inner_join;

#[derive(Debug, Parser)]
pub struct VerifyCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[arg(
        long = "keys_folder",
        short = 'k',
        help = "Folder containing the proving and verifying keys",
        required = false,
        default_value = "bin/olap/tmp/keys"
    )]
    pub keys_folder: String,

    #[arg(
        long = "cache_folder",
        short = 'c',
        help = "Folder containing the cache files",
        required = false
    )]
    pub cache_folder: Option<String>,

    #[arg(
        long = "proof_path",
        short = 'p',
        help = "Path to the proof file",
        required = false
    )]
    pub proof_path: Option<String>,

    #[clap(flatten)]
    pub common: CommonCommands,

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> VerifyCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: DeserializeOwned,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        keys_folder: String,
        cache_folder: Option<String>,
        proof_path: Option<String>,
    ) -> Result<()> {
        let afo = parse_afo_file(common.afo_path.clone());
        for op in afo.operations {
            match op.operation {
                InputFileOp::Filter => {
                    VerifyFilterCommand::execute(
                        config,
                        engine,
                        common,
                        op,
                        keys_folder.clone(),
                        cache_folder.clone(),
                        proof_path.clone(),
                    )
                    .unwrap();
                }
                // InputFileOp::GroupBy => {
                //     VerifyGroupByCommand::execute(config, engine, common, op).unwrap();
                // }
                InputFileOp::InnerJoin => {
                    VerifyInnerJoinCommand::execute(
                        config,
                        engine,
                        common,
                        op,
                        keys_folder.clone(),
                        proof_path.clone(),
                    )
                    .unwrap();
                }
                _ => {
                    panic!("Unsupported operation: {:?}", op);
                }
            }
        }
        Ok(())
    }
}
