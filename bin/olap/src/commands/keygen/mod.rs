use std::marker::PhantomData;

use afs_stark_backend::config::PcsProverData;
use ax_sdk::{engine::StarkEngine, page_config::PageConfig};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::InputFileOp;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;

use self::{filter::KeygenFilterCommand, inner_join::KeygenInnerJoinCommand};
use super::{parse_afo_file, CommonCommands};

pub mod filter;
pub mod inner_join;

#[derive(Debug, Parser)]
pub struct KeygenCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[arg(
        long = "keys_folder",
        short = 'k',
        help = "Folder to store the proving and verifying keys",
        required = false,
        default_value = "bin/olap/tmp/keys"
    )]
    pub keys_folder: String,

    #[clap(flatten)]
    pub common: CommonCommands,

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> KeygenCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        keys_folder: String,
    ) -> Result<()> {
        let afo = parse_afo_file(common.afo_path.clone());
        for op in afo.operations {
            match op.operation {
                InputFileOp::Filter => {
                    KeygenFilterCommand::execute(config, engine, common, op, keys_folder.clone())
                        .unwrap();
                }
                InputFileOp::InnerJoin => {
                    KeygenInnerJoinCommand::execute(
                        config,
                        engine,
                        common,
                        op,
                        keys_folder.clone(),
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
