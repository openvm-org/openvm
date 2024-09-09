use std::marker::PhantomData;

use afs_stark_backend::config::PcsProverData;
use ax_sdk::{engine::StarkEngine, page_config::PageConfig};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::InputFileOp;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;

use self::{filter::CacheFilterCommand, inner_join::CacheInnerJoinCommand};
use super::{parse_afo_file, CommonCommands};

pub mod filter;
pub mod inner_join;

#[derive(Debug, Parser)]
pub struct CacheCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[arg(
        long = "cache_folder",
        short = 'c',
        help = "Folder to store the cached trace data",
        required = false,
        default_value = "bin/olap/tmp/cache"
    )]
    pub cache_folder: String,

    #[clap(flatten)]
    pub common: CommonCommands,

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> CacheCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        cache_folder: String,
    ) -> Result<()> {
        let afo = parse_afo_file(common.afo_path.clone());
        for op in afo.operations {
            match op.operation {
                InputFileOp::Filter => {
                    CacheFilterCommand::execute(config, engine, common, op, cache_folder.clone())
                        .unwrap();
                }
                // InputFileOp::GroupBy => {
                //     CacheGroupByCommand::execute(config, engine, common, op).unwrap();
                // }
                InputFileOp::InnerJoin => {
                    CacheInnerJoinCommand::execute(
                        config,
                        engine,
                        common,
                        op,
                        cache_folder.clone(),
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
