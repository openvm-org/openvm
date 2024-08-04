use std::marker::PhantomData;

use afs_stark_backend::config::{Com, PcsProof, PcsProverData};
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use clap::Parser;
use color_eyre::eyre::Result;
use logical_interface::afs_input::types::InputFileOp;
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use self::{filter::ProveFilterCommand, inner_join::ProveInnerJoinCommand};
use super::{parse_afo_file, CommonCommands};

pub mod filter;
pub mod inner_join;

#[derive(Debug, Parser)]
pub struct ProveCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
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
        help = "Folder containing the cached prover trace data",
        required = false,
        default_value = "bin/olap/tmp/cache"
    )]
    pub cache_folder: String,

    #[clap(flatten)]
    pub common: CommonCommands,

    #[clap(skip)]
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> ProveCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Sync,
    SC::Challenge: Send + Sync,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &CommonCommands,
        keys_folder: String,
        cache_folder: String,
    ) -> Result<()> {
        let afo = parse_afo_file(common.afo_path.clone());
        for op in afo.operations {
            match op.operation {
                InputFileOp::Filter => {
                    ProveFilterCommand::execute(
                        config,
                        engine,
                        common,
                        op,
                        keys_folder.clone(),
                        cache_folder.clone(),
                    )
                    .unwrap();
                }
                // InputFileOp::GroupBy => {
                //     ProveGroupByCommand::execute(config, engine, common, op).unwrap();
                // }
                InputFileOp::InnerJoin => {
                    ProveInnerJoinCommand::execute(
                        config,
                        engine,
                        common,
                        op,
                        keys_folder.clone(),
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
