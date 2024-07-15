use std::marker::PhantomData;

use afs_stark_backend::config::PcsProverData;
use afs_test_utils::{engine::StarkEngine, page_config::PageConfig};
use clap::Parser;
use color_eyre::eyre::Result;
use p3_field::PrimeField64;
use p3_uni_stark::{StarkGenericConfig, Val};
use serde::Serialize;

use super::InnerJoinCommonCommands;

#[derive(Debug, Parser, Clone)]
pub struct InnerJoinProveCommand<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    #[clap(skip)]
    pub _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> InnerJoinProveCommand<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize,
{
    pub fn execute(
        config: &PageConfig,
        engine: &E,
        common: &InnerJoinCommonCommands,
    ) -> Result<()> {
        Ok(())
    }
}
