use std::sync::Arc;

use afs_stark_backend::{
    config::{PcsProverData, StarkGenericConfig},
    keygen::types::MultiStarkProvingKey,
};
use afs_test_utils::engine::StarkEngine;
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::TableSource};

use super::{AfsNode, AfsNodeExecutable};
use crate::{afs_expr::AfsExpr, committed_page::CommittedPage};

pub struct Filter<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    pub predicate: AfsExpr,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub input: Arc<AfsNode<SC, E>>,
    pub output: Option<Arc<CommittedPage<SC>>>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> AfsNodeExecutable<SC, E> for Filter<SC, E> {
    async fn execute(&mut self, ctx: &SessionContext) -> Result<()> {
        unimplemented!()
    }

    async fn keygen(&mut self, ctx: &SessionContext, engine: &E) -> Result<()> {
        Ok(())
    }

    async fn prove(&mut self, ctx: &SessionContext) -> Result<()> {
        Ok(())
    }

    async fn verify(&self, ctx: &SessionContext) -> Result<()> {
        Ok(())
    }

    fn output(&self) -> Option<Arc<CommittedPage<SC>>> {
        self.output.clone()
    }
}
