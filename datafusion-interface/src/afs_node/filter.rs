use std::sync::Arc;

use afs_stark_backend::{config::StarkGenericConfig, keygen::types::MultiStarkProvingKey};
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::TableSource};

use super::{AfsNode, AfsNodeExecutable};
use crate::{afs_expr::AfsExpr, committed_page::CommittedPage};

pub struct Filter<SC: StarkGenericConfig> {
    pub predicate: AfsExpr,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub input: Arc<AfsNode<SC>>,
    pub output: Option<Arc<CommittedPage<SC>>>,
}

impl<SC: StarkGenericConfig> AfsNodeExecutable<SC> for Filter<SC> {
    async fn execute(&mut self, ctx: &SessionContext) -> Result<()> {
        unimplemented!()
    }

    async fn keygen(&mut self, ctx: &SessionContext) -> Result<()> {
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
