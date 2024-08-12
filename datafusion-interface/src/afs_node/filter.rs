use std::sync::Arc;

use afs_stark_backend::{config::StarkGenericConfig, keygen::types::MultiStarkProvingKey};
use datafusion::{error::Result, logical_expr::TableSource};

use super::{AfsNode, AfsNodeExecutable};
use crate::afs_expr::AfsExpr;

pub struct Filter<SC: StarkGenericConfig> {
    pub predicate: AfsExpr,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub input: Arc<AfsNode<SC>>,
    pub output: Option<Arc<dyn TableSource>>,
}

impl<SC: StarkGenericConfig> AfsNodeExecutable<SC> for Filter<SC> {
    fn execute(&mut self) -> Result<()> {
        Ok(())
    }

    fn keygen(&mut self) -> Result<()> {
        Ok(())
    }

    fn prove(&mut self) -> Result<()> {
        Ok(())
    }

    fn verify(&self) -> Result<()> {
        Ok(())
    }
}
