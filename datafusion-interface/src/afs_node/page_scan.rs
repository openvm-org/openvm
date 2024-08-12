use std::sync::Arc;

use afs_stark_backend::{config::StarkGenericConfig, keygen::types::MultiStarkProvingKey};
use datafusion::{error::Result, logical_expr::TableSource};

use super::{AfsNode, AfsNodeExecutable};
use crate::afs_expr::AfsExpr;

pub struct PageScan<SC: StarkGenericConfig> {
    pub page_id: String,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub input: Arc<dyn TableSource>,
    pub output: Option<Arc<dyn TableSource>>,
}

impl<SC: StarkGenericConfig> AfsNodeExecutable<SC> for PageScan<SC> {
    fn execute(&mut self) -> Result<()> {
        let s = &self.input.schema();
        let f = &s.fields();
        println!("{:#?}", f);
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
