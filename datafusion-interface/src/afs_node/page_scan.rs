use std::sync::Arc;

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
};
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::TableSource};
use p3_field::PrimeField64;
use serde::{de::DeserializeOwned, Serialize};

use super::{AfsNode, AfsNodeExecutable};
use crate::{afs_expr::AfsExpr, committed_page::CommittedPage};

pub struct PageScan<SC: StarkGenericConfig> {
    pub page_id: String,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub input: Arc<dyn TableSource>,
    pub output: Option<Arc<CommittedPage<SC>>>,
}

impl<SC: StarkGenericConfig> AfsNodeExecutable<SC> for PageScan<SC>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    async fn execute(&mut self, ctx: &SessionContext) -> Result<()> {
        let df = ctx.table(&self.page_id).await.unwrap();
        // let s = self.input as CommittedPage<SC>;
        // println!("{:#?}", t);
        // t.show().await.unwrap();
        let record_batches = df.collect().await.unwrap();
        if record_batches.len() != 1 {
            panic!(
                "Unexpected number of record batches in PageScan: {}",
                record_batches.len()
            );
        }
        let rb = &record_batches[0];
        let page = CommittedPage::from_record_batch(rb.clone());
        self.output = Some(Arc::new(page));

        Ok(())
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
