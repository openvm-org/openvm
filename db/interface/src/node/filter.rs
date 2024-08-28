use std::sync::Arc;

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
    prover::types::Proof,
};
use afs_test_utils::engine::StarkEngine;
use async_trait::async_trait;
use datafusion::{error::Result, execution::context::SessionContext};
use futures::lock::Mutex;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info;

use super::{functionality::filter::FilterFn, AxdbNode, AxdbNodeExecutable};
use crate::{
    common::{committed_page::CommittedPage, expr::AxdbExpr},
    NUM_IDX_COLS,
};

pub struct Filter<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub input: Arc<Mutex<AxdbNode<SC, E>>>,
    pub output: Option<CommittedPage<SC>>,
    pub predicate: AxdbExpr,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub proof: Option<Proof<SC>>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> Filter<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    async fn input_clone(&self) -> CommittedPage<SC> {
        let input = self.input.lock().await;
        let input = input.output().as_ref().unwrap().clone();
        input
    }

    fn page_stats(&self, cp: &CommittedPage<SC>) -> (usize, usize, usize) {
        let schema = &cp.schema;
        // TODO: handle different data types
        // for field in schema.fields() {
        //     let data_type = field.data_type();
        //     let byte_len = data_type.primitive_width().unwrap();
        // }
        let idx_len = NUM_IDX_COLS;
        let data_len = schema.fields().len() - NUM_IDX_COLS;
        let page_width = 1 + idx_len + data_len;
        (idx_len, data_len, page_width)
    }
}

#[async_trait]
impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> AxdbNodeExecutable<SC, E>
    for Filter<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    async fn execute(&mut self, _ctx: &SessionContext, _engine: &E) -> Result<()> {
        info!("execute Filter");
        let input = self.input_clone().await;
        let output = FilterFn::<SC, E>::execute(&self.predicate, input).await?;
        self.output = Some(output);
        Ok(())
    }

    async fn keygen(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("keygen Filter");
        let input = self.input_clone().await;
        let (idx_len, data_len, _page_width) = self.page_stats(&input);
        let pk = FilterFn::<SC, E>::keygen(engine, &self.predicate, self.name(), idx_len, data_len)
            .await?;
        self.pk = Some(pk);

        Ok(())
    }

    async fn prove(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("prove Filter");
        let input = self.input_clone().await;
        let output = self.output.as_ref().unwrap();
        let (idx_len, data_len, _page_width) = self.page_stats(&input);
        let proof = FilterFn::<SC, E>::prove(
            engine,
            &input,
            output,
            &self.predicate,
            self.name(),
            idx_len,
            data_len,
        )
        .await?;
        self.proof = Some(proof);
        Ok(())
    }

    async fn verify(&self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("verify Filter");
        let input = self.input_clone().await;
        let (idx_len, data_len, _page_width) = self.page_stats(&input);
        let proof = self.proof.as_ref().unwrap();
        FilterFn::<SC, E>::verify(
            engine,
            proof,
            &self.predicate,
            self.name(),
            idx_len,
            data_len,
        )
        .await?;
        Ok(())
    }

    fn output(&self) -> &Option<CommittedPage<SC>> {
        &self.output
    }

    fn proof(&self) -> &Option<Proof<SC>> {
        &self.proof
    }

    fn name(&self) -> &str {
        "Filter"
    }
}
