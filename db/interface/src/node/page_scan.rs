use std::{marker::PhantomData, sync::Arc};

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
    prover::types::Proof,
};
use afs_test_utils::engine::StarkEngine;
use async_trait::async_trait;
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::TableSource};
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info;

use super::{functionality::filter::FilterFn, AxdbNodeExecutable};
use crate::{
    committed_page::CommittedPage, expr::AxdbExpr, utils::table::get_record_batches, MAX_ROWS,
    NUM_IDX_COLS,
};

pub struct PageScan<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
    pub input: Arc<dyn TableSource>,
    pub output: Option<CommittedPage<SC>>,
    pub table_name: String,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub proof: Option<Proof<SC>>,
    pub filters: Vec<AxdbExpr>,
    pub filter_io: Vec<CommittedPage<SC>>,
    pub filter_proofs: Vec<Proof<SC>>,
    // TODO: support projection
    pub projection: Option<Vec<usize>>,
    _marker: PhantomData<E>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> PageScan<SC, E> {
    pub fn new(
        table_name: String,
        input: Arc<dyn TableSource>,
        filters: Vec<AxdbExpr>,
        projection: Option<Vec<usize>>,
    ) -> Self {
        Self {
            table_name,
            pk: None,
            input,
            output: None,
            proof: None,
            filters,
            filter_io: vec![],
            filter_proofs: vec![],
            projection,
            _marker: PhantomData::<E>,
        }
    }
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> PageScan<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    // TODO: this will eventualy need to support Schemas with different data types
    pub fn page_stats(&self) -> (usize, usize) {
        let schema = self.input.schema();
        let idx_len = NUM_IDX_COLS;
        let data_len = schema.fields().len() - NUM_IDX_COLS;
        (idx_len, data_len)
    }

    fn filter_name(&self) -> String {
        format!("{}.{}", self.name(), "Filter")
    }
}

#[async_trait]
impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> AxdbNodeExecutable<SC, E>
    for PageScan<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    async fn execute(&mut self, ctx: &SessionContext, _engine: &E) -> Result<()> {
        info!("execute PageScan");

        // Convert RecordBatches to a CommittedPage
        // NOTE: only one RecordBatch is supported for now
        let record_batches = get_record_batches(ctx, &self.table_name).await.unwrap();
        if record_batches.len() != 1 {
            panic!(
                "Unexpected number of record batches in PageScan: {}",
                record_batches.len()
            );
        }
        let rb = &record_batches[0];
        let mut committed_page = CommittedPage::from_record_batch(rb.clone(), MAX_ROWS);
        self.filter_io.push(committed_page.clone());

        if !self.filters.is_empty() {
            for filter in &self.filters {
                committed_page = FilterFn::<SC, E>::execute(filter, committed_page).await?;
                self.filter_io.push(committed_page.clone());
            }
        }

        self.output = Some(committed_page);
        Ok(())
    }

    async fn keygen(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("keygen PageScan");
        if !self.filters.is_empty() {
            // Since filtering does not change the Schema, we can use the same proving key for all filters on this PageScan node
            let filter = &self.filters[0];
            let (idx_len, data_len) = self.page_stats();
            FilterFn::<SC, E>::keygen(
                engine,
                filter,
                self.filter_name().as_str(),
                idx_len,
                data_len,
            )
            .await?;
        }
        Ok(())
    }

    async fn prove(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("prove PageScan");
        if !self.filters.is_empty() {
            for (i, filter) in self.filters.iter().enumerate() {
                let proof = FilterFn::<SC, E>::prove(
                    engine,
                    &self.filter_io[i],
                    &self.filter_io[i + 1],
                    filter,
                    self.filter_name().as_str(),
                    self.filter_io[i].page.idx_len(),
                    self.filter_io[i].page.data_len(),
                )
                .await?;
                self.filter_proofs.push(proof);
            }
        }
        Ok(())
    }

    async fn verify(&self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("verify PageScan");
        if !self.filters.is_empty() {
            for (i, filter) in self.filters.iter().enumerate() {
                FilterFn::<SC, E>::verify(
                    engine,
                    &self.filter_proofs[i],
                    filter,
                    self.filter_name().as_str(),
                    self.filter_io[i].page.idx_len(),
                    self.filter_io[i].page.data_len(),
                )
                .await?;
            }
        }
        Ok(())
    }

    fn output(&self) -> &Option<CommittedPage<SC>> {
        &self.output
    }

    fn proof(&self) -> &Option<Proof<SC>> {
        &self.proof
    }

    fn name(&self) -> &str {
        "PageScan"
    }
}
