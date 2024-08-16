use std::{marker::PhantomData, sync::Arc};

use afs_page::{execution_air::ExecutionAir, page_rw_checker::page_controller::PageController};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
    prover::trace::TraceCommitmentBuilder,
};
use afs_test_utils::engine::StarkEngine;
use datafusion::{error::Result, execution::context::SessionContext, logical_expr::TableSource};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField64;
use p3_matrix::dense::DenseMatrix;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};

use super::AfsNodeExecutable;
use crate::{
    committed_page::CommittedPage,
    utils::table::{convert_to_ops, get_record_batches},
    BITS_PER_FE, MAX_ROWS, NUM_IDX_COLS, OPS_BUS_IDX, PAGE_BUS_IDX, PCS_LOG_DEGREE, RANGE_BUS_IDX,
    RANGE_CHECK_BITS,
};

pub struct PageScan<SC: StarkGenericConfig, E: StarkEngine<SC>> {
    pub input: Arc<dyn TableSource>,
    pub output: Option<CommittedPage<SC>>,
    pub page_id: String,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    _marker: PhantomData<E>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> PageScan<SC, E> {
    pub fn new(page_id: String, input: Arc<dyn TableSource>) -> Self {
        Self {
            page_id,
            pk: None,
            input,
            output: None,
            _marker: PhantomData::<E>,
        }
    }
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC>> AfsNodeExecutable<SC, E> for PageScan<SC, E>
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
        println!("execute PageScan");
        let record_batches = get_record_batches(ctx, &self.page_id).await.unwrap();
        if record_batches.len() != 1 {
            panic!(
                "Unexpected number of record batches in PageScan: {}",
                record_batches.len()
            );
        }
        let rb = &record_batches[0];
        let page = CommittedPage::from_record_batch(rb.clone(), MAX_ROWS);
        self.output = Some(page);

        Ok(())
    }

    async fn keygen(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        println!("keygen PageScan");
        let schema = self.input.schema();
        let idx_len = NUM_IDX_COLS;
        let data_len = schema.fields().len() - NUM_IDX_COLS;

        let page_controller: PageController<SC> = PageController::new(
            PAGE_BUS_IDX,
            RANGE_BUS_IDX,
            OPS_BUS_IDX,
            idx_len,
            data_len,
            BITS_PER_FE,
            RANGE_CHECK_BITS,
        );
        let ops_sender = ExecutionAir::new(OPS_BUS_IDX, idx_len, data_len);

        let mut keygen_builder = engine.keygen_builder();
        page_controller.set_up_keygen_builder(&mut keygen_builder, &ops_sender);
        let pk = keygen_builder.generate_pk();
        self.pk = Some(pk);

        Ok(())
    }

    async fn prove(&mut self, ctx: &SessionContext, engine: &E) -> Result<()> {
        println!("prove PageScan");
        let schema = self.input.schema();
        let idx_len = NUM_IDX_COLS;
        let data_len = schema.fields().len() - NUM_IDX_COLS;

        let record_batches = get_record_batches(ctx, &self.page_id).await.unwrap();
        if record_batches.len() != 1 {
            panic!(
                "Unexpected number of record batches in PageScan: {}",
                record_batches.len()
            );
        }
        let rb = &record_batches[0];
        let committed_page: CommittedPage<SC> =
            CommittedPage::from_record_batch(rb.clone(), MAX_ROWS);
        let zk_ops = convert_to_ops::<SC>(rb.clone());

        let mut page_controller: PageController<SC> = PageController::new(
            PAGE_BUS_IDX,
            RANGE_BUS_IDX,
            OPS_BUS_IDX,
            idx_len,
            data_len,
            BITS_PER_FE,
            RANGE_CHECK_BITS,
        );

        let ops_sender = ExecutionAir::new(OPS_BUS_IDX, idx_len, data_len);
        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
        let page_init = committed_page.page;

        let (init_page_pdata, final_page_pdata) = page_controller.load_page_and_ops(
            &page_init,
            None, //Some(Arc::new(init_prover_data)),
            None,
            &zk_ops,
            MAX_ROWS * 2,
            &mut trace_builder.committer,
        );

        let ops_sender_trace = ops_sender.generate_trace(&zk_ops, MAX_ROWS);
        let pk = self.pk.as_ref().unwrap();

        let proof = page_controller.prove(
            engine,
            pk,
            &mut trace_builder,
            init_page_pdata,
            final_page_pdata,
            &ops_sender,
            ops_sender_trace,
        );

        Ok(())
    }

    async fn verify(&self, ctx: &SessionContext, engine: &E) -> Result<()> {
        println!("verify PageScan");
        Ok(())
    }

    fn output(&self) -> &Option<CommittedPage<SC>> {
        &self.output
    }
}
