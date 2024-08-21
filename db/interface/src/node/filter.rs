use std::sync::Arc;

use afs_page::single_page_index_scan::{
    page_controller::PageController, page_index_scan_input::Comp,
};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
    prover::{trace::TraceCommitmentBuilder, types::Proof},
};
use afs_test_utils::engine::StarkEngine;
use async_trait::async_trait;
use datafusion::{error::Result, execution::context::SessionContext};
use futures::lock::Mutex;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};
use tracing::info;

use super::{AxdbNode, AxdbNodeExecutable};
use crate::{
    committed_page::CommittedPage, expr::AxdbExpr, BITS_PER_FE, PAGE_BUS_IDX, RANGE_BUS_IDX,
    RANGE_CHECK_BITS,
};

pub struct Filter<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
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

    fn page_stats(&self, page: &CommittedPage<SC>) -> (usize, usize, usize) {
        let idx_len = page.page.idx_len();
        let data_len = page.page.data_len();
        let page_width = 1 + idx_len + data_len;
        (idx_len, data_len, page_width)
    }

    fn decompose_predicate(&self) -> (Comp, u32) {
        // NOTE: we currently only support a predicate and a right-side value (left side value is the index column)
        match &self.predicate {
            AxdbExpr::BinaryExpr(expr) => {
                let op = &expr.op;
                let right = match *expr.right {
                    AxdbExpr::Literal(x) => x,
                    _ => panic!("Unsupported right side expression type"),
                };
                (op.clone(), right)
            }
            _ => panic!("Unsupported expression type"),
        }
    }

    fn page_controller(&self, idx_len: usize, data_len: usize, comp: Comp) -> PageController<SC> {
        PageController::new(
            PAGE_BUS_IDX,
            RANGE_BUS_IDX,
            idx_len,
            data_len,
            1 << RANGE_CHECK_BITS as u32,
            BITS_PER_FE,
            RANGE_CHECK_BITS,
            comp,
        )
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

        let (comp, right_value) = self.decompose_predicate();
        let input = self.input_clone().await;
        let (idx_len, data_len, page_width) = self.page_stats(&input);

        let page_input = input.page;
        let page_controller = self.page_controller(idx_len, data_len, comp.clone());
        let filter_output =
            page_controller.gen_output(page_input.clone(), vec![right_value], page_width, comp);

        let page_output = CommittedPage {
            // TODO: use a generated page_id
            page_id: "".to_string(),
            schema: input.schema,
            page: filter_output,
            cached_trace: None,
        };
        self.output = Some(page_output);
        Ok(())
    }

    async fn keygen(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("keygen Filter");

        let (comp, _right_value) = self.decompose_predicate();
        let input = self.input_clone().await;
        let (idx_len, data_len, page_width) = self.page_stats(&input);

        let page_controller = self.page_controller(idx_len, data_len, comp.clone());

        let mut keygen_builder = engine.keygen_builder();
        page_controller.set_up_keygen_builder(&mut keygen_builder, page_width, idx_len);
        let pk = keygen_builder.generate_pk();
        self.pk = Some(pk);

        Ok(())
    }

    async fn prove(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("prove Filter");

        let (comp, right_value) = self.decompose_predicate();
        let input = self.input_clone().await;
        let (idx_len, data_len, _page_width) = self.page_stats(&input);

        let mut page_controller = self.page_controller(idx_len, data_len, comp.clone());

        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let page_input = input.page;
        let page_output = self.output.as_ref().unwrap();
        let page_output = page_output.page.clone();

        let (input_prover_data, output_prover_data) = page_controller.load_page(
            page_input.clone(),
            page_output.clone(),
            None, // Some(Arc::new(input_trace_file)),
            None,
            vec![right_value],
            idx_len,
            data_len,
            BITS_PER_FE,
            RANGE_CHECK_BITS,
            &mut trace_builder.committer,
        );

        let pk = self.pk.as_ref().unwrap();

        let proof = page_controller.prove(
            engine,
            pk,
            &mut trace_builder,
            input_prover_data,
            output_prover_data,
            vec![right_value],
            RANGE_CHECK_BITS,
        );

        self.proof = Some(proof);

        Ok(())
    }

    async fn verify(&self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        info!("verify Filter");

        let (comp, right_value) = self.decompose_predicate();
        let input = self.input_clone().await;
        let (idx_len, data_len, _page_width) = self.page_stats(&input);

        let page_controller = self.page_controller(idx_len, data_len, comp.clone());
        let pk = self.pk.as_ref().unwrap();
        let vk = pk.vk();
        let proof = self.proof.as_ref().unwrap();
        page_controller
            .verify(engine, vk, proof, vec![right_value])
            .unwrap();

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
