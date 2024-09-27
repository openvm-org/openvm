use std::marker::PhantomData;

use afs_page::single_page_index_scan::{
    page_controller::PageController, page_index_scan_input::Comp,
};
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData},
    keygen::types::MultiStarkProvingKey,
    prover::{trace::TraceCommitmentBuilder, types::Proof},
};
use ax_sdk::engine::StarkEngine;
use datafusion::error::Result;
use p3_field::PrimeField64;
use p3_uni_stark::{Domain, StarkGenericConfig, Val};
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    common::{committed_page::CommittedPage, expr::AxdbExpr},
    utils::pk::PkUtil,
    BITS_PER_FE, PAGE_BUS_IDX, RANGE_BUS_IDX, RANGE_CHECK_BITS,
};

pub struct FilterFn<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
    _marker: PhantomData<(SC, E)>,
}

impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> FilterFn<SC, E>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub fn page_controller(idx_len: usize, data_len: usize, comp: Comp) -> PageController<SC> {
        PageController::new(
            PAGE_BUS_IDX,
            RANGE_BUS_IDX,
            idx_len,
            data_len,
            BITS_PER_FE,
            RANGE_CHECK_BITS,
            comp,
        )
    }

    pub async fn keygen(
        engine: &E,
        filter: &AxdbExpr,
        node_name: &str,
        idx_len: usize,
        data_len: usize,
    ) -> Result<MultiStarkProvingKey<SC>> {
        let page_width = 1 + idx_len + data_len;
        let comp = filter.decompose_binary_expr().1;
        let page_controller = Self::page_controller(idx_len, data_len, comp.clone());
        let mut keygen_builder = engine.keygen_builder();
        page_controller.set_up_keygen_builder(&mut keygen_builder, page_width);

        let pk = keygen_builder.generate_pk();
        PkUtil::<SC, E>::save_proving_key(node_name, idx_len, data_len, &pk)?;
        Ok(pk)
    }

    pub async fn execute(filter: &AxdbExpr, page: &CommittedPage<SC>) -> Result<CommittedPage<SC>> {
        let (_, comp, right_value) = filter.decompose_binary_expr();
        let right_value = match right_value {
            AxdbExpr::Literal(lit) => lit,
            _ => panic!("Only literal values are currently supported for filter"),
        };
        let idx_len = page.page.idx_len();
        let data_len = page.page.data_len();
        let page_width = page.page.width();

        let page_controller = Self::page_controller(idx_len, data_len, comp.clone());
        let filter_output =
            page_controller.gen_output(page.page.clone(), vec![right_value], page_width, comp);
        Ok(CommittedPage::new(page.schema.clone(), filter_output))
    }

    pub async fn prove(
        engine: &E,
        input: &CommittedPage<SC>,
        output: &CommittedPage<SC>,
        filter: &AxdbExpr,
        node_name: &str,
        idx_len: usize,
        data_len: usize,
    ) -> Result<Proof<SC>> {
        let (_, comp, right_value) = filter.decompose_binary_expr();
        let right_value = match right_value {
            AxdbExpr::Literal(lit) => lit,
            _ => panic!("Only literal values are currently supported for filter"),
        };
        let mut page_controller = Self::page_controller(idx_len, data_len, comp.clone());
        let prover = engine.prover();
        let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());

        let (input_prover_data, output_prover_data) = page_controller.load_page(
            input.page.clone(),
            output.page.clone(),
            None, // Some(Arc::new(input_trace_file)),
            None,
            vec![right_value],
            idx_len,
            data_len,
            BITS_PER_FE,
            RANGE_CHECK_BITS,
            &mut trace_builder.committer,
        );

        let pk = PkUtil::<SC, E>::find_proving_key(node_name, idx_len, data_len);

        let proof = page_controller.prove(
            engine,
            &pk,
            &mut trace_builder,
            input_prover_data,
            output_prover_data,
            vec![right_value],
            RANGE_CHECK_BITS,
        );

        Ok(proof)
    }

    pub async fn verify(
        engine: &E,
        proof: &Proof<SC>,
        filter: &AxdbExpr,
        node_name: &str,
        idx_len: usize,
        data_len: usize,
    ) -> Result<()> {
        let (_, comp, _) = filter.decompose_binary_expr();
        let pk = PkUtil::<SC, E>::find_proving_key(node_name, idx_len, data_len);
        let page_controller = Self::page_controller(idx_len, data_len, comp);
        page_controller.verify(engine, pk.vk(), proof).unwrap();
        Ok(())
    }
}
