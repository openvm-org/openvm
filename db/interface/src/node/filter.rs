use std::sync::Arc;

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
    prover::types::Proof,
};
use async_trait::async_trait;
use ax_sdk::engine::StarkEngine;
use datafusion::{error::Result, execution::context::SessionContext};
use futures::lock::Mutex;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};
use tracing::instrument;

use super::{functionality::filter::FilterFn, AxdbNode, AxdbNodeExecutable};
use crate::{
    common::{
        cryptographic_object::{CryptographicObject, CryptographicObjectTrait},
        expr::AxdbExpr,
    },
    NUM_IDX_COLS,
};

pub struct Filter<SC, E>
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC> + Send + Sync,
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub input: Arc<Mutex<AxdbNode<SC, E>>>,
    pub output: Option<CryptographicObject<SC>>,
    pub predicate: AxdbExpr,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub proof: Option<Proof<SC>>,
}

impl<SC, E> Filter<SC, E>
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC> + Send + Sync,
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    fn page_stats(&self, cp: &CryptographicObject<SC>) -> (usize, usize, usize) {
        let schema = cp.schema();
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
impl<SC, E> AxdbNodeExecutable<SC, E> for Filter<SC, E>
where
    SC: StarkGenericConfig,
    E: StarkEngine<SC> + Send + Sync,
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    #[instrument(level = "info", skip_all)]
    async fn keygen(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        let input = self.input.lock().await;
        let input = input.output().as_ref().unwrap();

        let (idx_len, data_len, _page_width) = self.page_stats(input);
        let pk = FilterFn::<SC, E>::keygen(engine, &self.predicate, self.name(), idx_len, data_len)
            .await?;
        self.pk = Some(pk);

        println!("input: {:?}", input);

        match input {
            CryptographicObject::CryptographicSchema(schema) => {
                let output = CryptographicObject::CryptographicSchema(schema.clone());
                self.output = Some(output);
            }
            _ => panic!("input is not a CryptographicSchema"),
        }
        Ok(())
    }

    #[instrument(level = "info", skip_all)]
    async fn execute(&mut self, _ctx: &SessionContext, _engine: &E) -> Result<()> {
        // let input = self.unlock_input().await;
        let input = self.input.lock().await;
        let input = input.output().as_ref().unwrap();
        match input {
            CryptographicObject::CommittedPage(cp) => {
                let output = FilterFn::<SC, E>::execute(&self.predicate, cp).await?;
                self.output = Some(output.into());
            }
            _ => panic!("input is not a CommittedPage<SC>"),
        }
        Ok(())
    }

    #[instrument(level = "info", skip_all)]
    async fn prove(&mut self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        let input = self.input.lock().await;
        let input = input.output().as_ref().unwrap();
        let (idx_len, data_len, _page_width) = self.page_stats(input);
        match input {
            CryptographicObject::CommittedPage(cp) => {
                let output = self.output.as_ref().unwrap();
                match output {
                    CryptographicObject::CommittedPage(output_page) => {
                        let proof = FilterFn::<SC, E>::prove(
                            engine,
                            cp,
                            output_page,
                            &self.predicate,
                            self.name(),
                            idx_len,
                            data_len,
                        )
                        .await?;
                        self.proof = Some(proof);
                    }
                    _ => panic!("output is not a CommittedPage<SC>"),
                }
            }
            _ => panic!("input is not a CommittedPage<SC>"),
        }
        Ok(())
    }

    #[instrument(level = "info", skip_all)]
    async fn verify(&self, _ctx: &SessionContext, engine: &E) -> Result<()> {
        let input = self.input.lock().await;
        let input = input.output().as_ref().unwrap();
        let (idx_len, data_len, _page_width) = self.page_stats(input);
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

    fn output(&self) -> &Option<CryptographicObject<SC>> {
        &self.output
    }

    fn proof(&self) -> &Option<Proof<SC>> {
        &self.proof
    }

    fn name(&self) -> &str {
        "Filter"
    }
}
