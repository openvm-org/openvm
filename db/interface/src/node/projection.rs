use std::sync::Arc;

use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    keygen::types::MultiStarkProvingKey,
    prover::types::Proof,
};
use afs_test_utils::engine::StarkEngine;
use async_trait::async_trait;
use datafusion::{arrow::datatypes::Schema, error::Result, execution::context::SessionContext};
use futures::lock::Mutex;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};

use super::{AxiomDbNode, AxiomDbNodeExecutable};
use crate::committed_page::CommittedPage;

pub struct Projection<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> {
    pub input: Arc<Mutex<AxiomDbNode<SC, E>>>,
    pub output: Option<CommittedPage<SC>>,
    pub schema: Schema,
    pub pk: Option<MultiStarkProvingKey<SC>>,
    pub proof: Option<Proof<SC>>,
}

#[async_trait]
impl<SC: StarkGenericConfig, E: StarkEngine<SC> + Send + Sync> AxiomDbNodeExecutable<SC, E>
    for Projection<SC, E>
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
        unimplemented!()
    }

    async fn keygen(&mut self, _ctx: &SessionContext, _engine: &E) -> Result<()> {
        unimplemented!()
    }

    async fn prove(&mut self, _ctx: &SessionContext, _engine: &E) -> Result<()> {
        unimplemented!()
    }

    async fn verify(&self, _ctx: &SessionContext, _engine: &E) -> Result<()> {
        unimplemented!()
    }

    fn output(&self) -> &Option<CommittedPage<SC>> {
        &self.output
    }

    fn proof(&self) -> &Option<Proof<SC>> {
        &self.proof
    }
}
