use std::sync::Arc;

use afs_page::common::page::Page;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    prover::trace::ProverTraceData,
};
use datafusion::arrow::{
    array::{ArrayRef, RecordBatch, UInt32Array},
    datatypes::Schema,
};
use p3_field::PrimeField64;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use self::utils::convert_to_record_batch;
use crate::BITS_PER_FE;

pub mod column;
pub mod execution_plan;
pub mod table_provider;
pub mod utils;

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "ProverTraceData<SC>: Serialize",
    deserialize = "ProverTraceData<SC>: Deserialize<'de>"
))]
pub struct CommittedPage<SC: StarkGenericConfig> {
    pub page_id: String,
    pub schema: Schema,
    pub page: Page,
    pub cached_trace: Option<ProverTraceData<SC>>,
}

impl<SC: StarkGenericConfig> CommittedPage<SC>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub fn new(
        page_id: String,
        schema: Schema,
        page: Page,
        cached_trace: Option<ProverTraceData<SC>>,
    ) -> Self {
        Self {
            page_id,
            schema,
            page,
            cached_trace,
        }
    }

    pub fn from_file(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let committed_page: CommittedPage<SC> = bincode::deserialize(&bytes).unwrap();
        committed_page
    }

    pub fn write_cached_trace(&mut self, trace: ProverTraceData<SC>) {
        self.cached_trace = Some(trace);
    }

    pub fn to_record_batch(&self) -> RecordBatch {
        convert_to_record_batch(self.page.clone(), self.schema.clone())
    }
}

#[macro_export]
macro_rules! committed_page {
    ($name:expr, $page_path:expr, $schema_path:expr, $config:tt) => {{
        let page_path = std::fs::read($page_path).unwrap();
        let page: Page = bincode::deserialize(&page_path).unwrap();
        let schema_path = std::fs::read($schema_path).unwrap();
        let schema: Schema = bincode::deserialize(&schema_path).unwrap();
        CommittedPage::<$config>::new($name.to_string(), schema, page, None)
    }};
}
