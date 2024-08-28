use std::sync::Arc;

use afs_page::common::page::Page;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    prover::trace::ProverTraceData,
};
use datafusion::arrow::{
    array::{Array, RecordBatch},
    datatypes::{Field, Schema},
};
use derivative::Derivative;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use self::utils::{convert_columns_to_page_rows, convert_to_record_batch};
use crate::{utils::generate_random_alpha_string, NUM_IDX_COLS};

pub mod column;
pub mod execution_plan;
pub mod table_provider;
pub mod utils;

#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "ProverTraceData<SC>: Clone"))]
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
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    pub fn new(schema: Schema, page: Page) -> Self {
        let page_id = generate_random_alpha_string(32);
        Self {
            page_id,
            schema,
            page,
            cached_trace: None,
        }
    }

    pub fn new_with_page_id(page_id: &str, schema: Schema, page: Page) -> Self {
        Self {
            page_id: page_id.to_string(),
            schema,
            page,
            cached_trace: None,
        }
    }

    pub fn from_cols(cols: Vec<(Field, Arc<dyn Array>)>, idx_len: usize) -> Self {
        let page_id = generate_random_alpha_string(32);
        let alloc_rows = cols.first().unwrap().1.len();
        let data_len = cols.len() - idx_len;

        let schema = Schema::new(
            cols.iter()
                .map(|(field, _)| field.clone())
                .collect::<Vec<Field>>(),
        );

        let columns = cols.into_iter().map(|(_, values)| values).collect();
        let rows = convert_columns_to_page_rows(columns, alloc_rows);

        let page = Page::from_2d_vec(&rows, idx_len, data_len);
        Self {
            page_id,
            schema,
            page,
            cached_trace: None,
        }
    }

    pub fn from_file(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap();
        let committed_page: CommittedPage<SC> = bincode::deserialize(&bytes).unwrap();
        committed_page
    }

    pub fn from_record_batch(rb: RecordBatch) -> Self {
        let page_id = generate_random_alpha_string(32);

        let schema = (*rb.schema()).clone();
        let num_rows = rb.num_rows();
        let columns = rb.columns();

        let rows = convert_columns_to_page_rows(columns.to_vec(), num_rows);

        // TODO: we will temporarily take the first NUM_IDX_COLS rows as the index and all other rows as the data fields
        let page = Page::from_2d_vec(&rows, NUM_IDX_COLS, columns.len() - NUM_IDX_COLS);
        Self {
            page_id,
            schema,
            page,
            cached_trace: None,
        }
    }

    pub fn to_record_batch(&self) -> RecordBatch {
        convert_to_record_batch(self.page.clone(), self.schema.clone())
    }

    pub fn write_cached_trace(&mut self, trace: ProverTraceData<SC>) {
        self.cached_trace = Some(trace);
    }
}

impl<SC: StarkGenericConfig> std::fmt::Debug for CommittedPage<SC> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CommittedPage {{ page_id: {}, schema: {:?}, page: {:?} }}",
            self.page_id, self.schema, self.page
        )
    }
}

#[macro_export]
macro_rules! committed_page {
    ($name:expr, $page_path:expr, $schema_path:expr, $config:tt) => {{
        let page_path = std::fs::read($page_path).unwrap();
        let page: Page = bincode::deserialize(&page_path).unwrap();
        let schema_path = std::fs::read($schema_path).unwrap();
        let schema: Schema = bincode::deserialize(&schema_path).unwrap();
        $crate::common::committed_page::CommittedPage::<$config>::new_with_page_id(
            $name, schema, page,
        )
    }};
    ($page_path:expr, $schema_path:expr, $config:tt) => {{
        let page_path = std::fs::read($page_path).unwrap();
        let page: Page = bincode::deserialize(&page_path).unwrap();
        let schema_path = std::fs::read($schema_path).unwrap();
        let schema: Schema = bincode::deserialize(&schema_path).unwrap();
        $crate::common::committed_page::CommittedPage::<$config>::new(schema, page)
    }};
}
