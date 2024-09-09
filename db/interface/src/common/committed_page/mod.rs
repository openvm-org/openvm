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

use self::utils::{convert_columns_to_page_rows, convert_to_record_batch, get_num_idx_fields};
use super::cryptographic_object::CryptographicObjectTrait;
use crate::{utils::generate_random_alpha_string, BITS_PER_FE, NUM_IDX_COLS};

pub mod column;
pub mod execution_plan;
pub mod table_provider;
pub mod utils;

/// A CommittedPage is a hybrid structure of Axiom's Page type and DataFusion's Schema type. It is used in
/// every step of AxdbNode execution and contains the necessary information convert the Page data into a
/// DataFusion RecordBatch.
#[derive(Derivative, Serialize, Deserialize)]
#[derivative(Clone(bound = "ProverTraceData<SC>: Clone"))]
#[serde(bound(
    serialize = "ProverTraceData<SC>: Serialize",
    deserialize = "ProverTraceData<SC>: Deserialize<'de>"
))]
pub struct CommittedPage<SC: StarkGenericConfig> {
    /// The unique identifier for the page (currently used as the table name, but this will change in the future
    /// with the various architecture changes. To support SQL table names, this is a lowercase alpha string.)
    pub page_id: String,
    /// The schema of the Page
    pub schema: Schema,
    /// The number of Fields that are part of the Page's index. Counting from the leftmost Field. Note that
    /// depending on the size of the Field, multiple columns may be used to represent that Field.
    pub schema_num_idx_fields: usize,
    /// The data represented as a Page
    pub page: Page,
    /// The cached trace data from this node's execution
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
        let schema_num_idx_fields = get_num_idx_fields(&schema, page[0].idx.len(), BITS_PER_FE);
        Self {
            page_id,
            schema,
            schema_num_idx_fields,
            page,
            cached_trace: None,
        }
    }

    pub fn new_with_page_id(page_id: &str, schema: Schema, page: Page) -> Self {
        let schema_num_idx_fields = get_num_idx_fields(&schema, page[0].idx.len(), BITS_PER_FE);
        Self {
            page_id: page_id.to_string(),
            schema,
            schema_num_idx_fields,
            page,
            cached_trace: None,
        }
    }

    pub fn new_from_paths(page_path: &str, schema_path: &str) -> Self {
        let page = std::fs::read(page_path).unwrap();
        let page: Page = bincode::deserialize(&page).unwrap();
        let schema = std::fs::read(schema_path).unwrap();
        let schema: Schema = bincode::deserialize(&schema).unwrap();
        Self::new(schema, page)
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
        let schema_num_idx_fields = get_num_idx_fields(&schema, idx_len, BITS_PER_FE);

        let columns = cols.into_iter().map(|(_, values)| values).collect();
        let rows = convert_columns_to_page_rows(columns, alloc_rows);

        let page = Page::from_2d_vec(&rows, idx_len, data_len);
        Self {
            page_id,
            schema,
            schema_num_idx_fields,
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
        let schema_num_idx_fields = get_num_idx_fields(&schema, page[0].idx.len(), BITS_PER_FE);
        Self {
            page_id,
            schema,
            schema_num_idx_fields,
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

impl<SC: StarkGenericConfig> CryptographicObjectTrait for CommittedPage<SC> {
    fn schema(&self) -> Schema {
        self.schema.clone()
    }

    fn page(&self) -> Page {
        self.page.clone()
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
