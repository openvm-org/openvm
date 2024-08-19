use afs_page::common::page::Page;
use afs_stark_backend::{
    config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val},
    prover::trace::ProverTraceData,
};
use datafusion::arrow::{
    array::{Int64Array, RecordBatch, UInt32Array},
    datatypes::{DataType, Schema},
};
use derivative::Derivative;
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use self::utils::convert_to_record_batch;
use crate::NUM_IDX_COLS;

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

    pub fn from_record_batch(rb: RecordBatch, height: usize) -> Self {
        let schema = (*rb.schema()).clone();
        let num_rows = rb.num_rows();
        let columns = rb.columns();

        // Initialize a vector to hold each row, with an extra column for `is_alloc`
        let mut rows: Vec<Vec<u32>> = vec![vec![0; columns.len() + 1]; num_rows];
        let zero_rows: Vec<Vec<u32>> = vec![vec![0; columns.len() + 1]; height - num_rows];

        // Iterate over columns and fill the rows
        for (col_idx, column) in columns.iter().enumerate() {
            // TODO: handle other data types
            let array = match column.data_type() {
                DataType::UInt32 => column.as_any().downcast_ref::<UInt32Array>().unwrap(),
                DataType::Int64 => {
                    let array = column.as_any().downcast_ref::<Int64Array>().unwrap();
                    let array = array
                        .values()
                        .iter()
                        .map(|&v| v as u32)
                        .collect::<Vec<u32>>();
                    &UInt32Array::from(array)
                }
                _ => panic!("Unsupported data type: {}", column.data_type()),
            };
            for (row_idx, row) in rows.iter_mut().enumerate() {
                row[0] = 1;
                row[col_idx + 1] = array.value(row_idx);
            }
        }
        rows.extend(zero_rows);

        // TODO: we will temporarily take the first row as the index and all other rows as the data fields
        let page = Page::from_2d_vec(&rows, NUM_IDX_COLS, columns.len() - NUM_IDX_COLS);
        Self {
            // TODO: generate a page_id based on the hash of the Page
            page_id: "".to_string(),
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
        $crate::committed_page::CommittedPage::<$config>::new($name.to_string(), schema, page, None)
    }};
}
