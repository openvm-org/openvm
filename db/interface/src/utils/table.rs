use afs_page::page_rw_checker::page_controller::{OpType, Operation};
use afs_stark_backend::config::{Com, PcsProof, PcsProverData, StarkGenericConfig, Val};
use datafusion::{arrow::array::RecordBatch, error::Result, execution::context::SessionContext};
use p3_field::PrimeField64;
use p3_uni_stark::Domain;
use serde::{de::DeserializeOwned, Serialize};

use crate::common::committed_page::CommittedPage;

pub async fn get_record_batches(ctx: &SessionContext, name: &str) -> Result<Vec<RecordBatch>> {
    let df = ctx.table(name).await.unwrap();
    let record_batches = df.collect().await.unwrap();
    Ok(record_batches)
}

/// Converts a RecordBatch into a vector of Insert operations
pub fn convert_to_ops<SC: StarkGenericConfig>(rb: RecordBatch) -> Vec<Operation>
where
    Val<SC>: PrimeField64,
    PcsProverData<SC>: Serialize + DeserializeOwned + Send + Sync,
    PcsProof<SC>: Send + Sync,
    Domain<SC>: Send + Sync,
    Com<SC>: Send + Sync,
    SC::Pcs: Send + Sync,
    SC::Challenge: Send + Sync,
{
    let cp: CommittedPage<SC> = CommittedPage::from_record_batch(rb);
    let page = cp.page;
    let ops = page
        .rows
        .iter()
        .enumerate()
        .map(|(i, row)| Operation {
            clk: i + 1,
            idx: row.idx.to_vec(),
            data: row.data.to_vec(),
            op_type: OpType::Write,
        })
        .collect();
    ops
}
