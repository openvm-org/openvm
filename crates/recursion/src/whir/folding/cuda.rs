use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::memory_manager::MemTracker;
use stark_backend_v2::{F, SystemParams};

use super::WhirFoldingCols;
use crate::whir::{
    cuda_abi::whir_folding_tracegen, cuda_tracegen::WhirBlobGpu, num_queries_per_round,
    total_num_queries,
};

#[tracing::instrument(skip_all)]
pub(in crate::whir) fn generate_trace(
    blob: &WhirBlobGpu,
    params: &SystemParams,
    num_proofs: usize,
) -> DeviceMatrix<F> {
    let mem = MemTracker::start("tracegen.whir_folding");
    let num_rounds = params.num_whir_rounds();
    let num_queries_per_round = num_queries_per_round(&params);
    let total_queries = total_num_queries(&num_queries_per_round);
    let k_whir = params.k_whir();
    let internal_nodes = (1 << k_whir) - 1;
    let num_rows_per_proof = total_queries * internal_nodes;
    let num_valid_rows = num_rows_per_proof * num_proofs;
    debug_assert_eq!(blob.folding_records.len(), num_valid_rows);

    let height = num_valid_rows.next_power_of_two();
    let width = WhirFoldingCols::<F>::width();
    let trace = DeviceMatrix::with_capacity(height, width);

    if num_valid_rows > 0 {
        unsafe {
            whir_folding_tracegen(
                trace.buffer(),
                u32::try_from(num_valid_rows).expect("num_valid_rows must fit in u32"),
                u32::try_from(height).expect("height must fit in u32"),
                &blob.folding_records,
                u32::try_from(num_rounds).expect("num_rounds must fit in u32"),
                u32::try_from(total_queries).expect("total_queries must fit in u32"),
                u32::try_from(k_whir).expect("k_whir must fit in u32"),
            )
            .unwrap();
        }
    }

    mem.emit_metrics();
    trace
}
