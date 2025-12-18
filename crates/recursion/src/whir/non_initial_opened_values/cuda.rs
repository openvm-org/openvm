use cuda_backend_v2::F;
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::memory_manager::MemTracker;
use p3_field::TwoAdicField;
use stark_backend_v2::SystemParams;

use crate::{
    cuda::to_device_or_nullptr,
    whir::{
        cuda_abi::non_initial_opened_values_tracegen, cuda_tracegen::WhirBlobGpu,
        non_initial_opened_values::NonInitialOpenedValuesCols, num_queries_per_round,
        query_offsets, total_num_queries,
    },
};

#[tracing::instrument(level = "trace", skip_all)]
pub(in crate::whir) fn generate_trace(
    blob: &WhirBlobGpu,
    params: &SystemParams,
) -> DeviceMatrix<F> {
    let mem = MemTracker::start("tracegen.whir_non_initial_opened_values");
    let num_valid_rows = blob.codeword_opened_values.len();
    let height = num_valid_rows.next_power_of_two();
    let width = NonInitialOpenedValuesCols::<F>::width();
    let trace_d = DeviceMatrix::with_capacity(height, width);

    let num_rounds = params.num_whir_rounds();
    let num_queries_per_round = num_queries_per_round(&params);
    let total_queries = total_num_queries(&num_queries_per_round);
    let query_offsets = query_offsets(&num_queries_per_round);
    let k_whir = params.k_whir();
    let rows_per_query = 1 << k_whir;

    // Compute round_row_offsets for rounds 1..num_rounds (non-initial rounds)
    let mut round_row_offsets = Vec::with_capacity(num_rounds);
    round_row_offsets.push(0usize);
    for whir_round in 1..num_rounds {
        let rows_this_round = num_queries_per_round[whir_round] * rows_per_query;
        round_row_offsets.push(round_row_offsets.last().unwrap() + rows_this_round);
    }
    let rows_per_proof = *round_row_offsets.last().unwrap();

    let round_row_offsets_d = to_device_or_nullptr(&round_row_offsets).unwrap();
    let query_offsets_d = to_device_or_nullptr(&query_offsets).unwrap();

    let omega_k = F::two_adic_generator(k_whir);
    unsafe {
        non_initial_opened_values_tracegen(
            trace_d.buffer(),
            num_valid_rows,
            height,
            &blob.codeword_opened_values,
            &blob.codeword_states,
            num_rounds,
            k_whir,
            omega_k,
            &blob.zis,
            &blob.zi_roots,
            &blob.yis,
            &blob.raw_queries,
            &round_row_offsets_d,
            rows_per_proof,
            &query_offsets_d,
            total_queries,
        )
        .unwrap();
    }

    mem.emit_metrics();
    trace_d
}
