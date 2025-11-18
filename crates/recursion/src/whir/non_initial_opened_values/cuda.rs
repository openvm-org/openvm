use cuda_backend_v2::F;
use openvm_cuda_backend::base::DeviceMatrix;
use p3_field::TwoAdicField;
use stark_backend_v2::SystemParams;

use crate::whir::{
    cuda_abi::non_initial_opened_values_tracegen, cuda_tracegen::WhirBlobGpu,
    non_initial_opened_values::NonInitialOpenedValuesCols,
};

pub(in crate::whir) fn generate_trace(blob: &WhirBlobGpu, params: SystemParams) -> DeviceMatrix<F> {
    let num_valid_rows = blob.non_initial_opened_values_records.len();
    let height = num_valid_rows.next_power_of_two();
    let width = NonInitialOpenedValuesCols::<F>::width();
    let trace_d = DeviceMatrix::with_capacity(height, width);

    let omega_k = F::two_adic_generator(params.k_whir);
    unsafe {
        non_initial_opened_values_tracegen(
            trace_d.buffer(),
            num_valid_rows,
            height,
            &blob.non_initial_opened_values_records,
            params.num_whir_rounds(),
            params.num_whir_queries,
            params.k_whir,
            omega_k,
            &blob.zis,
            &blob.zi_roots,
            &blob.yis,
            &blob.raw_queries,
        )
        .unwrap();
    }

    trace_d
}
