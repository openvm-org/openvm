use cuda_backend_v2::F;
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::memory_manager::MemTracker;

use crate::{
    cuda::{preflight::PreflightGpu, proof::ProofGpu},
    proof_shape::{cuda_abi::public_values_tracegen, pvs::PublicValuesCols},
};

#[tracing::instrument(level = "trace", skip_all)]
pub(in crate::proof_shape) fn generate_trace(
    proofs_gpu: &[ProofGpu],
    preflights_gpu: &[PreflightGpu],
) -> DeviceMatrix<F> {
    let mem = MemTracker::start("tracegen.public_values");
    debug_assert_eq!(proofs_gpu.len(), preflights_gpu.len());

    let num_pvs = proofs_gpu[0].proof_shape.public_values.len();
    let num_valid = proofs_gpu
        .iter()
        .map(|proof| {
            debug_assert_eq!(num_pvs, proof.proof_shape.public_values.len());
            proof.proof_shape.public_values.len()
        })
        .sum::<usize>();

    let height = num_valid.next_power_of_two();
    let width = PublicValuesCols::<u8>::width();
    let trace = DeviceMatrix::with_capacity(height, width);

    let pvs_data = proofs_gpu
        .iter()
        .map(|proof| proof.proof_shape.public_values.as_ptr())
        .collect::<Vec<_>>();
    let pvs_tidx = preflights_gpu
        .iter()
        .map(|preflight| preflight.proof_shape.pvs_tidx.as_ptr())
        .collect::<Vec<_>>();

    unsafe {
        public_values_tracegen(
            trace.buffer(),
            height,
            pvs_data,
            pvs_tidx,
            proofs_gpu.len(),
            num_pvs,
        )
        .unwrap();
    }
    mem.emit_metrics();
    trace
}
