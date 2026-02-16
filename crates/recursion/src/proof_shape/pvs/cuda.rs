use openvm_cuda_backend::{base::DeviceMatrix, GpuBackend};
use openvm_cuda_common::memory_manager::MemTracker;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    cuda::{preflight::PreflightGpu, proof::ProofGpu},
    proof_shape::{cuda_abi::public_values_tracegen, pvs::PublicValuesCols},
    tracegen::ModuleChip,
};

pub struct PublicValuesGpuTraceGenerator;

impl ModuleChip<GpuBackend> for PublicValuesGpuTraceGenerator {
    type Ctx<'a> = (&'a [ProofGpu], &'a [PreflightGpu]);

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        height: Option<usize>,
    ) -> Option<AirProvingContext<GpuBackend>> {
        let (proofs_gpu, preflights_gpu) = ctx;
        let mem = MemTracker::start("tracegen.public_values");
        debug_assert_eq!(proofs_gpu.len(), preflights_gpu.len());

        let num_pvs = proofs_gpu[0].proof_shape.public_values.len();
        let num_valid_rows = proofs_gpu
            .iter()
            .map(|proof| proof.proof_shape.public_values.len())
            .sum::<usize>();

        let height = if let Some(height) = height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };
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
        Some(AirProvingContext::simple_no_pis(trace))
    }
}
