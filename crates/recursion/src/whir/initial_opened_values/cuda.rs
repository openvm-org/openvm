use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::memory_manager::MemTracker;
use openvm_stark_backend::{prover::AirProvingContext, SystemParams};
use p3_field::TwoAdicField;

use crate::{
    tracegen::ModuleChip,
    whir::{
        cuda_abi::initial_opened_values_tracegen, cuda_tracegen::WhirBlobGpu,
        initial_opened_values::InitialOpenedValuesCols, num_queries_per_round, total_num_queries,
    },
};

pub(in crate::whir) struct InitialOpenedValuesGpuCtx<'a> {
    pub num_proofs: usize,
    pub blob: &'a WhirBlobGpu,
    pub params: &'a SystemParams,
}

pub(in crate::whir) struct InitialOpenedValuesGpuTraceGenerator;

impl ModuleChip<GpuBackend> for InitialOpenedValuesGpuTraceGenerator {
    type Ctx<'a> = InitialOpenedValuesGpuCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContext<GpuBackend>> {
        let num_proofs = ctx.num_proofs;
        let blob = ctx.blob;
        let params = ctx.params;

        let mem = MemTracker::start("tracegen.whir_initial_opened_values");
        let num_valid_rows = blob.codeword_value_accs.len();
        let omega_k = F::two_adic_generator(params.k_whir());
        let height = if let Some(h) = required_height {
            if h < num_valid_rows {
                return None;
            }
            h
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = InitialOpenedValuesCols::<F>::width();
        let trace_d = DeviceMatrix::with_capacity(height, width);

        let num_queries_per_round = num_queries_per_round(&params);
        let num_initial_queries = num_queries_per_round.first().copied().unwrap_or(0);
        let total_queries = total_num_queries(&num_queries_per_round);
        unsafe {
            initial_opened_values_tracegen(
                trace_d.buffer(),
                num_valid_rows,
                height,
                &blob.codeword_value_accs,
                &blob.poseidon_states,
                params.k_whir(),
                num_initial_queries,
                total_queries,
                omega_k,
                &blob.mus,
                &blob.zis,
                &blob.zi_roots,
                &blob.yis,
                &blob.raw_queries,
                &blob.iov_rows_per_proof_psums,
                &blob.commits_per_proof_psums,
                &blob.stacking_chunks_psums,
                &blob.stacking_widths_psums,
                &blob.mu_pows,
                num_proofs,
            )
            .unwrap();
        }
        mem.emit_metrics();
        Some(AirProvingContext::simple_no_pis(trace_d))
    }
}
