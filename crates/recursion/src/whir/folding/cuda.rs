use openvm_cuda_backend::{base::DeviceMatrix, GpuBackend};
use openvm_cuda_common::memory_manager::MemTracker;
use openvm_stark_backend::{prover::AirProvingContext, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;

use super::WhirFoldingCols;
use crate::{
    tracegen::ModuleChip,
    whir::{
        cuda_abi::whir_folding_tracegen, cuda_tracegen::WhirBlobGpu, num_queries_per_round,
        total_num_queries,
    },
};

pub(in crate::whir) struct FoldingGpuCtx<'a> {
    pub blob: &'a WhirBlobGpu,
    pub params: &'a SystemParams,
    pub num_proofs: usize,
}

pub(in crate::whir) struct FoldingGpuTraceGenerator;

impl ModuleChip<GpuBackend> for FoldingGpuTraceGenerator {
    type Ctx<'a> = FoldingGpuCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_proving_ctx(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<AirProvingContext<GpuBackend>> {
        let blob = ctx.blob;
        let params = ctx.params;
        let num_proofs = ctx.num_proofs;

        let mem = MemTracker::start("tracegen.whir_folding");
        let num_rounds = params.num_whir_rounds();
        let num_queries_per_round = num_queries_per_round(params);
        let total_queries = total_num_queries(&num_queries_per_round);
        let k_whir = params.k_whir();
        let internal_nodes = (1 << k_whir) - 1;
        let num_rows_per_proof = total_queries * internal_nodes;
        let num_valid_rows = num_rows_per_proof * num_proofs;
        debug_assert_eq!(blob.folding_records.len(), num_valid_rows);

        let height = if let Some(h) = required_height {
            if h < num_valid_rows {
                return None;
            }
            h
        } else {
            num_valid_rows.next_power_of_two()
        };
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
        Some(AirProvingContext::simple_no_pis(trace))
    }
}
