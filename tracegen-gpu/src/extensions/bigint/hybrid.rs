use openvm_bigint_circuit::Rv32LessThan256Chip;
use openvm_circuit::arch::{Arena, DenseRecordArena, EmptyAdapterCoreLayout, MatrixRecordArena};
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_rv32_adapters::{Rv32HeapAdapterStep, Rv32VecHeapAdapterCols, Rv32VecHeapAdapterRecord};
use openvm_rv32im_circuit::{adapters::INT256_NUM_LIMBS, LessThanCoreCols};
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use stark_backend_gpu::{prover_backend::GpuBackend, types::F};

use crate::{
    cpu_proving_ctx_to_gpu,
    extensions::bigint::{LessThan256AdapterRecord, LessThan256CoreRecord},
    get_empty_air_proving_ctx,
};

#[derive(derive_new::new)]
pub struct HybridLessThan256Chip {
    pub cpu: Rv32LessThan256Chip<F>,
}

impl Chip<DenseRecordArena, GpuBackend> for HybridLessThan256Chip {
    fn generate_proving_ctx(
        &self,
        mut dense_arena: DenseRecordArena,
    ) -> AirProvingContext<GpuBackend> {
        if dense_arena.allocated().is_empty() {
            return get_empty_air_proving_ctx();
        }

        let record_size = size_of::<(LessThan256AdapterRecord, LessThan256CoreRecord)>();
        let trace_width = LessThanCoreCols::<F, INT256_NUM_LIMBS, RV32_CELL_BITS>::width()
            + Rv32VecHeapAdapterCols::<F, 2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>::width();
        let rows_used = dense_arena.allocated().len().div_ceil(record_size);
        let height = rows_used.next_power_of_two();
        let mut seeker = dense_arena.get_record_seeker::<(
            &mut Rv32VecHeapAdapterRecord<2, 1, 1, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
            &mut LessThan256CoreRecord,
        ), EmptyAdapterCoreLayout<
            F,
            Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        >>();
        let mut matrix_arena = MatrixRecordArena::<F>::with_capacity(height, trace_width);
        seeker.transfer_to_matrix_arena(&mut matrix_arena, EmptyAdapterCoreLayout::new());
        let ctx = self.cpu.generate_proving_ctx(matrix_arena);
        cpu_proving_ctx_to_gpu(ctx)
    }
}
