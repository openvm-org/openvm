pub mod air;
pub mod columns;
pub mod execution;
#[cfg(test)]
pub mod tests;
mod trace;

use std::sync::Arc;

use openvm_circuit::{
    arch::{RowMajorMatrixArena, TraceFiller},
    system::memory::SharedMemoryHelper,
};
use openvm_circuit_primitives::bitwise_op_lookup::SharedBitwiseOperationLookupChip;
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    p3_field::PrimeField32,
    prover::{cpu::CpuBackend, types::AirProvingContext},
    Chip,
};

pub const NUM_KECCAKF_OP_ROWS: usize = 2;

#[derive(derive_new::new, Clone, Copy)]
pub struct KeccakfVmExecutor {
    pub offset: usize,
    pub pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct KeccakfVmFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<8>,
    pub pointer_max_bits: usize,
}

#[derive(derive_new::new)]
pub struct KeccakfOpChip<F> {
    pub inner: KeccakfVmFiller,
    pub mem_helper: SharedMemoryHelper<F>,
}

impl<SC, RA> Chip<RA, CpuBackend<SC>> for KeccakfOpChip<Val<SC>>
where
    SC: StarkGenericConfig,
    Val<SC>: PrimeField32,
    RA: RowMajorMatrixArena<Val<SC>>,
{
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<CpuBackend<SC>> {
        let rows_used = arena.trace_offset() / arena.width();
        let mut trace = arena.into_matrix();
        let mem_helper = self.mem_helper.as_borrowed();
        self.inner.fill_trace(&mem_helper, &mut trace, rows_used);
        AirProvingContext::simple(Arc::new(trace), self.inner.generate_public_values())
    }
}

pub type KeccakfVmChip<F> = KeccakfOpChip<F>;
