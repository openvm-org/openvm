use std::marker::PhantomData;

use openvm_cuda_backend::{
    base::DeviceMatrix, data_transporter::transport_matrix_h2d_col_major, prelude::SC, GpuBackend,
};
use openvm_stark_backend::prover::{AirProvingContext, CpuBackend};

use crate::Chip;

pub fn get_empty_air_proving_ctx() -> AirProvingContext<GpuBackend> {
    AirProvingContext {
        cached_mains: vec![],
        common_main: DeviceMatrix::dummy(),
        public_values: vec![],
    }
}

// Wraps a CPU chip for use with GpuBackend
pub struct HybridChip<RA, C: Chip<RA, CpuBackend<SC>>> {
    pub cpu_chip: C,
    _marker: PhantomData<RA>,
}

impl<RA, C: Chip<RA, CpuBackend<SC>>> HybridChip<RA, C> {
    pub fn new(cpu_chip: C) -> Self {
        Self {
            cpu_chip,
            _marker: PhantomData,
        }
    }
}

impl<RA, C: Chip<RA, CpuBackend<SC>>> Chip<RA, GpuBackend> for HybridChip<RA, C> {
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<GpuBackend> {
        let ctx = self.cpu_chip.generate_proving_ctx(arena);
        cpu_proving_ctx_to_gpu(ctx)
    }
}

pub fn cpu_proving_ctx_to_gpu(
    cpu_ctx: AirProvingContext<CpuBackend<SC>>,
) -> AirProvingContext<GpuBackend> {
    assert!(
        cpu_ctx.cached_mains.is_empty(),
        "CPU to GPU transfer of cached traces not supported"
    );
    let trace = transport_matrix_h2d_col_major(&cpu_ctx.common_main).unwrap();
    AirProvingContext {
        cached_mains: vec![],
        common_main: trace,
        public_values: cpu_ctx.public_values,
    }
}
