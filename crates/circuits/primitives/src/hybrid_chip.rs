use std::marker::PhantomData;

use openvm_cpu_backend::CpuBackend;
use openvm_cuda_backend::{
    base::DeviceMatrix, hash_scheme::GpuHashScheme, prelude::SC, GenericGpuBackend, GpuBackend,
};
use openvm_cuda_common::{copy::MemCopyH2D, stream::GpuDeviceCtx};
use openvm_stark_backend::prover::{AirProvingContext, ColMajorMatrix, MatrixDimensions};

use crate::Chip;

pub fn get_empty_air_proving_ctx<HS: GpuHashScheme>() -> AirProvingContext<GenericGpuBackend<HS>> {
    AirProvingContext {
        cached_mains: vec![],
        common_main: DeviceMatrix::dummy(),
        public_values: vec![],
    }
}

// Wraps a CPU chip for use with GpuBackend
pub struct HybridChip<RA, C: Chip<RA, CpuBackend<SC>>> {
    pub cpu_chip: C,
    pub device_ctx: GpuDeviceCtx,
    _marker: PhantomData<RA>,
}

impl<RA, C: Chip<RA, CpuBackend<SC>>> HybridChip<RA, C> {
    pub fn new(cpu_chip: C, device_ctx: GpuDeviceCtx) -> Self {
        Self {
            cpu_chip,
            device_ctx,
            _marker: PhantomData,
        }
    }
}

impl<RA, C: Chip<RA, CpuBackend<SC>>> Chip<RA, GpuBackend> for HybridChip<RA, C> {
    fn generate_proving_ctx(&self, arena: RA) -> AirProvingContext<GpuBackend> {
        let ctx = self.cpu_chip.generate_proving_ctx(arena);
        cpu_proving_ctx_to_gpu(ctx, &self.device_ctx)
    }
}

pub fn cpu_proving_ctx_to_gpu<HS: GpuHashScheme>(
    cpu_ctx: AirProvingContext<CpuBackend<SC>>,
    device_ctx: &GpuDeviceCtx,
) -> AirProvingContext<GenericGpuBackend<HS>> {
    assert!(
        cpu_ctx.cached_mains.is_empty(),
        "CPU to GPU transfer of cached traces not supported"
    );
    let cm = ColMajorMatrix::from_row_major(&cpu_ctx.common_main);
    // Safety: sync_stream(...) is not needed because the source is pageable memory and the CUDA
    // runtime guarantees that cudaMemcpyAsync from a pageable host buffer returns only after the
    // data has been staged into the driver's internal pinned buffers which avoids any
    // use-after-free issues.
    let (height, width) = (cm.height(), cm.width());
    let buffer = cm.values.to_device_on(device_ctx).unwrap();
    let trace = DeviceMatrix::new(std::sync::Arc::new(buffer), height, width);
    AirProvingContext {
        cached_mains: vec![],
        common_main: trace,
        public_values: cpu_ctx.public_values,
    }
}
