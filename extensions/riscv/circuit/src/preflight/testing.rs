use openvm_circuit::arch::{cuda::postflight::GpuPostflightError, MemoryConfig};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::program::Program;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{PostflightAccessRegistry, PreflightReplayProgram};

impl PreflightReplayProgram {
    pub fn upload<F: PrimeField32>(
        program: &Program<F>,
        memory_config: &MemoryConfig,
        device_ctx: &GpuDeviceCtx,
    ) -> Result<Self, GpuPostflightError> {
        Self::upload_with_postflight_access_registry(
            program,
            memory_config,
            &PostflightAccessRegistry::default(),
            device_ctx,
        )
    }
}
