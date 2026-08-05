use openvm_circuit::arch::{cuda::postflight::GpuPostflightError, MemoryConfig};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_instructions::program::Program;

use super::{PostflightAccessRegistry, PreflightReplayProgram};

impl PreflightReplayProgram {
    pub fn upload(
        program: &Program,
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
