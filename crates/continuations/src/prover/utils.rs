#[cfg(feature = "cuda")]
use openvm_cpu_backend::CpuDevice;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::device::GpuDevice;
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::DeviceContext;
use openvm_recursion_circuit::prelude::F;
#[cfg(feature = "cuda")]
use openvm_stark_backend::prover::ReferenceDevice;
use openvm_stark_backend::{
    prover::{AirProvingContext, MatrixDimensions, ProverBackend, ProvingContext},
    AirRef, StarkEngine, StarkProtocolConfig,
};

use crate::circuit::Circuit;

pub(crate) fn debug_checks_enabled() -> bool {
    std::env::var("OPENVM_SKIP_DEBUG") != Ok(String::from("1"))
}

#[cfg(feature = "cuda")]
pub trait MaybeDeviceContext {
    fn maybe_device_ctx(&self) -> Option<&DeviceContext>;
}

#[cfg(feature = "cuda")]
impl MaybeDeviceContext for GpuDevice {
    fn maybe_device_ctx(&self) -> Option<&DeviceContext> {
        Some(&self.ctx)
    }
}

#[cfg(feature = "cuda")]
impl<SC> MaybeDeviceContext for CpuDevice<SC> {
    fn maybe_device_ctx(&self) -> Option<&DeviceContext> {
        None
    }
}

#[cfg(feature = "cuda")]
impl<SC> MaybeDeviceContext for ReferenceDevice<SC> {
    fn maybe_device_ctx(&self) -> Option<&DeviceContext> {
        None
    }
}

#[cfg(feature = "cuda")]
pub fn device_ctx_for_engine<E>(engine: &E) -> Option<&DeviceContext>
where
    E: StarkEngine,
    E::PD: MaybeDeviceContext,
{
    engine.device().maybe_device_ctx()
}

pub fn debug_constraints<SC, C, E>(circuit: &C, ctx: &ProvingContext<E::PB>, engine: &E)
where
    SC: StarkProtocolConfig<F = F>,
    C: Circuit<SC>,
    E: StarkEngine<SC = SC>,
{
    let airs = circuit.airs();
    trace_heights_tracing_info(&ctx.per_trace, &airs);
    engine.debug(&airs, ctx);
}

pub(crate) fn trace_heights_tracing_info<PB: ProverBackend, SC: StarkProtocolConfig>(
    ctxs: &[(usize, AirProvingContext<PB>)],
    airs: &[AirRef<SC>],
) {
    let mut total_cells = 0usize;
    let mut total_width = 0usize;
    for ((_, ctx), air) in ctxs.iter().zip(airs) {
        let cells = ctx.common_main.height() * ctx.common_main.width();
        tracing::info!(
            "{:<40} | Height: {:>8} | Width: {:>8} | Cells: {:>8}",
            air.name(),
            ctx.common_main.height(),
            ctx.common_main.width(),
            cells
        );
        total_cells += cells;
        total_width += ctx.common_main.width();
    }
    tracing::info!("Total Common Cells: {total_cells}");
    tracing::info!("Total Width: {total_width}");
}
