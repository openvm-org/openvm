use openvm_stark_backend::{
    prover::{AirProvingContext, MatrixDimensions, ProverBackend, ProvingContext},
    AirRef, StarkEngine, StarkProtocolConfig,
};
use recursion_circuit::prelude::F;

use crate::circuit::Circuit;

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
