use openvm_stark_backend::{
    prover::{AirProvingContext, MatrixDimensions, ProverBackend, ProvingContext},
    AirRef, StarkEngine,
};

use crate::{aggregation::Circuit, SC};

pub fn debug_constraints<C, E>(circuit: &C, ctx: &ProvingContext<E::PB>, engine: &E)
where
    C: Circuit,
    E: StarkEngine<SC = SC>,
{
    let airs = circuit.airs();
    trace_heights_tracing_info(&ctx.per_trace, &airs);
    engine.debug(&airs, ctx);
}

pub(crate) fn trace_heights_tracing_info<PB: ProverBackend>(
    ctxs: &[(usize, AirProvingContext<PB>)],
    airs: &[AirRef<SC>],
) {
    let mut total_cells = 0usize;
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
    }
    tracing::info!("Total Common Cells: {total_cells}");
}
