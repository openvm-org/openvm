use openvm_recursion_circuit::prelude::F;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    prover::{
        AirProvingContext, DeviceDataTransporter, DeviceMultiStarkProvingKey,
        MatrixDimensions, ProverBackend, ProvingContext,
    },
    AirRef, StarkEngine, StarkProtocolConfig,
};

use crate::circuit::Circuit;

pub(crate) fn debug_checks_enabled() -> bool {
    std::env::var("OPENVM_SKIP_DEBUG") != Ok(String::from("1"))
}

pub(crate) fn keygen_for_proving_backend<SC, E, Keygen>(
    engine: &E,
    airs: &[AirRef<SC>],
    cuda_keygen: Keygen,
) -> (MultiStarkProvingKey<SC>, MultiStarkVerifyingKey<SC>)
where
    SC: StarkProtocolConfig<F = F>,
    E: StarkEngine<SC = SC>,
    Keygen: FnOnce() -> (MultiStarkProvingKey<SC>, MultiStarkVerifyingKey<SC>),
{
    #[cfg(feature = "cuda")]
    {
        let _ = (engine, airs);
        cuda_keygen()
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = cuda_keygen;
        engine.keygen(airs)
    }
}

pub(crate) fn transport_pk<E>(
    engine: &E,
    pk: &MultiStarkProvingKey<E::SC>,
) -> DeviceMultiStarkProvingKey<E::PB>
where
    E: StarkEngine,
    E::PD: DeviceDataTransporter<E::SC, E::PB>,
{
    engine.device().transport_pk_to_device(pk)
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
