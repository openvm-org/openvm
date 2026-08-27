use openvm_recursion_circuit::prelude::F;
use openvm_stark_backend::{
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    prover::{AirProvingContext, MatrixDimensions, ProverBackend, ProverDevice, ProvingContext},
    AirRef, EngineDeviceCtx, StarkEngine, StarkProtocolConfig,
};

use crate::circuit::Circuit;

/// Panics if any AIR in the proving key is not marked as required.
pub fn assert_all_airs_required<SC: StarkProtocolConfig>(pk: &MultiStarkProvingKey<SC>) {
    assert!(
        pk.per_air.iter().all(|air_pk| air_pk.vk.is_required),
        "cached proving key must have all AIRs marked as required"
    );
}

/// Generates keys with every AIR marked as required.
pub fn keygen_all_required<E: StarkEngine>(
    engine: &E,
    airs: &[AirRef<E::SC>],
) -> (MultiStarkProvingKey<E::SC>, MultiStarkVerifyingKey<E::SC>) {
    let mut keygen_builder = MultiStarkKeygenBuilder::new(engine.config().clone());
    for air in airs {
        keygen_builder.add_required_air(air.clone());
    }
    let pk = keygen_builder.generate_pk().unwrap();
    let vk = pk.get_vk();
    (pk, vk)
}

#[cfg(debug_assertions)]
pub(crate) fn debug_checks_enabled() -> bool {
    std::env::var("OPENVM_SKIP_DEBUG") != Ok(String::from("1"))
}

pub fn engine_device_ctx<E>(engine: &E) -> &EngineDeviceCtx<E>
where
    E: StarkEngine,
{
    engine.device().device_ctx()
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
