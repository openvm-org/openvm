use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{
    AirRef,
    prover::{MatrixDimensions, types::AirProofRawInput},
};
use openvm_stark_sdk::{
    config::{
        FriParameters,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::{StarkEngine, StarkFriEngine},
};
use recursion_circuit::system::AggregationSubCircuit;
use stark_backend_v2::{
    F, StarkWhirEngine,
    prover::{
        AirProvingContextV2, ColMajorMatrix, DeviceDataTransporterV2, ProverBackendV2,
        StridedColMajorMatrixView,
    },
};

use crate::aggregation::AggregationCircuit;

pub fn debug_constraints<S, E>(
    circuit: &AggregationCircuit<S>,
    ctxs: &[(usize, AirProvingContextV2<E::PB>)],
    engine: &E,
) where
    S: AggregationSubCircuit,
    E: StarkWhirEngine,
{
    let device = engine.device();
    let transpose = |mat: ColMajorMatrix<F>| {
        Arc::new(StridedColMajorMatrixView::from(mat.as_view()).to_row_major_matrix())
    };
    let cpu_engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let inputs = ctxs
        .iter()
        .map(|(_, ctx)| {
            let common_main = device.transport_matrix_from_device_to_host(&ctx.common_main);
            AirProofRawInput {
                cached_mains: ctx
                    .cached_mains
                    .iter()
                    .map(|cd| transpose(device.transport_matrix_from_device_to_host(&cd.trace)))
                    .collect_vec(),
                common_main: Some(transpose(common_main)),
                public_values: ctx.public_values.clone(),
            }
        })
        .collect_vec();
    let mut keygen_builder = cpu_engine.keygen_builder();
    let airs = circuit.airs();
    for air in &airs {
        keygen_builder.add_air(air.clone());
    }
    trace_heights_tracing_info(ctxs, &airs);
    cpu_engine.debug(&airs, &keygen_builder.generate_pk().per_air, &inputs);
}

pub(crate) fn trace_heights_tracing_info<PB: ProverBackendV2>(
    ctxs: &[(usize, AirProvingContextV2<PB>)],
    airs: &[AirRef<BabyBearPoseidon2Config>],
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
