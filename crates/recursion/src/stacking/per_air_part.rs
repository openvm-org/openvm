use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage,
        StackingWidthBusMessage, StackingWidthsBus,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct DummyPerAirPartCols<T> {
    is_valid: T,
    tidx: T,
    sorted_air_idx: T,
    air_idx: T,
    part_idx: T,
    is_present: T,
    hypercube_dim: T,
    has_preprocessed: T,
    num_main_parts: T,
    num_interactions: T,
    width: T,
    commit_idx: T,
    commit_width: T,
    is_summary: T,
}

// Temporary dummy AIR to represent this module.
pub struct DummyPerAirPartAir {
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub stacking_widths_bus: StackingWidthsBus,
}

impl BaseAirWithPublicValues<F> for DummyPerAirPartAir {}
impl PartitionedBaseAir<F> for DummyPerAirPartAir {}

impl<F> BaseAir<F> for DummyPerAirPartAir {
    fn width(&self) -> usize {
        DummyPerAirPartCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyPerAirPartAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &DummyPerAirPartCols<AB::Var> = (*local).borrow();
        let next: &DummyPerAirPartCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_one(local.is_valid);
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        self.air_shape_bus.receive(
            builder,
            AirShapeBusMessage {
                sort_idx: local.sorted_air_idx,
                idx: local.air_idx,
                hypercube_dim: local.hypercube_dim,
                has_preprocessed: local.has_preprocessed,
                num_main_parts: local.num_main_parts,
                num_interactions: local.num_interactions,
            },
            local.is_valid - local.is_summary,
        );
        self.air_part_shape_bus.receive(
            builder,
            AirPartShapeBusMessage {
                idx: local.air_idx,
                part: local.part_idx,
                width: local.width,
            },
            local.is_valid - local.is_summary,
        );
        self.stacking_widths_bus.receive(
            builder,
            StackingWidthBusMessage {
                commit_idx: local.commit_idx,
                width: local.commit_width,
            },
            local.is_summary,
        );
    }
}

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let vk = &vk.inner;

    let num_parts_per_air: Vec<_> = vk
        .per_air
        .iter()
        .map(|avk| 1 + avk.num_cached_mains() + avk.preprocessed_data.is_some() as usize)
        .collect();

    let num_valid_rows = num_parts_per_air.iter().sum::<usize>() + 1;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyPerAirPartCols::<usize>::width();

    let mut j = 0;

    let mut trace = vec![F::ZERO; num_rows * width];
    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut DummyPerAirPartCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;

        if i == num_valid_rows - 1 {
            cols.is_summary = F::ONE;
            cols.commit_idx = F::ZERO;
            cols.commit_width = F::from_canonical_usize(preflight.stacked_common_width);
        } else {
            let (air_id, shape) = &preflight.sorted_trace_shapes[j];
            let avk = &vk.per_air[*air_id];

            cols.tidx = F::ZERO;
            cols.air_idx = F::from_canonical_usize(*air_id);
            cols.sorted_air_idx = F::from_canonical_usize(j);
            cols.has_preprocessed = F::from_bool(avk.preprocessed_data.is_some());
            cols.hypercube_dim = F::from_canonical_usize(shape.hypercube_dim);
            cols.num_main_parts = F::from_canonical_usize(avk.num_cached_mains() + 1);
            cols.num_interactions =
                F::from_canonical_usize(avk.symbolic_constraints.interactions.len());
            cols.width = F::from_canonical_usize(avk.params.width.common_main);
        }
    }

    RowMajorMatrix::new(trace, width)
}
