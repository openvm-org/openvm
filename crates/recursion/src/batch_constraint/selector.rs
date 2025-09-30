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

use crate::bus::{AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage};

#[repr(C)]
#[derive(AlignedBorrow)]
struct BatchConstraintSelectorCols<T> {
    is_valid: T,
    tidx: T,
    sorted_air_idx: T,
    air_idx: T,
    is_present: T,
    hypercube_dim: T,
    has_preprocessed: T,
    num_main_parts: T,
    num_interactions: T,
    common_width: T,
}

// Temporary dummy AIR to represent this module.
pub struct BatchConstraintSelectorAir {
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
}

impl BaseAirWithPublicValues<F> for BatchConstraintSelectorAir {}
impl PartitionedBaseAir<F> for BatchConstraintSelectorAir {}

impl<F> BaseAir<F> for BatchConstraintSelectorAir {
    fn width(&self) -> usize {
        BatchConstraintSelectorCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for BatchConstraintSelectorAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &BatchConstraintSelectorCols<AB::Var> = (*local).borrow();
        let next: &BatchConstraintSelectorCols<AB::Var> = (*next).borrow();

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
                is_present: local.is_present,
                hypercube_dim: local.hypercube_dim,
                has_preprocessed: local.has_preprocessed,
                num_main_parts: local.num_main_parts,
                num_interactions: local.num_interactions,
            },
            local.is_valid,
        );
        // TODO: preprocessed/cached commits
        self.air_part_shape_bus.receive(
            builder,
            AirPartShapeBusMessage {
                idx: local.air_idx.into(),
                part: AB::Expr::ZERO,
                width: local.common_width.into(),
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace(vk: &MultiStarkVerifyingKeyV2, proof: &Proof) -> RowMajorMatrix<F> {
    let vk = &vk.inner;

    let num_optional_airs: usize = proof
        .is_optional_air_present
        .iter()
        .map(|x| 1 - *x as usize)
        .sum();

    let num_valid_rows: usize = vk.per_air.len() - num_optional_airs;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = BatchConstraintSelectorCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let avk = &vk.per_air[i];

        let cols: &mut BatchConstraintSelectorCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.tidx = F::ZERO;
        cols.air_idx = F::from_canonical_usize(i);
        cols.sorted_air_idx = F::from_canonical_usize(i);
        cols.is_present = F::from_bool(proof.is_optional_air_present[i]);
        cols.has_preprocessed = F::from_bool(avk.preprocessed_data.is_some());
        cols.hypercube_dim = F::from_canonical_u8(proof.log_heights[i]);
        cols.num_main_parts = F::from_canonical_usize(avk.num_cached_mains() + 1);
        cols.num_interactions =
            F::from_canonical_usize(avk.symbolic_constraints.interactions.len());
        cols.common_width = F::from_canonical_usize(avk.params.width.common_main);
    }

    RowMajorMatrix::new(trace, width)
}
