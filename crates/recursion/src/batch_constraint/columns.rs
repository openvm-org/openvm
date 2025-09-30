use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{ColumnClaimsBus, ColumnClaimsMessage},
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct DummyPerColumnCols<T> {
    is_valid: T,
    idx: T,
    air_idx: T,
    part_idx: T,
    col_idx: T,
    col_claim: [T; D_EF],
    rot_claim: [T; D_EF],
}

// Temporary dummy AIR to represent this module.
pub struct DummyPerColumnAir {
    pub column_claims_bus: ColumnClaimsBus,
}

impl BaseAirWithPublicValues<F> for DummyPerColumnAir {}
impl PartitionedBaseAir<F> for DummyPerColumnAir {}

impl<F> BaseAir<F> for DummyPerColumnAir {
    fn width(&self) -> usize {
        DummyPerColumnCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyPerColumnAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &DummyPerColumnCols<AB::Var> = (*local).borrow();
        let next: &DummyPerColumnCols<AB::Var> = (*next).borrow();

        builder.when_first_row().assert_one(local.is_valid);
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(next.is_valid)
            .assert_one(local.is_valid);

        self.column_claims_bus.send(
            builder,
            ColumnClaimsMessage {
                idx: local.idx,
                air_idx: local.air_idx,
                part_idx: local.part_idx,
                col_idx: local.col_idx,
                col_claim: local.col_claim,
                rot_claim: local.rot_claim,
            },
            local.is_valid,
        );
    }
}

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    _proof: &Proof,
    _preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let vk = &vk.inner;

    // TODO: optional airs
    let num_valid_rows: usize = vk
        .per_air
        .iter()
        .map(|avk| avk.params.width.total_width(0))
        .sum::<usize>();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyPerColumnCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    let mut air_idx = 0;
    let mut part_idx = 0;
    let mut col_idx = 0;
    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let avk = &vk.per_air[air_idx];

        let num_parts = 1 + avk.num_cached_mains() + avk.preprocessed_data.is_some() as usize;

        let cols: &mut DummyPerColumnCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.idx = F::from_canonical_usize(i);
        cols.air_idx = F::from_canonical_usize(air_idx);
        cols.col_idx = F::from_canonical_usize(col_idx);

        let part_width = if part_idx == 0 {
            avk.params.width.common_main
        } else if part_idx == 1 && avk.preprocessed_data.is_some() {
            avk.params.width.preprocessed.unwrap()
        } else {
            avk.params.width.cached_mains[part_idx - 1 - avk.preprocessed_data.is_some() as usize]
        };
        cols.part_idx = F::from_canonical_usize(part_idx);

        col_idx += 1;
        if col_idx == part_width {
            part_idx += 1;
            col_idx = 0;
        }
        if part_idx == num_parts {
            air_idx += 1;
            part_idx = 0;
            col_idx = 0;
        }
    }

    RowMajorMatrix::new(trace, width)
}
