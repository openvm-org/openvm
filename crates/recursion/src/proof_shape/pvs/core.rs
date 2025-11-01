use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{AlignedBorrow, SubAir, utils::not};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{F, proof::Proof};

use crate::{
    bus::{PublicValuesBus, TranscriptBus, TranscriptBusMessage},
    proof_shape::bus::{NumPublicValuesBus, NumPublicValuesMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct PublicValuesCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub is_first: F,

    pub air_idx: F,
    pub tidx: F,
    pub num_pvs_left: F,

    pub is_first_for_air: F,
    pub value: F,
}

pub(crate) fn generate_trace(proofs: &[Proof], preflights: &[Preflight]) -> RowMajorMatrix<F> {
    let total_num_pvs: usize = proofs
        .iter()
        .map(|proof| {
            proof
                .public_values
                .iter()
                .fold(0usize, |acc, per_air| acc + per_air.len())
        })
        .sum();
    let num_rows = total_num_pvs.next_power_of_two();
    let width = PublicValuesCols::<u8>::width();

    debug_assert_eq!(proofs.len(), preflights.len());

    let mut trace = vec![F::ZERO; num_rows * width];
    let mut chunks = trace.chunks_exact_mut(width);

    for (proof_idx, (proof, preflight)) in proofs.iter().zip(preflights).enumerate() {
        let mut row_idx = 0usize;

        for ((air_idx, pvs), &starting_tidx) in proof
            .public_values
            .iter()
            .enumerate()
            .filter(|(_, per_air)| !per_air.is_empty())
            .zip(&preflight.proof_shape.pvs_tidx)
        {
            let mut tidx = starting_tidx;
            let mut num_pvs_left = pvs.len();

            for (pv_idx, pv) in pvs.iter().enumerate() {
                let chunk = chunks.next().unwrap();
                let cols: &mut PublicValuesCols<F> = chunk.borrow_mut();

                cols.proof_idx = F::from_canonical_usize(proof_idx);
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(row_idx == 0);

                cols.air_idx = F::from_canonical_usize(air_idx);
                cols.tidx = F::from_canonical_usize(tidx);
                cols.num_pvs_left = F::from_canonical_usize(num_pvs_left);

                cols.is_first_for_air = F::from_bool(pv_idx == 0);
                cols.value = *pv;

                tidx += 1;
                num_pvs_left -= 1;
                row_idx += 1;
            }
        }
    }

    for chunk in chunks {
        let cols: &mut PublicValuesCols<F> = chunk.borrow_mut();
        cols.proof_idx = F::from_canonical_usize(proofs.len());
    }

    RowMajorMatrix::new(trace, width)
}

pub struct PublicValuesAir {
    pub _public_values_bus: PublicValuesBus,
    pub num_pvs_bus: NumPublicValuesBus,
    pub transcript_bus: TranscriptBus,
}

impl<F> BaseAir<F> for PublicValuesAir {
    fn width(&self) -> usize {
        PublicValuesCols::<F>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for PublicValuesAir {}
impl<F> PartitionedBaseAir<F> for PublicValuesAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for PublicValuesAir
where
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &PublicValuesCols<AB::Var> = (*local).borrow();
        let next: &PublicValuesCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1, 0> {}.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid,
                        counter: [local.proof_idx],
                        is_first: [local.is_first],
                    }
                    .map_into(),
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid,
                        counter: [next.proof_idx],
                        is_first: [next.is_first],
                    }
                    .map_into(),
                ),
                NestedForLoopAuxCols { is_transition: [] },
            ),
        );

        // Constrain is_first_for_air, send NumPublicValuesBus message when true
        builder.assert_bool(local.is_first_for_air);
        builder
            .when(local.is_first)
            .assert_one(local.is_first_for_air);
        builder
            .when(local.is_first_for_air)
            .assert_one(local.is_valid);
        builder
            .when(next.is_valid * (next.air_idx - local.air_idx))
            .assert_one(next.is_first_for_air);

        self.num_pvs_bus.receive(
            builder,
            local.proof_idx,
            NumPublicValuesMessage {
                air_idx: local.air_idx,
                tidx: local.tidx,
                num_pvs: local.num_pvs_left,
            },
            local.is_first_for_air,
        );

        // Constrain rows for the same AIR are in the correct order
        let mut when_same_air =
            builder.when(local.is_valid * next.is_valid * not(next.is_first_for_air));
        when_same_air.assert_eq(local.air_idx, next.air_idx);
        when_same_air.assert_one(local.num_pvs_left - next.num_pvs_left);
        when_same_air.assert_one(next.tidx - local.tidx);

        // Receive transcript read of public values
        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            TranscriptBusMessage {
                tidx: local.tidx.into(),
                value: local.value.into(),
                is_sample: AB::Expr::ZERO,
            },
            local.is_valid,
        );
    }
}
