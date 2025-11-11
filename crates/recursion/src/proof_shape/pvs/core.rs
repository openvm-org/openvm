use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{AlignedBorrow, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{F, proof::Proof};

use crate::{
    bus::{PublicValuesBus, PublicValuesBusMessage, TranscriptBus, TranscriptBusMessage},
    proof_shape::bus::{NumPublicValuesBus, NumPublicValuesMessage},
    subairs::nested_for_loop::{NestedForLoopAuxCols, NestedForLoopIoCols, NestedForLoopSubAir},
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct PublicValuesCols<F> {
    pub is_valid: F,

    pub proof_idx: F,
    pub air_idx: F,
    pub pv_idx: F,

    pub is_first_in_proof: F,
    pub is_first_in_air: F,

    pub is_same_proof: F,
    pub is_same_air: F,

    pub tidx: F,
    pub value: F,
}

pub(in crate::proof_shape) fn generate_trace(
    proofs: &[Proof],
    preflights: &[Preflight],
) -> RowMajorMatrix<F> {
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
        for ((air_idx, pvs), &starting_tidx) in proof
            .public_values
            .iter()
            .enumerate()
            .filter(|(_, per_air)| !per_air.is_empty())
            .zip(&preflight.proof_shape.pvs_tidx)
        {
            let mut tidx = starting_tidx;

            for (pv_idx, pv) in pvs.iter().enumerate() {
                let chunk = chunks.next().unwrap();
                let cols: &mut PublicValuesCols<F> = chunk.borrow_mut();

                cols.is_valid = F::ONE;

                cols.proof_idx = F::from_canonical_usize(proof_idx);
                cols.air_idx = F::from_canonical_usize(air_idx);
                cols.pv_idx = F::from_canonical_usize(pv_idx);

                cols.is_first_in_air = F::from_bool(pv_idx == 0);
                cols.is_first_in_proof = F::from_bool(pv_idx == 0 && air_idx == 0);

                cols.is_same_air = F::from_bool(pv_idx + 1 < pvs.len());
                cols.is_same_proof =
                    F::from_bool(pv_idx + 1 < pvs.len() || air_idx + 1 < proof.public_values.len());

                cols.tidx = F::from_canonical_usize(tidx);
                cols.value = *pv;

                tidx += 1;
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
    pub public_values_bus: PublicValuesBus,
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

        NestedForLoopSubAir::<3, 2>.eval(
            builder,
            (
                (
                    NestedForLoopIoCols {
                        is_enabled: local.is_valid.into(),
                        counter: [
                            local.proof_idx.into(),
                            local.air_idx.into(),
                            local.pv_idx.into(),
                        ],
                        is_first: [
                            local.is_first_in_proof.into(),
                            local.is_first_in_air.into(),
                            AB::Expr::ONE,
                        ],
                    },
                    NestedForLoopIoCols {
                        is_enabled: next.is_valid.into(),
                        counter: [
                            next.proof_idx.into(),
                            next.air_idx.into(),
                            next.pv_idx.into(),
                        ],
                        is_first: [
                            next.is_first_in_proof.into(),
                            next.is_first_in_air.into(),
                            AB::Expr::ONE,
                        ],
                    },
                ),
                NestedForLoopAuxCols {
                    is_transition: [local.is_same_proof, local.is_same_air],
                }
                .map_into(),
            ),
        );

        builder.assert_bool(local.is_valid);
        builder
            .when(local.is_same_air)
            .assert_eq(next.tidx, local.tidx + AB::Expr::ONE);

        self.num_pvs_bus.receive(
            builder,
            local.proof_idx,
            NumPublicValuesMessage {
                air_idx: local.air_idx.into(),
                tidx: local.tidx - local.pv_idx,
                num_pvs: local.pv_idx + AB::Expr::ONE,
            },
            local.is_valid - local.is_same_air,
        );
        self.public_values_bus.send(
            builder,
            local.proof_idx,
            PublicValuesBusMessage {
                air_idx: local.air_idx,
                pv_idx: local.pv_idx,
                value: local.value,
            },
            local.is_valid,
        );

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
