use std::borrow::Borrow;

use openvm_circuit_primitives::{utils::not, AlignedBorrow, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;

use crate::{
    bus::{PublicValuesBus, PublicValuesBusMessage, TranscriptBus, TranscriptBusMessage},
    proof_shape::bus::{NumPublicValuesBus, NumPublicValuesMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
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

    pub tidx: F,
    pub value: F,
}

pub struct PublicValuesAir {
    pub public_values_bus: PublicValuesBus,
    pub num_pvs_bus: NumPublicValuesBus,
    pub transcript_bus: TranscriptBus,
    pub(crate) continuations_enabled: bool,
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

        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &PublicValuesCols<AB::Var> = (*local).borrow();
        let next: &PublicValuesCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<1> {}.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_valid,
                    counter: [local.proof_idx],
                    is_first: [local.is_first_in_proof],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_valid,
                    counter: [next.proof_idx],
                    is_first: [next.is_first_in_proof],
                }
                .map_into(),
            ),
        );
        // Constrain is_first_for_air, send NumPublicValuesBus message when true
        builder.assert_bool(local.is_first_in_air);
        builder
            .when(local.is_first_in_proof)
            .assert_one(local.is_first_in_air);
        builder
            .when(local.is_first_in_air)
            .assert_one(local.is_valid);
        builder
            .when(next.is_valid * (next.air_idx - local.air_idx))
            .assert_one(next.is_first_in_air);

        let is_same_air = local.is_valid * next.is_valid * not(next.is_first_in_air);
        self.num_pvs_bus.receive(
            builder,
            local.proof_idx,
            NumPublicValuesMessage {
                air_idx: local.air_idx.into(),
                tidx: local.tidx - local.pv_idx,
                num_pvs: local.pv_idx + AB::Expr::ONE,
            },
            local.is_valid - is_same_air.clone(),
        );

        let mut when_same_air = builder.when(is_same_air);
        when_same_air.assert_eq(local.air_idx, next.air_idx);
        when_same_air.assert_eq(next.pv_idx, local.pv_idx + AB::Expr::ONE);
        when_same_air.assert_eq(next.tidx, local.tidx + AB::Expr::ONE);

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
        if self.continuations_enabled {
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
        }

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
