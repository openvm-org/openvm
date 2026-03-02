use std::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use recursion_circuit::bus::{PublicValuesBus, PublicValuesBusMessage};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::circuit::inner::bus::{PvsAirConsistencyBus, PvsAirConsistencyMessage};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UnsetPvsCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
}

pub struct UnsetPvsAir {
    pub public_values_bus: PublicValuesBus,
    pub pvs_air_consistency_bus: PvsAirConsistencyBus,
    pub air_idx: usize,
    pub num_pvs: usize,
    pub def_flag: u32,
}

impl<F> BaseAir<F> for UnsetPvsAir {
    fn width(&self) -> usize {
        UnsetPvsCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UnsetPvsAir {}
impl<F> PartitionedBaseAir<F> for UnsetPvsAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UnsetPvsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let next = main.row_slice(1).expect("window should have two elements");
        let local: &UnsetPvsCols<AB::Var> = (*local).borrow();
        let next: &UnsetPvsCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .assert_one(next.proof_idx - local.proof_idx);

        let air_idx = AB::F::from_usize(self.air_idx);

        for pv_idx in 0..self.num_pvs {
            self.public_values_bus.receive(
                builder,
                local.proof_idx,
                PublicValuesBusMessage {
                    air_idx,
                    pv_idx: AB::F::from_usize(pv_idx),
                    value: AB::F::ZERO,
                },
                local.is_valid,
            );
        }

        self.pvs_air_consistency_bus.lookup_key(
            builder,
            local.proof_idx,
            PvsAirConsistencyMessage {
                deferral_flag: AB::F::from_u32(self.def_flag),
                has_verifier_pvs: AB::F::ONE,
            },
            local.is_valid,
        );
    }
}
