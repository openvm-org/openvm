use std::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_matrix::Matrix;
use recursion_circuit::bus::{PublicValuesBus, PublicValuesBusMessage};
use stark_recursion_circuit_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct UserPvsReceiverCols<F> {
    pub proof_idx: F,
    pub is_valid: F,
    pub pv_bus_msg: PublicValuesBusMessage<F>,
}

pub struct UserPvsReceiverAir {
    pub public_values_bus: PublicValuesBus,
}

impl<F> BaseAir<F> for UserPvsReceiverAir {
    fn width(&self) -> usize {
        UserPvsReceiverCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UserPvsReceiverAir {}
impl<F> PartitionedBaseAir<F> for UserPvsReceiverAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UserPvsReceiverAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let local: &UserPvsReceiverCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_valid);
        self.public_values_bus.receive(
            builder,
            local.proof_idx,
            local.pv_bus_msg.clone(),
            local.is_valid,
        );
    }
}
