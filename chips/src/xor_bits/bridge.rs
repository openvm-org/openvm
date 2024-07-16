use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::sub_chip::SubAirBridge;

use super::{columns::XorIoCols, XorBitsAir};

impl<AB: InteractionBuilder, const N: usize> SubAirBridge<AB> for XorBitsAir<N> {
    type View = XorIoCols<AB::Var>;

    fn eval_interactions(&self, builder: &mut AB, io: Self::View) {
        builder.push_receive(self.bus_index, vec![io.x, io.y, io.z], AB::F::one());
    }
}
