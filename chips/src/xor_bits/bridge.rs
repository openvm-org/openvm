use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::XorIoCols, XorBitsAir};

impl<const N: usize> XorBitsAir<N> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: XorIoCols<AB::Var>,
    ) {
        builder.push_receive(self.bus_index, vec![io.x, io.y, io.z], AB::F::one());
    }
}
