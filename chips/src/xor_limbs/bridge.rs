use crate::sub_chip::SubAirBridge;

use super::{air::XorLimbsAir, columns::XorLimbsCols};
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

impl<AB: InteractionBuilder, const N: usize, const M: usize> SubAirBridge<AB>
    for XorLimbsAir<N, M>
{
    type View = XorLimbsCols<N, M, AB::Var>;

    fn eval_interactions(&self, builder: &mut AB, local: Self::View) {
        // Send (x, y, z) where x and y have M bits.
        for (x, y, z) in izip!(local.x_limbs, local.y_limbs, local.z_limbs) {
            builder.push_send(self.bus_index, vec![x, y, z], AB::F::one());
        }

        // Receive (x, y, z) where x and y have N bits.
        builder.push_receive(
            self.bus_index,
            vec![local.x, local.y, local.z],
            AB::F::one(),
        );
    }
}
