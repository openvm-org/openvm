use super::{air::XorLimbsAir, columns::XorLimbsCols};
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

impl<const N: usize, const M: usize> XorLimbsAir<N, M> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: XorLimbsCols<N, M, AB::Var>,
    ) {
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
