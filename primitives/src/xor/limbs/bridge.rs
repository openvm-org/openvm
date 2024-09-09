use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_field::AbstractField;

use super::{air::XorLimbsAir, columns::XorLimbsCols};

impl<const N: usize, const M: usize> XorLimbsAir<N, M> {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: XorLimbsCols<N, M, AB::Var>,
    ) {
        // Send (x, y, z) where x and y have M bits.
        for (x, y, z) in izip!(local.x_limbs, local.y_limbs, local.z_limbs) {
            self.bus.send(x, y, z).eval(builder, AB::F::one());
        }

        // Receive (x, y, z) where x and y have N bits.
        self.bus
            .receive(local.x, local.y, local.z)
            .eval(builder, AB::F::one());
    }
}
